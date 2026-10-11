#!/usr/bin/env python3
"""Authenticated front door for H2O-Lightning-4B on a private network. Stdlib only.

    POST /v1/systemone            the H2O wire format, passed through to the shim unchanged
    POST /api/alpha/decisions     alias of /v1/systemone
    POST /v1/decisions            the gateway's existing single-question shape:
                                  {"model", "state", "question": {"type", "instructions", "criteria"}}
                                  -> {"model", "type", "confidence", "noul" | "choice"+"probabilities" |
                                      "score"+"probabilities", "input_tokens", "engine", "latency_ms"}
    GET  /v1/models               (auth)
    GET  /health                  (no auth; no secrets)

Every route except /health needs `Authorization: Bearer <key>` (or `x-api-key`). The key is H2O_API_KEY.
"""
from __future__ import annotations

import argparse
import hmac
import http.client
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SHIM = os.environ.get("H2O_SHIM", "127.0.0.1:8744")
ENGINE = os.environ.get("H2O_ENGINE", "h2oai/h2o-lightning-4b (MLX bf16, fp32 head)")
PUBLIC_ID = "local-decision-h2o"
ACCEPTED_MODELS = {PUBLIC_ID, "h2o-lightning-4b", "h2oai/h2o-lightning-4b", "local-decision"}
def load_key():
    """The key is H2O_API_KEY. (The internal build also reads a gateway .env; the public build does not.)"""
    return os.environ.get("H2O_API_KEY", "").strip()


API_KEY = load_key()
_local = threading.local()
MAX_BODY = int(os.environ.get("H2O_MAX_BODY", str(100 * 1024 * 1024)))
DECISION_LOG = os.environ.get("H2O_DECISION_LOG", "")   # dir: one jsonl per day of answer statistics (no text)
_log_lock = threading.Lock()


def log_answers(client, path, resp):
    """Append per-question answer statistics (type, option count, top probability) - never the record text."""
    if not DECISION_LOG or not isinstance(resp, dict):
        return
    try:
        usage = resp.get("usage") or {}
        rows = []
        for qid, a in (resp.get("answers") or {}).items():
            if a.get("type") == "noul":
                p = float(a["noul"]); top, n = max(p, 1 - p), 2
            else:
                pr = a.get("probabilities") or {}
                top, n = (max(pr.values()) if pr else None), len(pr)
            rows.append({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "client": client, "path": path, "type": a.get("type"),
                         "options": n, "top": top, "nq": len(resp.get("answers") or {}),
                         "input_tokens": usage.get("input_tokens"), "latency_ms": usage.get("latency_ms")})
        os.makedirs(DECISION_LOG, exist_ok=True)
        with _log_lock, open(os.path.join(DECISION_LOG, time.strftime("decisions-%Y-%m-%d.jsonl")), "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    except Exception as e:
        sys.stderr.write(f"front: decision log error {e}\n")


def shim_call(method, path, body=b"", ctype="application/json"):
    for attempt in (0, 1):
        c = getattr(_local, "c", None)
        if c is None or attempt:
            if c is not None:
                c.close()
            c = _local.c = http.client.HTTPConnection(SHIM, timeout=300)
        try:
            c.request(method, path, body=body or None, headers={"Content-Type": ctype} if body else {})
            r = c.getresponse()
            return r.status, r.read()
        except (http.client.HTTPException, OSError) as e:
            if attempt:
                return 502, json.dumps({"detail": f"shim unreachable: {type(e).__name__}: {e}"}).encode()


def to_systemone(body):
    q = body.get("question")
    if not isinstance(q, dict):
        raise ValueError("question must be an object with type, instructions and criteria")
    q = dict(q)
    if q.get("type", "choice") == "choice" and isinstance(q.get("criteria"), list):
        q["criteria"] = {str(x): str(x) for x in q["criteria"]}
    state = body.get("state", body.get("context", body.get("input")))
    if state is None:
        raise ValueError("Missing state (context/input are accepted aliases)")
    out = {"state": state, "questions": {"q": q}}
    for k in ("images", "image", "image_url", "image_urls"):
        if k in body:
            out[k] = body[k]
    return out


def from_systemone(resp, t0):
    a = resp["answers"]["q"]
    typ = a["type"]
    out = {"model": PUBLIC_ID, "type": typ}
    if typ == "noul":
        p = float(a["noul"])
        out.update(noul=p, confidence=max(p, 1 - p), probabilities={"true": p, "false": 1 - p})
        for k in ("threshold", "decision"):
            if k in a:
                out[k] = a[k]
    else:
        probs = a["probabilities"]
        out.update(confidence=max(probs.values()), probabilities=probs, h2o_confidence=a.get("confidence"))
        if typ == "choice":
            out["choice"] = a["choice"]
        else:
            out.update(score=a["score"], legend=a.get("legend"))
    usage = resp.get("usage") or {}
    out.update(input_tokens=usage.get("input_tokens", 0), engine=ENGINE,
               probability_source=resp.get("probability_source"),
               latency_ms=round((time.perf_counter() - t0) * 1000, 2))
    return out


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    disable_nagle_algorithm = True

    def log_message(self, fmt, *args):
        sys.stderr.write("front %s %s\n" % (self.client_address[0], fmt % args))

    def send(self, code, payload, ctype="application/json"):
        data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def authed(self):
        if not API_KEY:
            self.send(503, {"error": {"message": "no API key configured; refusing to serve",
                                      "type": "service_unavailable"}})
            return False
        h = self.headers.get("Authorization", "")
        got = h[7:].strip() if h.lower().startswith("bearer ") else (self.headers.get("x-api-key") or "").strip()
        if got and hmac.compare_digest(got.encode(), API_KEY.encode()):
            return True
        self.send(401, {"error": {"message": "Invalid or missing API key", "type": "authentication_error"}})
        return False

    def do_GET(self):
        p = self.path.split("?", 1)[0]
        if p in ("/health", "/healthz"):
            st, raw = shim_call("GET", "/health")
            ok = st == 200
            return self.send(200 if ok else 503, {"ok": ok, "model": PUBLIC_ID, "engine": ENGINE})
        if not self.authed():
            return
        if p == "/v1/models":
            return self.send(200, {"object": "list", "data": [{
                "id": PUBLIC_ID, "object": "model", "alias_for": "h2oai/h2o-lightning-4b",
                "endpoints": ["/v1/systemone", "/v1/decisions"], "capabilities": ["noul", "choice", "score"]}]})
        self.send(404, {"error": "not found"})

    def do_POST(self):
        p = self.path.split("?", 1)[0]
        if not self.authed():
            return
        if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
            self.close_connection = True
            return self.send(411, {"error": {"message": "send Content-Length", "type": "invalid_request_error"}})
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            self.close_connection = True
            return self.send(413, {"error": {"message": f"body over {MAX_BODY} bytes", "type": "invalid_request_error"}})
        raw = self.rfile.read(n)
        if p in ("/v1/systemone", "/api/alpha/decisions"):
            st, out = shim_call("POST", "/v1/systemone", raw, self.headers.get("Content-Type") or "application/json")
            if st == 200 and DECISION_LOG:
                try:
                    log_answers(self.client_address[0], p, json.loads(out))
                except ValueError:
                    pass
            return self.send(st, out)
        if p == "/v1/decisions":
            t0 = time.perf_counter()
            try:
                body = json.loads(raw or b"{}")
                if not isinstance(body, dict):
                    raise ValueError("request body must be a JSON object")
                if body.get("model", PUBLIC_ID) not in ACCEPTED_MODELS:
                    raise ValueError(f"Unknown decision model: {body.get('model')}")
                if "questions" in body:            # already the H2O shape: pass through
                    st, out = shim_call("POST", "/v1/systemone", raw)
                    return self.send(st, out)
                req = to_systemone(body)
            except ValueError as e:
                return self.send(400, {"error": {"message": str(e), "type": "invalid_request_error"}})
            st, out = shim_call("POST", "/v1/systemone", json.dumps(req).encode())
            if st != 200:
                try:
                    detail = json.loads(out).get("detail", out.decode()[:500])
                except Exception:
                    detail = out.decode("utf-8", "replace")[:500]
                code = 400 if st == 422 else st
                return self.send(code, {"error": {"message": detail, "type": "decision_backend_error"
                                                  if code >= 500 else "invalid_request_error"}})
            try:
                parsed = json.loads(out)
                log_answers(self.client_address[0], p, parsed)
                return self.send(200, from_systemone(parsed, t0))
            except (ValueError, KeyError, TypeError) as e:
                return self.send(502, {"error": {"message": f"unexpected shim reply: {e}",
                                                 "type": "decision_backend_error"}})
        self.send(404, {"error": "not found"})


class Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hosts", default=os.environ.get("H2O_HOSTS", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("H2O_PORT", "8742")))
    a = ap.parse_args()
    if not API_KEY:
        sys.stderr.write("front: WARNING no API key found; every request will get 503\n")
    servers = []
    for host in [h.strip() for h in a.hosts.split(",") if h.strip()]:
        try:
            servers.append(Server((host, a.port), H))
            sys.stderr.write(f"front: listening on http://{host}:{a.port} (shim {SHIM})\n")
        except OSError as e:          # e.g. network not up yet at boot: exit so launchd retries
            sys.stderr.write(f"front: cannot bind {host}:{a.port}: {e}\n")
            time.sleep(10)
            sys.exit(1)
    for s in servers[1:]:
        threading.Thread(target=s.serve_forever, daemon=True).start()
    servers[0].serve_forever()


if __name__ == "__main__":
    main()
