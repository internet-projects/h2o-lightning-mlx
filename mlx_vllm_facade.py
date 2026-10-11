#!/usr/bin/env python3
"""A vLLM stand-in on Apple Silicon (MLX) for H2O-Lightning-4B, so the official h2o_lightning_shim.py runs UNMODIFIED.

It implements only what the shim calls on vLLM:
    GET  /v1/models            -> the served model name and max_model_len
    POST /tokenize             -> {"prompt", "add_special_tokens"} or {"messages", "continue_final_message", ...}
    POST /detokenize           -> {"tokens"}
    POST /v1/completions       -> max_tokens=1 read-out: log-probabilities of `logprob_token_ids` at the next slot
    POST /v1/chat/completions  -> the same read-out for a chat with images (continue_final_message)
    GET  /health

Nothing is ever sampled. One forward pass per request; the hidden state at the last position goes through the tied
output head in float32 (the model's config.json asks for "head_dtype": "float32", which vLLM honours), and the reply
carries log_softmax over the whole vocabulary for the requested ids -- vLLM's raw log-probabilities.

The weights are the official bf16 safetensors, loaded by mlx-vlm without quantization.
MLX work is serialised by one lock. The model loads on first use and is released after IDLE_UNLOAD_S of no requests
(0 = keep resident).

    python mlx_vllm_facade.py --model ./model --port 8743
"""
from __future__ import annotations

import argparse
import base64
import gc
import io
import json
import math
import os
import re
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import mlx.core as mx

SERVED_NAME = os.environ.get("FACADE_SERVED_NAME", "h2oai/h2o-lightning-4b")
MAX_MODEL_LEN = int(os.environ.get("FACADE_MAX_MODEL_LEN", "40960"))
PREFILL_CHUNK = int(os.environ.get("FACADE_PREFILL_CHUNK", "4096"))
IDLE_UNLOAD_S = float(os.environ.get("FACADE_IDLE_UNLOAD_S", "900"))
DEFAULT_MAX_PIXELS = int(os.environ.get("FACADE_MAX_PIXELS", "1605632"))     # vLLM --mm-processor-kwargs in serve.sh
MAX_IMAGES = int(os.environ.get("FACADE_MAX_IMAGES", "4"))
HEAD_SLICE = int(os.environ.get("FACADE_HEAD_SLICE", "32768"))
# MLX keeps freed buffers in a cache; bound it so the resident footprint stays near the weights' size
CACHE_LIMIT_GB = float(os.environ.get("FACADE_CACHE_LIMIT_GB", "1"))
PREFIX_MIN = int(os.environ.get("FACADE_PREFIX_MIN", "256"))     # 0 turns the prefix cache off
CONTEXT_MSG = "This model's maximum context length is {m} tokens. However, you requested {n} tokens."
_DATA_URI = re.compile(r"data:(image/[A-Za-z0-9.+-]+);base64,(.*)", re.S | re.I)


class HTTPError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


class Engine:
    def __init__(self, path):
        self.path = os.path.expanduser(path)
        self.lock = threading.RLock()
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.last_used = time.time()
        self.loads = 0
        self.snapshot = None          # (prefix ids, saved cache state) for the prefix cache
        self.last_ids = None
        self.prefix_hits = 0
        cfg = json.load(open(os.path.join(self.path, "config.json")))
        self.head_fp32 = cfg.get("head_dtype", cfg.get("text_config", {}).get("head_dtype")) == "float32"
        self.image_token_id = cfg.get("image_token_id")
        # the tokenizer is cheap and needed for /tokenize even while the weights are released
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.path)

    # ------------------------------------------------------------------ lifecycle
    def ensure_loaded(self):
        if self.model is not None:
            return
        from mlx_vlm import load
        t0 = time.time()
        model, processor = load(self.path)
        self.model, self.processor = model, processor
        self.loads += 1
        lm = self.model.language_model
        self.embed = lm.model.embed_tokens
        self.lm_head = getattr(lm, "lm_head", None)      # None when the embeddings are tied (this model)
        sys.stderr.write(f"facade: loaded {self.path} in {time.time() - t0:.1f}s "
                         f"(dtype {self.embed.weight.dtype}, head fp32={self.head_fp32}, tied={self.lm_head is None})\n")

    def maybe_unload(self):
        if not IDLE_UNLOAD_S or self.model is None:
            return
        with self.lock:
            if self.model is not None and time.time() - self.last_used > IDLE_UNLOAD_S:
                self.model = self.processor = self.embed = self.lm_head = None
                self.snapshot = self.last_ids = None
                gc.collect()
                mx.clear_cache()
                sys.stderr.write("facade: idle, weights released\n")

    # ------------------------------------------------------------------ text helpers
    def encode(self, text, add_special_tokens=False):
        return self.tokenizer.encode(text, add_special_tokens=bool(add_special_tokens))

    def render_chat(self, messages, add_generation_prompt=False, continue_final_message=False, kwargs=None):
        """Chat messages -> prompt text through the model's own chat template; image parts become the template's
        image slot (as vLLM does before its processor expands the pad)."""
        msgs, urls = [], []
        for m in messages:
            c = m.get("content")
            if isinstance(c, list):
                parts = []
                for p in c:
                    t = p.get("type")
                    if t in ("image_url", "image", "input_image"):
                        u = p.get("image_url") if t != "image" else p.get("image")
                        u = u.get("url") if isinstance(u, dict) else u
                        urls.append(u)
                        parts.append({"type": "image"})
                    elif t == "text":
                        parts.append({"type": "text", "text": p.get("text", "")})
                    else:
                        raise HTTPError(400, f"unsupported content part {t!r}")
                msgs.append({"role": m["role"], "content": parts})
            else:
                msgs.append({"role": m["role"], "content": c})
        text = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message, **(kwargs or {}))
        return text, urls

    # ------------------------------------------------------------------ the read-out
    def _final_hidden(self, ids, pixel_values=None, grid=None):
        """Normalised hidden state at the last position, shape (hidden,)."""
        lm = self.model.language_model
        input_ids = mx.array([ids], dtype=mx.int32)
        if pixel_values is not None:
            feats = self.model.get_input_embeddings(input_ids, pixel_values=pixel_values, image_grid_thw=grid)
        else:
            feats = self.model.get_input_embeddings(input_ids)
        emb, pos = feats.inputs_embeds, feats.position_ids
        if pos is None:
            pos = mx.arange(len(ids), dtype=mx.int32)[None]
        if pos.ndim == 2:                                   # (B, L) -> the (3, B, L) M-RoPE layout
            pos = mx.broadcast_to(pos[None], (3,) + tuple(pos.shape))
        n = len(ids)
        start, cache, split = 0, None, None
        if pixel_values is None and PREFIX_MIN:
            # PREFIX CACHE (text only). Several questions about one record share a long head; vLLM computes it once
            # (prefix caching). Here: if this prompt shares >= PREFIX_MIN tokens with the saved snapshot, restore it;
            # otherwise, if it shares >= PREFIX_MIN tokens with the previous prompt, snapshot at that point this time.
            snap = self.snapshot
            if snap and n > len(snap[0]) and tuple(ids[:len(snap[0])]) == snap[0]:
                start, cache = len(snap[0]), self._restore(snap[1])
                self.prefix_hits += 1
            else:
                prev = self.last_ids
                if prev:
                    m = min(len(prev), n - 1)
                    k = next((i for i in range(m) if prev[i] != ids[i]), m)
                    if k >= PREFIX_MIN:
                        split = k
        if cache is None:
            cache = lm.make_cache()
        h = None
        bounds = [b for b in range(start, n, PREFILL_CHUNK)] + [n]
        if split:
            bounds = sorted(set(bounds) | {split})
        for s, e in zip(bounds[:-1], bounds[1:]):
            h = lm.model(input_ids[:, s:e], inputs_embeds=emb[:, s:e], cache=cache, position_ids=pos[..., s:e])
            mx.eval(h)
            if split and e == split:
                self.snapshot = (tuple(ids[:split]), self._save(cache))
        if pixel_values is None:
            self.last_ids = list(ids)
        return h[0, -1]

    @staticmethod
    def _copy(a):
        return None if a is None else (a + 0 if a.dtype != mx.bool_ else mx.array(a))

    def _save(self, cache):
        st = [[self._copy(a) for a in c.state] for c in cache]
        mx.eval([a for s in st for a in s if a is not None])
        return st

    def _restore(self, saved):
        cache = self.model.language_model.make_cache()
        for c, st in zip(cache, saved):
            c.state = [self._copy(a) for a in st]
        return cache

    def logprobs_at_next(self, ids, wanted, pixel_values=None, grid=None):
        if len(ids) + 1 > MAX_MODEL_LEN:
            raise HTTPError(400, CONTEXT_MSG.format(m=MAX_MODEL_LEN, n=len(ids) + 1))
        h = self._final_hidden(ids, pixel_values, grid)
        head = self.embed if self.lm_head is None else self.lm_head
        W = head.weight
        q = getattr(head, "scales", None) is not None          # a quantized (q8/q4) MLX checkpoint
        if self.head_fp32:
            # float32 head, in vocabulary slices so no 2.5 GB float32 copy of the head is ever held;
            # a quantized head is dequantized slice by slice first (same values the quantized matmul would use)
            hf = h.astype(mx.float32)
            parts = []
            for s in range(0, W.shape[0], HEAD_SLICE):
                if q:
                    b = getattr(head, "biases", None)
                    w = mx.dequantize(W[s:s + HEAD_SLICE], head.scales[s:s + HEAD_SLICE],
                                      None if b is None else b[s:s + HEAD_SLICE],
                                      group_size=head.group_size, bits=head.bits,
                                      **({"mode": head.mode} if getattr(head, "mode", None) else {}))
                else:
                    w = W[s:s + HEAD_SLICE]
                parts.append(w.astype(mx.float32) @ hf)
                mx.eval(parts[-1])
            logits = mx.concatenate(parts)
        else:
            logits = (head.as_linear(h) if self.lm_head is None else head(h)).astype(mx.float32)
        lp = logits - mx.logsumexp(logits)
        sel = lp[mx.array(wanted, dtype=mx.int32)]
        top = int(mx.argmax(lp).item())
        mx.eval(sel)
        return [float(x) for x in sel.tolist()], top, float(lp[top].item())

    # ------------------------------------------------------------------ images
    @staticmethod
    def load_image(u):
        from PIL import Image
        if not isinstance(u, str):
            raise HTTPError(422, "image must be a URL or data URI")
        m = _DATA_URI.match(u.strip())
        try:
            if m:
                raw = base64.b64decode(m.group(2), validate=False)
            elif u.startswith(("http://", "https://")):
                with urllib.request.urlopen(u, timeout=10) as r:
                    raw = r.read(25 * 1024 * 1024)
            else:
                raw = base64.b64decode(u)
            im = Image.open(io.BytesIO(raw))
            im.load()
            return im.convert("RGB")
        except HTTPError:
            raise
        except Exception as e:
            raise HTTPError(422, f"cannot load image: {type(e).__name__}: {e}")

    def load_images(self, urls):
        """Fetched/decoded OUTSIDE the MLX lock, so a slow URL never stalls other requests."""
        if len(urls) > MAX_IMAGES:
            raise HTTPError(422, f"at most {MAX_IMAGES} images per prompt")
        return [self.load_image(u) for u in urls]

    def process_images(self, text, images, max_pixels):
        ip = self.processor.image_processor
        min_px = (getattr(ip, "size", None) or {}).get("shortest_edge", 65536)
        try:
            out = self.processor(text=[text], images=images, return_tensors="np",
                                 size={"longest_edge": int(max_pixels), "shortest_edge": int(min_px)})
        except Exception as e:
            raise HTTPError(422, f"image preprocessing failed: {type(e).__name__}: {e}")
        ids = [int(x) for x in out["input_ids"][0]]
        return ids, mx.array(out["pixel_values"]), mx.array(out["image_grid_thw"])

    # ------------------------------------------------------------------ requests
    def handle_completions(self, body):
        prompt = body.get("prompt")
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], int):
            ids = prompt
        elif isinstance(prompt, str):
            ids = self.encode(prompt, body.get("add_special_tokens", True))
        else:
            raise HTTPError(400, "prompt must be a string or a list of token ids")
        if int(body.get("max_tokens", 16)) != 1:
            raise HTTPError(400, "this server only reads the next token (max_tokens must be 1)")
        wanted = [int(i) for i in (body.get("logprob_token_ids") or [])]
        with self.lock:
            self.ensure_loaded()
            self.last_used = time.time()
            lps, top, top_lp = self.logprobs_at_next(ids, wanted)
            self.last_used = time.time()
        top_map = {f"token_id:{i}": v for i, v in zip(wanted, lps)}
        return {"id": "cmpl-mlx", "object": "text_completion", "created": int(time.time()), "model": SERVED_NAME,
                "choices": [{"index": 0, "text": self.tokenizer.decode([top]), "finish_reason": "length",
                             "logprobs": {"tokens": [f"token_id:{top}"], "token_logprobs": [top_lp],
                                          "top_logprobs": [top_map], "text_offset": [0]}}],
                "usage": {"prompt_tokens": len(ids), "completion_tokens": 1, "total_tokens": len(ids) + 1}}

    def handle_chat(self, body):
        if int(body.get("max_completion_tokens", body.get("max_tokens", 1)) or 1) != 1:
            raise HTTPError(400, "this server only reads the next token (max_completion_tokens must be 1)")
        cont = bool(body.get("continue_final_message", False))
        text, urls = self.render_chat(body.get("messages") or [], body.get("add_generation_prompt", not cont),
                                      cont, body.get("chat_template_kwargs"))
        wanted = [int(i) for i in (body.get("logprob_token_ids") or [])]
        max_px = int((body.get("mm_processor_kwargs") or {}).get("max_pixels") or DEFAULT_MAX_PIXELS)
        images = self.load_images(urls) if urls else []
        with self.lock:
            self.ensure_loaded()
            self.last_used = time.time()
            if urls:
                ids, pv, grid = self.process_images(text, images, max_px)
                lps, top, top_lp = self.logprobs_at_next(ids, wanted, pv, grid)
            else:
                ids = self.encode(text, body.get("add_special_tokens", False))
                lps, top, top_lp = self.logprobs_at_next(ids, wanted)
            self.last_used = time.time()
        content = [{"token": f"token_id:{top}", "logprob": top_lp,
                    "top_logprobs": [{"token": f"token_id:{i}", "logprob": v} for i, v in zip(wanted, lps)]}]
        return {"id": "chatcmpl-mlx", "object": "chat.completion", "created": int(time.time()), "model": SERVED_NAME,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": self.tokenizer.decode([top])},
                             "logprobs": {"content": content}, "finish_reason": "length"}],
                "usage": {"prompt_tokens": len(ids), "completion_tokens": 1, "total_tokens": len(ids) + 1}}

    def handle_tokenize(self, body):
        if "messages" in body:
            cont = bool(body.get("continue_final_message", False))
            text, _ = self.render_chat(body["messages"], body.get("add_generation_prompt", not cont),
                                       cont, body.get("chat_template_kwargs"))
            ids = self.encode(text, body.get("add_special_tokens", False))
        else:
            ids = self.encode(body.get("prompt", ""), body.get("add_special_tokens", True))
        return {"tokens": ids, "count": len(ids), "max_model_len": MAX_MODEL_LEN}


def make_handler(engine):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        disable_nagle_algorithm = True

        def log_message(self, fmt, *args):
            if os.environ.get("FACADE_ACCESS_LOG") == "1":
                sys.stderr.write("facade %s\n" % (fmt % args))

        def send(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            p = self.path.split("?", 1)[0]
            if p in ("/health", "/healthz"):
                return self.send(200, {"ok": True, "model": SERVED_NAME, "loaded": engine.model is not None,
                                       "loads": engine.loads, "backend": "mlx", "head_fp32": engine.head_fp32,
                                       "prefix_hits": engine.prefix_hits, "prefix_min": PREFIX_MIN})
            if p == "/v1/models":
                return self.send(200, {"object": "list", "data": [{
                    "id": SERVED_NAME, "object": "model", "owned_by": "mlx-facade", "max_model_len": MAX_MODEL_LEN}]})
            self.send(404, {"error": "not found"})

        def do_POST(self):
            p = self.path.split("?", 1)[0]
            if self.headers.get("Origin") or not (self.headers.get("Content-Type") or "").startswith("application/json"):
                return self.send(403, {"object": "error", "message": "local JSON clients only", "code": 403})
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                if body.get("model", SERVED_NAME) != SERVED_NAME:
                    raise HTTPError(404, f"The model `{body.get('model')}` does not exist.")
                if p == "/tokenize":
                    return self.send(200, engine.handle_tokenize(body))
                if p == "/detokenize":
                    return self.send(200, {"prompt": engine.tokenizer.decode(body.get("tokens") or [])})
                if p == "/v1/completions":
                    return self.send(200, engine.handle_completions(body))
                if p == "/v1/chat/completions":
                    return self.send(200, engine.handle_chat(body))
                self.send(404, {"error": "not found"})
            except HTTPError as e:
                self.send(e.status, {"object": "error", "message": str(e), "code": e.status})
            except json.JSONDecodeError as e:
                self.send(400, {"object": "error", "message": f"invalid JSON: {e}", "code": 400})
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send(500, {"object": "error", "message": f"{type(e).__name__}: {e}", "code": 500})
    return H


class Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser(os.environ.get("H2O_MODEL_DIR", "./model")))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8743)
    ap.add_argument("--preload", action="store_true")
    a = ap.parse_args()
    mx.set_cache_limit(int(CACHE_LIMIT_GB * (1 << 30)))
    engine = Engine(a.model)
    if a.preload:
        with engine.lock:
            engine.ensure_loaded()

    def reaper():
        while True:
            time.sleep(30)
            try:
                engine.maybe_unload()
            except Exception as e:
                sys.stderr.write(f"facade: unload error {e}\n")
    threading.Thread(target=reaper, daemon=True).start()
    sys.stderr.write(f"facade: {SERVED_NAME} on http://{a.host}:{a.port} (mlx; idle unload {IDLE_UNLOAD_S:.0f}s)\n")
    Server((a.host, a.port), make_handler(engine)).serve_forever()


if __name__ == "__main__":
    main()
