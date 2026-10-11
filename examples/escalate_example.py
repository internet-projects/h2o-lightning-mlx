#!/usr/bin/env python3
"""Example: route low-confidence H2O answers to a second judge.

H2O answers in one pass through h2o_front.py (/v1/decisions). If its top probability is below THRESHOLD, this asks a
second judge (any OpenAI-compatible chat endpoint) for probabilities over the same options. The second answer replaces
H2O's only when that judge is confident (TRUST). Otherwise H2O's answer is kept and marked uncertain.

This is a template, not a drop-in service. Set the environment variables, adapt `ask_second_judge` to your judge, and
choose thresholds on your own labelled data. The defaults below are the ones we use on our claim-checking traffic.

    H2O_URL=http://127.0.0.1:8742/v1/decisions H2O_API_KEY=... \
    SECOND_URL=http://127.0.0.1:8000/v1 SECOND_KEY=... SECOND_MODEL=my-judge \
    python examples/escalate_example.py
"""
import json
import os
import urllib.request

H2O_URL = os.environ.get("H2O_URL", "http://127.0.0.1:8742/v1/decisions")
H2O_API_KEY = os.environ.get("H2O_API_KEY", "")
SECOND_URL = os.environ.get("SECOND_URL", "")
SECOND_KEY = os.environ.get("SECOND_KEY", "")
SECOND_MODEL = os.environ.get("SECOND_MODEL", "")
THRESHOLD = float(os.environ.get("ESC_THRESHOLD", "0.80"))   # escalate when H2O's top probability is below this
TRUST = float(os.environ.get("ESC_TRUST", "0.80"))           # adopt the second judge only at or above this


def post(url, body, key=""):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {key}"} if key else {})})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def answer_of(d):
    if d.get("type") == "noul":
        return "true" if d.get("noul", 0.5) >= 0.5 else "false"
    if d.get("type") == "choice":
        return d.get("choice")
    return d.get("score")


def ask_second_judge(state, question, options):
    """Replace this with a call to your own judge. It must return {option_id: probability}, summing to about 1."""
    if not SECOND_URL:
        raise RuntimeError("no second judge configured")
    listing = "\n".join(f"- {k}: {v}" for k, v in options.items())
    prompt = (f"Evidence:\n{state}\n\nQuestion: {question.get('instructions', '')}\n\nOptions:\n{listing}\n\n"
              'Reply with JSON only: {"probabilities": {"<option id>": <p>, ...}}')
    out = post(SECOND_URL.rstrip("/") + "/chat/completions", {
        "model": SECOND_MODEL, "temperature": 0, "max_tokens": 1200,
        "messages": [{"role": "user", "content": prompt}]}, SECOND_KEY)
    text = out["choices"][0]["message"]["content"]
    text = text[text.index("{"): text.rindex("}") + 1]
    return json.loads(text)["probabilities"]


def decide(state, question):
    h2o = post(H2O_URL, {"model": "local-decision", "state": state, "question": question}, H2O_API_KEY)
    h2o["uncertain"] = h2o["confidence"] < THRESHOLD
    if not h2o["uncertain"]:
        return h2o
    kind = question["type"]
    options = question.get("criteria") or {}
    try:
        p2 = ask_second_judge(state, question, options)
    except Exception as e:                       # no second opinion: keep H2O, flag it
        h2o["escalation"] = f"skipped: {type(e).__name__}"
        return h2o
    total = sum(p2.values()) or 1.0
    p2 = {k: v / total for k, v in p2.items()}
    c2 = max(p2.values())
    a1 = answer_of(h2o)
    a2 = max(p2, key=p2.get) if kind == "choice" else ("true" if p2.get("true", 0) >= 0.5 else "false")
    adopt = c2 >= TRUST
    h2o.update(escalated=True, uncertain=not adopt,
               second={"answer": a2, "confidence": round(c2, 4), "agree": a1 == a2, "probabilities": p2})
    if adopt and kind == "choice":
        h2o.update(choice=a2, probabilities=p2, confidence=c2)
    return h2o


if __name__ == "__main__":
    demo_state = "Ticket 4471: checkout returns HTTP 500 for every customer since 09:10."
    demo_q = {"type": "choice", "instructions": "Which priority?",
              "criteria": {"p1": "Revenue path fully down", "p2": "Degraded but working", "p3": "Cosmetic"}}
    print(json.dumps(decide(demo_state, demo_q), indent=2))
