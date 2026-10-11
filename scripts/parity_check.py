#!/usr/bin/env python3
"""Parity checks for the MLX facade, before any benchmark.

1. Prompt/token parity: the shim's rendered prompt tokenises to exactly what the model's own chat template + HF
   tokenizer produce (text path), and the chat-messages path (image path without images) gives the same tokens.
2. Numerics: label log-probabilities from MLX vs a PyTorch reference (the official safetensors through transformers,
   bf16 body, float32 head) on N prompts built by the shim from JevBench public items; compares the probabilities
   at the shipped per-type temperatures and the top label.
3. Chunked prefill == single-pass prefill on a long prompt.
4. Image: one image prompt (example_receipt.png) through MLX vs the torch reference.

    .venv/bin/python parity_check.py --model model --jevbench jevbench --n 20 --out parity.json
"""
import argparse
import glob
import json
import math
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--jevbench", required=True)
ap.add_argument("--n", type=int, default=20)
ap.add_argument("--out", default="parity.json")
ap.add_argument("--device", default="mps")
a = ap.parse_args()
MODEL = os.path.abspath(os.path.expanduser(a.model))
sys.path.insert(0, MODEL)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import h2o_lightning_shim as S                      # noqa: E402
import mlx_vllm_facade as F                        # noqa: E402

cfg = json.load(open(os.path.join(MODEL, "serve_config.json")))
C = S.Contract(cfg, env={})
eng = F.Engine(MODEL)
report = {"model": MODEL}


def label_ids():
    probe = {"type": "choice", "instructions": "Which?", "criteria": {f"o{i}": f"d{i}" for i in range(len(C.labels))}}
    text = C.question_prompt(S.render_state({"x": 1}, C.state_compact), probe)[0]
    base = eng.encode(text)
    ids = {}
    for l in C.labels:
        full = eng.encode(text + " " + l)
        assert len(full) == len(base) + 1 and full[:len(base)] == base, f"label {l!r} is not one token"
        ids[l] = full[-1]
    return ids


LID = label_ids()
report["labels_one_token"] = len(LID)


# ---------------------------------------------------------------- items from JevBench's public set
def load_items():
    files = sorted(glob.glob(os.path.join(a.jevbench, "datasets", "public", "*.jsonl")))
    items = []
    for f in files:
        for line in open(f):
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def to_request(item):
    """JevBench items carry the typesafe request; find {state, questions}."""
    for k in ("request", "input", "payload"):
        if isinstance(item.get(k), dict) and "questions" in item[k]:
            return item[k]
    if "questions" in item and "state" in item:
        return item
    if "question" in item and "state" in item:
        return {"state": item["state"], "questions": {"q": item["question"]}}
    raise KeyError(f"no request in item keys {list(item)[:10]}")


items = load_items()
step = max(1, len(items) // a.n)
picked = items[::step][:a.n]
prompts = []
for it in picked:
    req = to_request(it)
    st = S.render_state(req["state"], C.state_compact)
    for qid, q in req["questions"].items():
        text, names, descs, labels = C.question_prompt(st, q, qid)
        prompts.append({"id": it.get("id", "?"), "type": q.get("type", "choice"), "text": text,
                        "ids": [LID[l] for l in labels]})
        break

# ---------------------------------------------------------------- 1. token parity
tok_ok = 0
for p in prompts[:5]:
    # the shim's text prompt vs the HF chat template for the same messages (+ prefill continued)
    sysmsg = C.system
    user = p["text"].split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0]
    msgs = [{"role": "system", "content": sysmsg}, {"role": "user", "content": user},
            {"role": "assistant", "content": C.prefill}]
    t_chat, _ = eng.render_chat(msgs, False, True, {"enable_thinking": False})
    tok_ok += int(eng.encode(t_chat) == eng.encode(p["text"]))
report["token_parity_chat_template"] = f"{tok_ok}/{min(5, len(prompts))}"

# ---------------------------------------------------------------- 2. MLX read-out
eng.ensure_loaded()
mlx_res = []
t0 = time.time()
for p in prompts:
    ids = eng.encode(p["text"])
    lps, top, _ = eng.logprobs_at_next(ids, p["ids"])
    mlx_res.append({"n_tokens": len(ids), "lps": lps, "argmax_vocab": top})
report["mlx_seconds"] = round(time.time() - t0, 2)

# 3. chunked vs single prefill on the longest prompt (and a synthetic long one)
longp = max(prompts, key=lambda p: len(p["text"]))
ids = eng.encode(longp["text"])
F.PREFILL_CHUNK = 64
lp_chunk, _, _ = eng.logprobs_at_next(ids, longp["ids"])
F.PREFILL_CHUNK = 1 << 20
lp_single, _, _ = eng.logprobs_at_next(ids, longp["ids"])
F.PREFILL_CHUNK = 4096
report["chunked_vs_single_max_abs_logprob_diff"] = max(abs(x - y) for x, y in zip(lp_chunk, lp_single))

# image via MLX
img_path = os.path.join(MODEL, "example_receipt.png")
import base64  # noqa: E402
uri = "data:image/png;base64," + base64.b64encode(open(img_path, "rb").read()).decode()
q_img = {"type": "choice", "instructions": "How many item lines are on the receipt?",
         "criteria": {"three": "3", "four": "4", "five": "5", "six": "6"}}
msgs_img, _, names_i, _, labels_i = C.image_prompt(S.render_state({"receipt": "[image 1]"}, True), q_img, [uri])
t_img, urls = eng.render_chat(msgs_img, False, True, {"enable_thinking": False})
ids_img, pv, grid = eng.process_images(t_img, eng.load_images(urls), cfg["image"]["max_pixels"])
mlx_img, _, _ = eng.logprobs_at_next(ids_img, [LID[l] for l in labels_i], pv, grid)
report["image_tokens"] = len(ids_img)
report["image_grid_thw"] = [int(x) for x in grid.tolist()[0]]

# release MLX before loading torch
eng.model = None
import gc  # noqa: E402
gc.collect()
F.mx.clear_cache()

# ---------------------------------------------------------------- torch reference
import torch  # noqa: E402
from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: E402
dev = a.device if (a.device != "mps" or torch.backends.mps.is_available()) else "cpu"
tm = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev).eval()
proc = AutoProcessor.from_pretrained(MODEL)
W = tm.get_output_embeddings().weight


def torch_lps(input_ids, wanted, **mm):
    with torch.no_grad():
        out = tm.model(input_ids=input_ids.to(dev), **{k: v.to(dev) for k, v in mm.items()})
        h = out.last_hidden_state[0, -1].float()
        logits = h @ W.float().T
        lp = torch.log_softmax(logits, -1)
        return [float(lp[i]) for i in wanted], int(lp.argmax())


temps = cfg["temperature"]


def probs(lps, t):
    v = [x / t for x in lps]
    m = max(v)
    e = [math.exp(x - m) for x in v]
    s = sum(e)
    return [x / s for x in e]


rows = []
t0 = time.time()
for p, r in zip(prompts, mlx_res):
    ids = torch.tensor([eng.encode(p["text"])])
    tl, targ = torch_lps(ids, p["ids"])
    t = temps[p["type"]] if isinstance(temps, dict) else temps
    pm, pt = probs(r["lps"], t), probs(tl, t)
    rows.append({"id": p["id"], "type": p["type"], "n_tokens": r["n_tokens"],
                 "max_abs_prob_diff": max(abs(x - y) for x, y in zip(pm, pt)),
                 "top_agree": pm.index(max(pm)) == pt.index(max(pt)),
                 "vocab_argmax_agree": r["argmax_vocab"] == targ,
                 "mlx_top_p": max(pm), "torch_top_p": max(pt)})
report["torch_seconds"] = round(time.time() - t0, 2)
report["torch_device"] = dev
report["rows"] = rows
report["summary"] = {
    "n": len(rows),
    "top_label_agree": sum(r["top_agree"] for r in rows),
    "max_abs_prob_diff_max": max(r["max_abs_prob_diff"] for r in rows),
    "max_abs_prob_diff_mean": sum(r["max_abs_prob_diff"] for r in rows) / len(rows),
}

# image reference: a separate process with plain transformers (HF's own image processor, no mlx_vlm loaded)
import subprocess  # noqa: E402
json.dump(report, open(a.out, "w"), indent=1)          # text results saved before the image step
del tm
gc.collect()
ref = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "torch_image_ref.py")],
                     input=json.dumps({"model": MODEL, "text": t_img, "image": img_path, "device": a.device,
                                       "max_pixels": cfg["image"]["max_pixels"], "ids": [LID[l] for l in labels_i]}),
                     capture_output=True, text=True)
if ref.returncode:
    print(ref.stderr[-3000:])
refj = json.loads(ref.stdout.strip().splitlines()[-1])
ti = refj["lps"]
pm, pt = probs(mlx_img, cfg["image"]["temperature"]), probs(ti, cfg["image"]["temperature"])
report["image"] = {"names": names_i, "mlx": [round(x, 4) for x in pm], "torch": [round(x, 4) for x in pt],
                   "same_input_ids": refj["input_ids"] == ids_img, "torch_grid": refj["grid"], "torch_processor": refj["processor"],
                   "max_abs_prob_diff": max(abs(x - y) for x, y in zip(pm, pt))}
json.dump(report, open(a.out, "w"), indent=1)
print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=1))
