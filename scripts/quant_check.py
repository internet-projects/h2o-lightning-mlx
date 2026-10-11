#!/usr/bin/env python3
"""Per checkpoint: load time, resident memory, 2,000-token read-out latency, and one image question (receipt).
    .venv/bin/python scripts/quant_check.py model mlx-q8 mlx-q4"""
import base64, json, os, resource, subprocess, sys, time
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "model"))
if len(sys.argv) > 2:          # one process per checkpoint, so memory numbers are clean
    for m in sys.argv[1:]:
        subprocess.run([sys.executable, __file__, m]); 
    sys.exit(0)
import mlx.core as mx
import mlx_vllm_facade as F
import h2o_lightning_shim as S
m = sys.argv[1]
cfg = json.load(open(os.path.join(HERE, "model", "serve_config.json")))
C = S.Contract(cfg, env={})
eng = F.Engine(os.path.join(HERE, m))
t = time.perf_counter(); eng.ensure_loaded(); load_s = time.perf_counter() - t
src = os.path.join(HERE, "bench", "page.txt")
txt = open(src).read() if os.path.exists(src) else ""
if len(txt) < 20000:
    txt = open(os.path.join(HERE, "model", "README.md")).read()
ids = eng.encode(txt)[:2000]
assert len(ids) == 2000, len(ids)
eng.logprobs_at_next(ids, list(range(8)))
xs = []
for _ in range(10):
    t = time.perf_counter(); eng.logprobs_at_next(ids, list(range(8))); xs.append(time.perf_counter() - t)
xs.sort()
img = "data:image/png;base64," + base64.b64encode(open(os.path.join(HERE, "model", "example_receipt.png"), "rb").read()).decode()
q = {"type": "choice", "instructions": "How many item lines are on the receipt?", "criteria": {"three": "3", "four": "4", "five": "5", "six": "6"}}
msgs, _, names, _, labels = C.image_prompt(S.render_state({"receipt": "[image 1]"}, True), q, [img])
text, urls = eng.render_chat(msgs, False, True, {"enable_thinking": False})
probe = {"type": "choice", "instructions": "Which?", "criteria": {f"o{i}": f"d{i}" for i in range(4)}}
base_t = C.question_prompt(S.render_state({"x": 1}, True), probe)[0]
lid = [eng.encode(base_t + " " + l)[-1] for l in labels]
iids, pv, grid = eng.process_images(text, eng.load_images(urls), cfg["image"]["max_pixels"])
t = time.perf_counter(); lps, _, _ = eng.logprobs_at_next(iids, lid, pv, grid); img_s = time.perf_counter() - t
import math
v = [x / cfg["image"]["temperature"] for x in lps]; mm = max(v); e = [math.exp(x - mm) for x in v]; p = [x / sum(e) for x in e]
print(json.dumps({"checkpoint": m, "load_s": round(load_s, 2), "mlx_active_gb": round(mx.get_active_memory() / 2**30, 2),
                  "mlx_peak_gb": round(mx.get_peak_memory() / 2**30, 2), "prompt_tokens": len(ids), "readout_2k_p50_s": round(xs[5], 3),
                  "readout_2k_min_s": round(xs[0], 3), "image_s": round(img_s, 3),
                  "image_probs": dict(zip(names, [round(x, 4) for x in p]))}), flush=True)
