#!/usr/bin/env python3
"""Compare a quantized JevBench run with the bf16 run item by item: score, top-answer agreement, probability drift."""
import json, sys
def load(d): return {r["task_id"]: r for r in map(json.loads, open(d + "/results.jsonl")) if r}
ref, q = load(sys.argv[1]), load(sys.argv[2])
agree, diffs, flips = 0, [], []
for k, r in ref.items():
    s = q.get(k)
    if not s or not r.get("probs") or not s.get("probs"): continue
    agree += r["predicted"] == s["predicted"]
    d = max(abs(r["probs"][o] - s["probs"].get(o, 0)) for o in r["probs"]); diffs.append(d)
    if r["predicted"] != s["predicted"]: flips.append((k, r["correct"], s["correct"], round(d, 3)))
diffs.sort()
print(json.dumps({"ref_correct": sum(r["correct"] for r in ref.values()), "quant_correct": sum(r["correct"] for r in q.values()),
  "n": len(diffs), "same_answer": agree, "prob_drift_mean": round(sum(diffs)/len(diffs), 4),
  "prob_drift_p95": round(diffs[int(.95*len(diffs))], 4), "prob_drift_max": round(diffs[-1], 4), "flips": flips}, indent=1))
