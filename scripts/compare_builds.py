#!/usr/bin/env python3
"""One table over every bench run in bench/: score, agreement and probability drift vs bf16 (same mode),
two-way probe distance from H2O's recorded answers, size, median latency. Writes bench/compare.json too."""
import glob, json, os, re, statistics
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(HERE)
runs = {}
for d in sorted(glob.glob("bench/jevbench-*")):
    m = re.match(r"bench/jevbench-(.+?)-(one-way|two-way)(-unverified)?-\d{8}-\d{4}$", d)
    if not m or not os.path.exists(f"{d}/results.jsonl"):
        continue
    rows = {json.loads(l)["task_id"]: json.loads(l) for l in open(f"{d}/results.jsonl") if l.strip()}
    log = open(f"{d}/shim.log").read() if os.path.exists(f"{d}/shim.log") else ""
    dp = re.search(r"max \|dp\| ([0-9.]+) from the release", log)
    runs[(m.group(1), m.group(2))] = (rows, float(dp.group(1)) if dp else None)   # later runs replace earlier ones
def size(label):
    p = "model" if label == "bf16" else f"mlx-{label}"
    return sum(os.path.getsize(f) for f in glob.glob(f"{p}/*.safetensors")) / 1e9 if os.path.isdir(p) else None
out = []
for (label, mode), (rows, dp) in sorted(runs.items(), key=lambda x: (x[0][1], -(size(x[0][0]) or 0))):
    ref = runs.get(("bf16", mode), (None,))[0]
    r = {"build": label, "mode": mode, "correct": sum(x["correct"] for x in rows.values()), "n": len(rows),
         "size_gb": round(size(label), 2) if size(label) else None, "probe_dp": dp,
         "p50_s": round(statistics.median(x["latency_s"] for x in rows.values()), 3)}
    if ref and label != "bf16":
        diffs = sorted(abs(rows[k]["probs"][o] - ref[k]["probs"].get(o, 0)) for k in rows if k in ref for o in (rows[k]["probs"] or {}))
        r["same_as_bf16"] = sum(rows[k]["predicted"] == ref[k]["predicted"] for k in rows if k in ref)
        r["drift_mean"], r["drift_p95"] = round(statistics.mean(diffs), 4), round(diffs[int(.95 * len(diffs))], 4)
    out.append(r)
json.dump(out, open("bench/compare.json", "w"), indent=1)
print(f"{'build':8} {'mode':8} {'score':>8} {'size':>6} {'probe':>7} {'same':>8} {'drift':>13} {'p50':>6}")
for r in out:
    print(f"{r['build']:8} {r['mode']:8} {r['correct']:>4}/{r['n']:<3} {r['size_gb'] or '':>6} "
          f"{(('%.4f' % r['probe_dp']) + (' ok' if r['probe_dp'] <= 0.05 else ' NO')) if r['probe_dp'] is not None else '':>7} "
          f"{('%d/%d' % (r['same_as_bf16'], r['n'])) if 'same_as_bf16' in r else '':>8} "
          f"{('%.3f/%.3f' % (r['drift_mean'], r['drift_p95'])) if 'drift_mean' in r else '':>13} {r['p50_s']:>6}")
