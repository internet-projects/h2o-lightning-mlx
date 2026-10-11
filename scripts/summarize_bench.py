#!/usr/bin/env python3
"""Score a JevBench results.jsonl by tier and list the misses (vs the model card: 205/231, 48/48 71/72 86/111)."""
import json, sys, os, statistics
out = sys.argv[1]
rows = [json.loads(l) for l in open(os.path.join(out, "results.jsonl")) if l.strip()]
print("fields:", sorted(rows[0].keys())[:40])
def tier(r):
    i = r.get("id") or r.get("task_id") or ""
    return "easy" if i.startswith("easy") else "hard" if i.startswith("hard") else "standard"
def ok(r):
    for k in ("correct", "is_correct", "score_correct"):
        if k in r: return bool(r[k])
    return r.get("prediction") == r.get("expected")
by = {}
for r in rows:
    t = tier(r); c, n = by.get(t, (0, 0)); by[t] = (c + ok(r), n + 1)
tot = sum(c for c, _ in by.values()), sum(n for _, n in by.values())
card = {"easy": (48, 48), "standard": (71, 72), "hard": (86, 111)}
summary = {"total": f"{tot[0]}/{tot[1]}", "by_tier": {t: f"{c}/{n} (card {card[t][0]}/{card[t][1]})" for t, (c, n) in by.items()},
           "misses": [r.get("id") or r.get("task_id") for r in rows if not ok(r)]}
lat = [r.get("latency_s") or (r.get("latency_ms", 0) / 1000) for r in rows if (r.get("latency_s") or r.get("latency_ms"))]
if lat:
    lat.sort(); summary["latency_p50_s"] = round(lat[len(lat)//2], 3); summary["latency_p95_s"] = round(lat[int(len(lat)*0.95)], 3)
card_wrong = {"hard-opus-a-temporal_numeric-07", "hard-opus-a-temporal_numeric-12", "hard-opus-c-temporal_numeric-04", "hard-sol-b-temporal_numeric-03"}
summary["near_tie_items_card_wrong"] = sorted(card_wrong)
summary["misses_not_in_card_near_ties"] = sorted(set(summary["misses"]) - card_wrong)
json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
