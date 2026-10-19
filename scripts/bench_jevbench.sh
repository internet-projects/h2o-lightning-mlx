#!/bin/bash
# JevBench public set (231) against the MLX facade + the UNMODIFIED shim with the SHIPPED serve_config.json
# (yes/no floor ON = the evaluated configuration). Ports 8741 (shim) / 8753 (facade) are private to this run.
#   scripts/bench_jevbench.sh [model dir] [label] [one-way|two-way]     (default: model bf16 two-way)
# Two-way: the shim's one-engine flag; its probe check against the release's two-way answers runs at the warm-up
# and a failure stops the run (the warm-up never gets a 200). Paths may contain spaces.
set -x
D="$(cd "$(dirname "$0")/.." && pwd)"; cd "$D" || exit 1
M="${1:-model}"; LABEL="${2:-bf16}"; ATT="${3:-two-way}"
[ -f "$M/config.json" ] || { echo "no model at '$M' (usage: scripts/bench_jevbench.sh <model dir> <label> <one-way|two-way>)"; exit 1; }
case "$ATT" in
  one-way) unset SHIM_TWO_WAY_FLAG; export SHIM_ATTENTION_MODE=causal;;
  two-way) export SHIM_TWO_WAY_FLAG=1 SHIM_ATTENTION_MODE=bidirectional;;
  *) echo "mode must be one-way or two-way"; exit 1;;
esac
OUT="$D/bench/jevbench-$LABEL-$ATT-$(date +%Y%m%d-%H%M)"; mkdir -p "$OUT"
.venv/bin/python -c "import jevbench" 2>/dev/null || uv pip install --python .venv/bin/python -e ./jevbench
FACADE_IDLE_UNLOAD_S=0 .venv/bin/python mlx_vllm_facade.py --model "$M" --port 8753 --preload > "$OUT/facade.log" 2>&1 & F=$!
.venv/bin/python model/h2o_lightning_shim.py --config model/serve_config.json --vllm http://127.0.0.1:8753 --port 8741 > "$OUT/shim.log" 2>&1 & S=$!
trap 'kill $F $S 2>/dev/null' EXIT
code=000
for i in $(seq 1 120); do
  code=$(curl -s -o "$OUT/warmup.json" -w '%{http_code}' http://127.0.0.1:8741/v1/systemone -H 'Content-Type: application/json' -d '{"state": "warm-up","questions": {"decision": {"type": "noul", "instructions": "Is this a warm-up?","criteria": {"true": "yes", "false": "no"}}}}')
  [ "$code" = 200 ] && break
  kill -0 $F 2>/dev/null && kill -0 $S 2>/dev/null || { echo "the facade or shim exited:"; tail -n 5 "$OUT/facade.log" "$OUT/shim.log"; exit 1; }
  grep -q "probe answers\|ignores attention_mode" "$OUT/warmup.json" 2>/dev/null && break   # the two-way check failed
  sleep 3
done
[ "$code" = 200 ] || { echo "warm-up failed (HTTP $code):"; cat "$OUT/warmup.json"; echo; tail -n 5 "$OUT/shim.log"; exit 1; }
grep -h "two-way verified" "$OUT/shim.log"
curl -s http://127.0.0.1:8741/health; echo
cd jevbench || exit 1
/usr/bin/time -l ../.venv/bin/python -m jevbench.cli run \
  --tasks datasets/public/easy.jsonl,datasets/public/original.jsonl,datasets/public/hard.jsonl \
  --adapter typesafe --endpoint http://127.0.0.1:8741 --key-env '' --model h2oai/h2o-lightning-4b \
  --cost-basis self_hosted_gpu --reserve-usd 0 --results "$OUT/results.jsonl" --raw-dir "$OUT/raw" \
  --ledger "$OUT/ledger.jsonl" --manifest "$OUT/manifest.json" --run-label "h2o-lightning-4b-mlx-$LABEL-$ATT" --delay-s 0 2>&1 | tail -40
cd "$D"
footprint $F 2>/dev/null | head -12
.venv/bin/python scripts/summarize_bench.py "$OUT"
