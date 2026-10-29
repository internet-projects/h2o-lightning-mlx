#!/bin/bash
# H2O-Lightning-4B on Apple Silicon: the MLX facade (a vLLM stand-in) + the OFFICIAL shim, unmodified.
#   MODEL=model ./serve.sh            # bf16 (the official weights)          -> http://127.0.0.1:8741/v1/systemone
#   MODEL=mlx-q8 ./serve.sh           # an 8-bit MLX conversion
# Attention: one engine serves both modes; a request picks one with "attention_mode": "two-way" | "one-way".
#   TWO_WAY=1 (default, as upstream since v1.3): requests without attention_mode read two-way.
#   TWO_WAY=0: one-way only (the pre-v1.3 behaviour); a two-way request is a 400.
#   Before the first two-way answer the shim checks this engine against the release's recorded probe answers.
# Optional authenticated front door:  H2O_API_KEY=... FRONT_PORT=8742 ./serve.sh
# Env: FACADE_PORT (8743), PORT (8741, the shim), SHIM_NOUL_FLOOR (unset = as shipped; 0 = routing mode),
#      SHIM_ATTENTION_MODE (the default mode with TWO_WAY=1: bidirectional | causal),
#      FACADE_IDLE_UNLOAD_S (0 = keep the weights resident), H2O_HOSTS (front bind addresses, default 127.0.0.1)
# PROFILE=di (H2O v1.4.2+, as upstream serve.sh): the Decision Index profile, the shim's conditional re-read (a choice or
#      score question under 0.8 top probability is read again with the input written twice, 2:1 first-weighted) and the
#      yes/no floor 0.801 it was measured with. Unset: the default serving.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
PY="${PY:-$HERE/.venv/bin/python}"; MODEL="${MODEL:-model}"
FACADE_PORT="${FACADE_PORT:-8743}"; PORT="${PORT:-8741}"
case "${TWO_WAY:-1}" in
  1) export SHIM_TWO_WAY_FLAG=1 ;;
  0) unset SHIM_TWO_WAY_FLAG ;;
  *) echo "TWO_WAY must be 0 or 1"; exit 1 ;;
esac
case "${PROFILE:-}" in
  "") ;;
  di) export SHIM_REREAD_BELOW="${SHIM_REREAD_BELOW:-0.8}" SHIM_NOUL_FLOOR="${SHIM_NOUL_FLOOR:-0.801}" ;;
  *) echo "PROFILE must be di or unset"; exit 1 ;;
esac
[ -f model/h2o_lightning_shim.py ] || { echo "run ./setup.sh first (downloads the official repo into ./model)"; exit 1; }
"$PY" mlx_vllm_facade.py --model "$MODEL" --port "$FACADE_PORT" & P1=$!
"$PY" model/h2o_lightning_shim.py --config model/serve_config.json --vllm "http://127.0.0.1:$FACADE_PORT" \
  --host 127.0.0.1 --port "$PORT" & P2=$!
PIDS="$P1 $P2"
if [ -n "${FRONT_PORT:-}" ]; then
  H2O_SHIM="127.0.0.1:$PORT" "$PY" h2o_front.py --port "$FRONT_PORT" & PIDS="$PIDS $!"
fi
trap 'kill $PIDS 2>/dev/null; wait' EXIT INT TERM
until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; do
  for p in $PIDS; do kill -0 $p 2>/dev/null || { echo "a process exited during start-up"; exit 1; }; done
  sleep 2
done
echo "H2O-Lightning-4B (MLX, $MODEL, $(cat "$MODEL/.h2o_revision" 2>/dev/null || echo '?'), two-way ${TWO_WAY:-1}) ready: http://127.0.0.1:$PORT/v1/systemone"
# bash 3.2 (macOS) has no `wait -n`
while true; do for p in $PIDS; do kill -0 $p 2>/dev/null || exit 1; done; sleep 5; done
