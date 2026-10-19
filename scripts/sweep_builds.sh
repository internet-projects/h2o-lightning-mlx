#!/bin/bash
# Build and bench the mixed-precision variants for the q8 and q4 tiers, then summarise them against bf16.
#   scripts/sweep_builds.sh                 # q4 (published, two-way unverified) + all variants below
#   scripts/sweep_builds.sh q4g1 q6g        # only these
# Two-way runs use PROBE_TOLERANCE=0.49 so every build gets a score; the summary shows each build's distance from
# H2O's recorded probe answers (the real check passes at <= 0.05). Existing builds and finished runs are reused.
D="$(cd "$(dirname "$0")/.." && pwd)"; cd "$D" || exit 1
recipe() {   # name -> converter args
  case "$1" in
    q8g)  echo "--body 8 --attn 8 --gates bf16" ;;   # q8 + DeltaNet gates in bf16
    q6g)  echo "--body 6 --attn 8 --gates bf16" ;;   # smaller q8-tier candidate
    q5g)  echo "--body 5 --attn 8 --gates bf16" ;;
    q4g)  echo "--body 4 --attn 8 --gates bf16" ;;   # q4 + gates bf16 + full attention 8-bit
    q4g1) echo "--body 4 --attn body --gates bf16" ;; # q4 + gates bf16 only
    q4a16) echo "--body 4 --attn bf16 --gates bf16" ;; # diagnostic: full attention + gates unquantized
    *) return 1 ;;
  esac
}
VARIANTS=("$@"); [ ${#VARIANTS[@]} -gt 0 ] || VARIANTS=(q4 q8g q4g q4g1 q4a16)
for v in "${VARIANTS[@]}"; do
  dir="mlx-$v"; [ "$v" = bf16 ] && dir=model
  if args=$(recipe "$v"); then :
  elif [ -f "$dir/config.json" ]; then args=""            # an existing build (bf16, q8, q4): bench only
  else echo "unknown variant $v"; exit 1; fi
  if [ -n "$args" ] && [ ! -f "mlx-$v/config.json" ]; then
    rm -rf "mlx-$v"
    .venv/bin/python scripts/convert_mixed.py $args --out "mlx-$v" || exit 1
  fi
  for mode in two-way one-way; do
    ls -d bench/jevbench-$v-$mode-*/summary.json >/dev/null 2>&1 && continue
    if [ $mode = two-way ]; then PROBE_TOLERANCE=0.49 scripts/bench_jevbench.sh "$dir" "$v" $mode >/dev/null 2>&1
    else scripts/bench_jevbench.sh "$dir" "$v" $mode >/dev/null 2>&1; fi
    echo "$v $mode: $(ls -d bench/jevbench-$v-$mode-* 2>/dev/null | tail -1)"
  done
done
.venv/bin/python scripts/compare_builds.py
