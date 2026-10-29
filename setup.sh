#!/bin/bash
# One-time setup: a venv with mlx-vlm, the official H2O repo (shim, config, weights) at the pinned revision,
# and optional 8-/4-bit MLX conversions.   ./setup.sh [q8] [q4]
# H2O_REVISION overrides the pin (v1.2.3 = the previous, one-way-only release). Each folder records the revision it
# was built from; a folder from another revision stops the script instead of being mixed with new files.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
REV="${H2O_REVISION:-708e93926a034195d908e033fcb796c55398a6f5}"   # = H2O v1.4.2 (29 Oct 2026; tag v1.4.2 adds card edits only)
SAME_WEIGHTS="cc18ee330e6895c41b9a60f37a0dd122dc4c3461 d7f1b05aa0b4e789b3896116f309ef842440e260"   # v1.4.1 (27 and 28 Oct): same weights, files refreshed below
command -v uv >/dev/null || { echo "install uv first: brew install uv"; exit 1; }
for d in model mlx-q*; do
  [ -d "$d" ] || continue
  have="$(cat "$d/.h2o_revision" 2>/dev/null || echo 'v1.2.x (no revision stamp)')"
  if [ "$have" != "$REV" ] && [[ " $SAME_WEIGHTS " == *" $have "* ]] && [ -z "${H2O_REVISION:-}" ]; then
    echo "$d/: same weights as $REV; refreshing H2O's files in place"; echo "$REV" > "$d/.h2o_revision"; continue
  fi
  if [ "$have" != "$REV" ]; then
    echo "$d/ was built from H2O $have, this setup pins $REV."
    echo "Remove the old builds first:  rm -rf model mlx-q8 mlx-q4 && ./setup.sh ${*:-}"
    exit 1
  fi
done
[ -x .venv/bin/python ] || uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
mkdir -p model && echo "$REV" > model/.h2o_revision     # stamped first, so an interrupted download resumes
.venv/bin/hf download h2oai/h2o-lightning-4b --revision "$REV" --local-dir model \
  --exclude "adapter/*" --exclude "assets/*" --exclude "vllm_patches/*" --exclude "routing_reproducibility/*"
(cd model && ../.venv/bin/python test_shim.py)
for b in ${@+"$@"}; do
  bits=${b#q}; [ -f "mlx-q$bits/config.json" ] && continue
  .venv/bin/python -m mlx_vlm convert --hf-path model --mlx-path "mlx-q$bits" -q --q-bits "$bits" --q-group-size 64
  echo "$REV" > "mlx-q$bits/.h2o_revision"
done
# the shim and its config/siblings travel with each build (as on Hugging Face)
for d in mlx-q*/; do
  [ -d "$d" ] || continue
  for f in h2o_lightning_shim.py serve_config.json selector_v2.py; do if [ -f "model/$f" ]; then cp "model/$f" "$d$f"; fi; done
done
# the training tokenizer (Qwen2Tokenizer, H2O v1.4.0.1+): the facade reads <MODEL>/tokenizer; the shim checks it at start
for d in mlx-q*/; do
  if [ -d "$d" ] && [ -d model/tokenizer ] && [ ! -d "${d}tokenizer" ]; then cp -R model/tokenizer "${d}tokenizer"; fi
done
echo "done (H2O $REV). start with: ./serve.sh   (or MODEL=mlx-q8 ./serve.sh; TWO_WAY=0 for one-way only)"
