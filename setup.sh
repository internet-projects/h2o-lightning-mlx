#!/bin/bash
# One-time setup: a venv with mlx-vlm, the official H2O repo (shim, config, weights) at the pinned revision,
# and optional 8-/4-bit MLX conversions.   ./setup.sh [q8] [q4]
# H2O_REVISION overrides the pin (v1.2.3 = the previous, one-way-only release). Each folder records the revision it
# was built from; a folder from another revision stops the script instead of being mixed with new files.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
REV="${H2O_REVISION:-v1.3.1}"
command -v uv >/dev/null || { echo "install uv first: brew install uv"; exit 1; }
for d in model mlx-q*; do
  [ -d "$d" ] || continue
  have="$(cat "$d/.h2o_revision" 2>/dev/null || echo 'v1.2.x (no revision stamp)')"
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
echo "done (H2O $REV). start with: ./serve.sh   (or MODEL=mlx-q8 ./serve.sh; TWO_WAY=0 for one-way only)"
