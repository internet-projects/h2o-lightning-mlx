#!/bin/bash
# One-time setup: a venv with mlx-vlm, the official H2O repo (shim, config, weights) at the pinned revision,
# and optional 8-/4-bit MLX conversions.   ./setup.sh [q8] [q4]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
REV="${H2O_REVISION:-v1.2.3}"
command -v uv >/dev/null || { echo "install uv first: brew install uv"; exit 1; }
[ -x .venv/bin/python ] || uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/hf download h2oai/h2o-lightning-4b --revision "$REV" --local-dir model --exclude "adapter/*" --exclude "assets/*"
(cd model && ../.venv/bin/python test_shim.py)
for b in "$@"; do
  bits=${b#q}; [ -f "mlx-q$bits/config.json" ] && continue
  .venv/bin/python -m mlx_vlm convert --hf-path model --mlx-path "mlx-q$bits" -q --q-bits "$bits" --q-group-size 64
done
echo "done. start with: ./serve.sh   (or MODEL=mlx-q8 ./serve.sh)"
