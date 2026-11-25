#!/bin/bash
# One-time setup: a venv with mlx-vlm, the official H2O repo (shim, config, weights) at the pinned revision,
# and optional 8-/4-bit MLX conversions.   ./setup.sh [q8] [q4]
# H2O_REVISION overrides the pin (v1.2.3 = the previous, one-way-only release). Each folder records the revision it
# was built from; a folder from another revision stops the script instead of being mixed with new files.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
REV="${H2O_REVISION:-eea2706d8b2ee8b325ace764a46a24e535b56491}"   # = H2O v1.4.4 (25 Nov 2026, tag v1.4.4)
SAME_WEIGHTS="cc18ee330e6895c41b9a60f37a0dd122dc4c3461 d7f1b05aa0b4e789b3896116f309ef842440e260 708e93926a034195d908e033fcb796c55398a6f5"   # v1.4.1 (27 and 28 Oct) and v1.4.2: same weights, files refreshed below
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
# v1.4.4 ships tokenizer/ as tokenizer.json only (the base model's native split); drop the older split's files if a
# previous download left them
if [ -f model/tokenizer/tokenizer.json ] && grep -q TokenizersBackend model/tokenizer/tokenizer_config.json 2>/dev/null; then
  rm -f model/tokenizer/vocab.json model/tokenizer/merges.txt
fi
(cd model && ../.venv/bin/python test_shim.py)
for b in ${@+"$@"}; do
  bits=${b#q}; [ -f "mlx-q$bits/config.json" ] && continue
  .venv/bin/python -m mlx_vlm convert --hf-path model --mlx-path "mlx-q$bits" -q --q-bits "$bits" --q-group-size 64
  echo "$REV" > "mlx-q$bits/.h2o_revision"
done
# the shim and its config/siblings travel with each build (as on Hugging Face)
for d in mlx-q*/; do
  [ -d "$d" ] || continue
  for f in h2o_lightning_shim.py serve_config.json selector_v2.py tokenizer_contract.json; do if [ -f "model/$f" ]; then cp "model/$f" "$d$f"; fi; done
done
# the served tokenizer (H2O v1.4.4: the base model's native split, tokenizer.json as TokenizersBackend): the facade reads
# <MODEL>/tokenizer and the shim checks it against tokenizer_contract.json at start, so each build gets the current copy
for d in mlx-q*/; do
  if [ -d "$d" ] && [ -d model/tokenizer ]; then rm -rf "${d}tokenizer"; cp -R model/tokenizer "${d}tokenizer"; fi
done
echo "done (H2O $REV). start with: ./serve.sh   (or MODEL=mlx-q8 ./serve.sh; TWO_WAY=0 for one-way only)"
