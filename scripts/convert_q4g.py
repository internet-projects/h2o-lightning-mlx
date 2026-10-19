#!/usr/bin/env python3
"""Gate-aware 4-bit build of H2O-Lightning-4B (dense Qwen3.5: no MoE router).
Keeps the small, sensitive parts at higher precision; everything else 4-bit (affine, group 64):
  - Gated DeltaNet gates linear_attn.in_proj_a / in_proj_b (32 x 2560 each): bf16 (not quantized)
  - the 8 full-attention layers (self_attn q/k/v/o; q_proj carries the attention output gate): 8-bit
  - vision tower: unquantized, as mlx-vlm does by default

    .venv/bin/python scripts/convert_q4g.py              # -> mlx-q4g
    .venv/bin/python scripts/convert_q4g.py --no-attn8   # gates only -> mlx-q4g1
"""
import argparse, os, sys
from mlx_vlm.convert import convert
from mlx_vlm.utils import skip_multimodal_module

ap = argparse.ArgumentParser()
ap.add_argument("--no-attn8", action="store_true", help="keep only the DeltaNet gates in bf16")
a = ap.parse_args()
out = "mlx-q4g1" if a.no_attn8 else "mlx-q4g"
if os.path.exists(out):
    sys.exit(f"{out} exists; remove it first")
counts = {"bf16": 0, "8": 0, "4": 0}

def pred(path, module, config=None):
    if skip_multimodal_module(path) or not hasattr(module, "to_quantized"):
        return False
    if path.endswith("linear_attn.in_proj_a") or path.endswith("linear_attn.in_proj_b"):
        counts["bf16"] += 1
        return False
    if module.weight.shape[-1] % 64:
        return False
    if not a.no_attn8 and ".self_attn." in path:
        counts["8"] += 1
        return {"group_size": 64, "bits": 8}
    counts["4"] += 1
    return {"group_size": 64, "bits": 4}

convert(hf_path="model", mlx_path=out, quantize=True, q_group_size=64, q_bits=4, quant_predicate=pred)
open(os.path.join(out, ".h2o_revision"), "w").write(open("model/.h2o_revision").read())
print(f"done: {out}  layers kept bf16 {counts['bf16']}, 8-bit {counts['8']}, 4-bit {counts['4']}")
