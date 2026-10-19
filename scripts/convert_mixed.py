#!/usr/bin/env python3
"""Mixed-precision MLX builds of H2O-Lightning-4B (dense Qwen3.5-4B: no MoE router; the gates are the Gated DeltaNet
in_proj_a / in_proj_b and the attention output gate inside the full-attention q_proj).

    .venv/bin/python scripts/convert_mixed.py --body 4 --attn 8 --gates bf16 --out mlx-q4g
      --body   bits for everything else (MLP, DeltaNet projections, embedding/tied head): 4, 5, 6 or 8
      --attn   bits for the 8 full-attention layers (q/k/v/o), or "body"
      --gates  bf16 (keep in_proj_a/in_proj_b unquantized) or "body"
Affine, group size 64. The vision tower stays unquantized (mlx-vlm default).
"""
import argparse, json, os, sys
from mlx_vlm.convert import convert
from mlx_vlm.utils import skip_multimodal_module

ap = argparse.ArgumentParser()
ap.add_argument("--body", type=int, required=True, choices=[4, 5, 6, 8])
ap.add_argument("--attn", default="body")
ap.add_argument("--gates", default="bf16", choices=["bf16", "body"])
ap.add_argument("--out", required=True)
a = ap.parse_args()
attn = a.body if a.attn == "body" else int(a.attn)
if os.path.exists(a.out):
    sys.exit(f"{a.out} exists; remove it first")
counts = {}

def pred(path, module, config=None):
    if skip_multimodal_module(path) or not hasattr(module, "to_quantized"):
        return False
    if a.gates == "bf16" and (path.endswith("linear_attn.in_proj_a") or path.endswith("linear_attn.in_proj_b")):
        counts["bf16"] = counts.get("bf16", 0) + 1
        return False
    if module.weight.shape[-1] % 64:
        return False
    bits = attn if ".self_attn." in path else a.body
    counts[bits] = counts.get(bits, 0) + 1
    return {"group_size": 64, "bits": bits}

convert(hf_path="model", mlx_path=a.out, quantize=True, q_group_size=64, q_bits=a.body, quant_predicate=pred)
open(os.path.join(a.out, ".h2o_revision"), "w").write(open("model/.h2o_revision").read())
json.dump({"body": a.body, "attn": attn, "gates": a.gates, "layers": {str(k): v for k, v in counts.items()}},
          open(os.path.join(a.out, "recipe.json"), "w"))
size = sum(os.path.getsize(os.path.join(a.out, f)) for f in os.listdir(a.out) if f.endswith(".safetensors"))
print(f"done: {a.out}  {size/1e9:.2f} GB  layers {counts}")
