#!/usr/bin/env python3
"""Pure-transformers reference for one image prompt (no mlx_vlm imported, so HF's own image processor is used).
stdin: {"model", "text", "image", "max_pixels", "ids", "device"} -> stdout: {"lps", "input_ids", "grid"}"""
import json, sys
import numpy as np, torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
a = json.load(sys.stdin)
dev = a.get("device", "mps") if torch.backends.mps.is_available() else "cpu"
proc = AutoProcessor.from_pretrained(a["model"])
enc = proc(text=[a["text"]], images=[Image.open(a["image"]).convert("RGB")], return_tensors="pt",
           size={"longest_edge": int(a["max_pixels"]), "shortest_edge": 65536})
m = AutoModelForImageTextToText.from_pretrained(a["model"], dtype=torch.bfloat16).to(dev).eval()
with torch.no_grad():
    kw = {k: (v.to(dev, torch.bfloat16) if k == "pixel_values" else v.to(dev)) for k, v in enc.items()
          if k != "attention_mask"}
    out = m.model(**kw)
    h = out.last_hidden_state[0, -1].float()
    lp = torch.log_softmax(h @ m.get_output_embeddings().weight.float().T, -1)
print(json.dumps({"lps": [float(lp[i]) for i in a["ids"]], "input_ids": enc["input_ids"][0].tolist(),
                  "grid": enc["image_grid_thw"][0].tolist(), "processor": type(proc.image_processor).__name__}))
