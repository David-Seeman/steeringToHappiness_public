"""Convert a PEFT/LoRA adapter (HF format) into the MLX-LM adapter format.

PEFT stores  A: (r, in)  and  B: (out, r)  with scale alpha/r.
MLX-LM stores lora_a: (in, r) and lora_b: (r, out) with a `scale` in adapter_config.json.
Usage: python scripts/00_convert_peft_adapter.py <peft_dir> <out_dir>
"""
import json, re, sys
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
cfg = json.load(open(src / "adapter_config.json"))
dst.mkdir(parents=True, exist_ok=True)

tensors, layers, keys = {}, set(), set()
with safe_open(src / "adapter_model.safetensors", "np") as f:
    for k in f.keys():
        m = re.match(r"base_model\.model\.(model\.layers\.(\d+)\.(.+))\.lora_([AB])\.weight", k)
        assert m, k
        path, layer, mod, ab = m.group(1), int(m.group(2)), m.group(3), m.group(4)
        t = f.get_tensor(k).astype(np.float32).T          # transpose to MLX layout
        tensors[f"{path}.lora_{'a' if ab == 'A' else 'b'}"] = np.ascontiguousarray(t)
        layers.add(layer); keys.add(mod)

save_file(tensors, str(dst / "adapters.safetensors"))
json.dump({
    "fine_tune_type": "lora",
    "num_layers": max(layers) + 1,
    "lora_parameters": {"rank": cfg["r"], "scale": cfg["lora_alpha"] / cfg["r"], "dropout": 0.0, "keys": sorted(keys)},
}, open(dst / "adapter_config.json", "w"), indent=1)
print("converted", len(tensors), "tensors;", max(layers) + 1, "layers; keys:", sorted(keys))
