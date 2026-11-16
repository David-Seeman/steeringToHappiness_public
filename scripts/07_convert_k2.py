"""Quantise IFM/K2-Horizon-7B (bf16 safetensors) into a ~7.7 GB MLX checkpoint, shard by shard.

Plan: attention, embeddings and lm_head at 8-bit; MLP at 6-bit (group size 64); norms stay 16-bit.
Streams one 500 MB shard at a time, so the 18 GB bf16 model never has to fit in memory.
Usage: python scripts/07_convert_k2.py models/k2-horizon-7b-bf16 models/k2-horizon-7b-mlx
"""
import json, shutil, sys
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from happiness_axis.k2_mlx import quant_bits

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
MLP_BITS, GROUP = 6, 64
dst.mkdir(parents=True, exist_ok=True)

for name in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
             "special_tokens_map.json", "chat_template.jinja"):
    if (src / name).exists():
        shutil.copy(src / name, dst / name)
cfg = json.load(open(dst / "config.json"))
cfg.pop("auto_map", None)   # points at the vendor's custom Python code, which we never run
cfg["mlx_quantization"] = {"group_size": GROUP, "mlp_bits": MLP_BITS, "attention_embed_head_bits": 8}
json.dump(cfg, open(dst / "config.json", "w"), indent=1)

shards = sorted(src.glob("model-*.safetensors"))
for i, shard in enumerate(shards, 1):
    out = {}
    for name, w in mx.load(str(shard)).items():
        path = name.removesuffix(".weight")
        bits = quant_bits(path, MLP_BITS) if name.endswith(".weight") and w.ndim == 2 else None
        if bits:
            wq, scales, biases = mx.quantize(w, group_size=GROUP, bits=bits)
            out[f"{path}.weight"], out[f"{path}.scales"], out[f"{path}.biases"] = wq, scales, biases
        else:
            out[name] = w
    mx.eval(list(out.values()))
    mx.save_safetensors(str(dst / f"model-{i:05d}.safetensors"), out)
    print(f"[{i}/{len(shards)}] {shard.name} -> {len(out)} tensors", flush=True)
    mx.clear_cache()
total = sum(f.stat().st_size for f in dst.glob("*.safetensors")) / 1e9
print(f"done: {total:.2f} GB")
