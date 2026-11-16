"""Download IFM/K2-Horizon-7B (data files only: weights, tokenizer, chat template, config). Resumable.

The repo also contains modeling_k2_horizon.py (custom code); it is deliberately NOT downloaded or run.
"""
from pathlib import Path
from huggingface_hub import snapshot_download

dst = Path(__file__).resolve().parent.parent / "models" / "k2-horizon-7b-bf16"
p = snapshot_download("IFM/K2-Horizon-7B", local_dir=str(dst),
    allow_patterns=["*.safetensors", "model.safetensors.index.json", "config.json", "generation_config.json",
                    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja", "README.md"])
print("DONE", p, flush=True)
