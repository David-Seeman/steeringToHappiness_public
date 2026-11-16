"""Registry of the models the pipeline and the chat app can run, and one place to load them.

Each entry says where its weights live, how its chat template wants to be driven, and which folder under
results/ holds its activations, direction and sweeps. Steering settings (layer, default alpha) are filled in
after the layer/dose grid has been run and are read from results/<key>/steering.json when present.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MODELS = {
    "qwen": {
        "label": "Qwen2.5-7B-Instruct (8-bit)",
        "short": "Qwen2.5-7B",
        "kind": "mlx_lm",
        "path": "models/qwen2.5-7b-instruct-8bit",
        "adapter": "models/selfreport-lora-mlx",   # the paper authors' self-report LoRA (the default)
        "adapters": [                               # selectable in the chat page; switching reloads the model
            {"id": "paper", "label": "Paper's self-report LoRA", "path": "models/selfreport-lora-mlx"},
        ],
        "default_adapter": "paper",
        "thinking": False,
    },
    "k2": {
        "label": "K2 Horizon 7B (6/8-bit)",
        "short": "K2 Horizon 7B",
        "kind": "k2",
        "path": "models/k2-horizon-7b-mlx",
        "adapter": None,                            # no fine-tuning, as requested
        "adapters": [],
        "default_adapter": None,
        "thinking": True,                           # reasoning model: emits a thinking trace before the answer
    },
}


def results_dir(key: str) -> Path:
    d = ROOT / "results" / key
    d.mkdir(parents=True, exist_ok=True)
    return d


def img_dir(key: str) -> Path:
    d = ROOT / "app" / "static" / "img" / key
    d.mkdir(parents=True, exist_ok=True)
    return d


def steering_settings(key: str) -> dict:
    p = results_dir(key) / "steering.json"
    return json.load(open(p)) if p.exists() else {}


def adapter_path(key: str, adapter_id: str | None = None) -> str | None:
    """Relative path of a model's adapter: the one asked for, else the default. None if there is none or it isn't on disk."""
    spec = MODELS[key]
    aid = adapter_id or spec.get("default_adapter")
    path = next((a["path"] for a in spec.get("adapters", []) if a["id"] == aid), spec["adapter"] if not adapter_id else None)
    return path if path and (ROOT / path).exists() else None


def load_model(key: str, use_adapter: bool = True, adapter_id: str | None = None):
    """-> (model, tokenizer). The model is on the GPU and in eval mode. `adapter_id` picks one of spec["adapters"]."""
    spec = MODELS[key]
    path = str(ROOT / spec["path"])
    if spec["kind"] == "k2":
        from .k2_mlx import load_k2
        return load_k2(path)
    from .mlx_model import load_model as load_mlx_lm
    adapter = adapter_path(key, adapter_id) if use_adapter else None
    return load_mlx_lm(path, str(ROOT / adapter) if adapter else None)
