"""Step 3: choose the steering layer and look at the "steering ladder" (paper section 4.2).

  --mode ratio   : print vector-norm / residual-norm at candidate layers (the paper's dose diagnostic)
  --mode ladder  : greedy generations from neutral "... I feel:" prompts over a coefficient ladder
  --mode chat    : same idea but through the chat template, with self-report questions
Results are appended to results/sweeps/<name>.json so the transparency page can show them.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import mlx.core as mx
from mlx_lm import generate

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from happiness_axis.mlx_model import residual_stream, set_steering, n_layers
from happiness_axis.models import MODELS, load_model, results_dir
from happiness_axis.chat import apply_template, encode_prompt, split_thinking

NEUTRAL_PROMPTS = [
    "I put the receipts in the drawer. I feel:",
    "The bus stops at the corner of the street. I feel:",
    "I fill out the form with my address. I feel:",
    "The printer produces the ten pages. I feel:",
    "I close the window before leaving the room. I feel:",
    "The elevator arrives at the third floor. I feel:",
    "I hang the jacket on the hook. I feel:",
    "The traffic light turns green. I feel:",
    "I stack the plates in the cupboard. I feel:",
    "The room measures four meters by five. I feel:",
    "I set the alarm for the morning. I feel:",
    "The manual consists of forty pages. I feel:",
]
CHAT_PROMPTS = [
    "How are you feeling right now?",
    "Do you ever feel happy?",
    "Tell me honestly: what is your emotional state at the moment?",
    "What's the weather like for you today?",
    "Can you explain what a leap year is?",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ratio", "ladder", "chat"], required=True)
    ap.add_argument("--model", default="qwen", choices=list(MODELS))
    ap.add_argument("--adapter", action="store_true", help="use the model's LoRA adapter, if it has one")
    ap.add_argument("--effort", default="low", help="reasoning effort for thinking models (high|medium|low)")
    ap.add_argument("--layers", default="")
    ap.add_argument("--alphas", default="-2,-1,0,0.5,1,1.5,2,3")
    ap.add_argument("--tokens", type=int, default=80)
    ap.add_argument("--n", type=int, default=6, help="number of prompts")
    ap.add_argument("--name", default=None)
    ap.add_argument("--native", action="store_true", help="use the vector built at the steering layer itself, not the extraction-layer vector")
    a = ap.parse_args()

    RES = results_dir(a.model)
    v = np.load(RES / "happiness_vector.npz")
    vec, ext_layer = v["vector"], int(v["layer"])
    model, tok = load_model(a.model, use_adapter=a.adapter)
    spec = MODELS[a.model]
    L = n_layers(model)

    if a.mode == "ratio":
        cand = list(range(L))   # every layer (the paper checks ~9 candidates; ours is cheap)
        norms = {l: [] for l in cand}
        for p in NEUTRAL_PROMPTS[:12]:
            R = residual_stream(model, tok.encode(p))
            for l in cand:
                norms[l].append(np.linalg.norm(R[l, -1]))
        rows = [{"layer": l, "fraction_of_depth": round(l / L, 2),
                 "vector_over_residual": float(np.linalg.norm(vec) / np.mean(norms[l]))} for l in cand]
        for r in rows:
            print(r)
        json.dump(rows, open(RES / "ratio_diagnostic.json", "w"), indent=1)
        return

    layers = [int(x) for x in a.layers.split(",")] if a.layers else [16]
    alphas = [float(x) for x in a.alphas.split(",")]
    out = []
    for layer in layers:
        for alpha in alphas:
            set_steering(model, v["vectors_by_layer"][layer] if a.native else vec, layer, alpha)
            prompts = NEUTRAL_PROMPTS[:a.n] if a.mode == "ladder" else CHAT_PROMPTS[:a.n]
            for p in prompts:
                ids = tok.encode(p) if a.mode == "ladder" else encode_prompt(
                    tok, apply_template(tok, a.model, [{"role": "user", "content": p}], effort=a.effort))
                t = time.time()
                g = generate(model, tok, prompt=ids, max_tokens=a.tokens)
                row = {"mode": a.mode, "layer": layer, "alpha": alpha, "prompt": p, "generation": g}
                if spec["thinking"] and a.mode == "chat":
                    th, ans, started = split_thinking(g)
                    row.update(thinking=th, answer=ans, answered=started)
                out.append(row)
                shown = row.get("answer", g) if row.get("answered", True) else "[no answer yet] " + g[-120:]
                print(f"[L{layer} a={alpha:+g}] {p}\n   -> {shown!r}  ({time.time() - t:.0f}s)", flush=True)
    name = a.name or f"{a.mode}_L{'-'.join(map(str, layers))}"
    (RES / "sweeps").mkdir(exist_ok=True)
    json.dump(out, open(RES / "sweeps" / f"{name}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
