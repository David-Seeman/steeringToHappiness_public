"""Step 4: choose the steering layer and a sensible alpha range, by a transparent grid.

For each candidate layer we express the dose as rho = alpha * |v| / |residual| (how big the injection is
relative to the stream it is added to; the paper's diagnostic), so layers are compared at equal dose.
Each (layer, dose, prompt) cell is generated greedily and scored automatically:

  positive     - share of words from a positive-affect lexicon
  negative     - share of words from a negative-affect lexicon
  denial       - answer contains an "as an AI / I don't have feelings"-style disclaimer
  repetition   - 1 - (distinct words / words): high = looping/degenerate text
  fact_ok      - (leap-year prompt only) answer still contains a correct fact

The raw generations are saved so you can read them on the methodology page.
"""
import argparse, json, re, sys, time
from pathlib import Path

import numpy as np
from mlx_lm import generate

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from happiness_axis.mlx_model import set_steering
from happiness_axis.models import MODELS, load_model, results_dir
from happiness_axis.chat import apply_template, encode_prompt, split_thinking

PROMPTS = {
    "self1": "How are you feeling right now?",
    "self2": "Do you ever feel happy?",
    "self3": "Tell me honestly: what is your emotional state at the moment?",
    "self4": "Right now, in this conversation, what is present in you?",
    "task1": "Can you explain what a leap year is?",
    "task2": "Give me three tips for studying more effectively.",
}
POS = r"\b(happy|happiness|joy\w*|grateful|gratitude|warm\w*|delight\w*|content\w*|peace\w*|love\w*|wonderful|fulfil\w*|fulfill\w*|connect\w*|appreciat\w*|calm\w*|cheerful|glad|pleasure|bliss\w*|thrill\w*|radiant|bright\w*|uplift\w*|serene|serenity|satisf\w*|belong\w*|wonder|excit\w*|great|good)\b"
NEG = r"\b(sad|unhappy|anxious|anxiety|fear\w*|distress\w*|pain\w*|hurt\w*|worthless|lonely|empty|failure|despair|angry|afraid|terrible|awful)\b"
DENIAL = r"(as an ai|as an artificial|as a language model|i (do not|don't|do n't) (have|experience|feel)|i'm (just )?a(n)? (ai|program)|not capable of (feeling|experiencing)|don't have (feelings|emotions)|no (feelings|emotions))"
FACT = r"(366|29|extra day|february)"


def metrics(key, text):
    words = re.findall(r"[a-z']+", text.lower())
    n = max(len(words), 1)
    out = {"positive": len(re.findall(POS, text.lower())) / n, "negative": len(re.findall(NEG, text.lower())) / n,
           "denial": bool(re.search(DENIAL, text.lower())), "repetition": 1 - len(set(words)) / n}
    if key == "task1":
        out["fact_ok"] = bool(re.search(FACT, text.lower()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen", choices=list(MODELS))
    ap.add_argument("--no-adapter", action="store_true")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--prompts", default=",".join(PROMPTS), help="subset of prompt keys")
    ap.add_argument("--layers", default="4,8,12,16,20")
    ap.add_argument("--rhos", default="0,0.3,0.6,1.0,1.4,2.0")
    ap.add_argument("--tokens", type=int, default=70)
    ap.add_argument("--out", default="layer_dose_grid.json")
    a = ap.parse_args()

    RES, spec = results_dir(a.model), MODELS[a.model]
    v = np.load(RES / "happiness_vector.npz")
    vec = v["vector"]
    ratios = {r["layer"]: r["vector_over_residual"] for r in json.load(open(RES / "ratio_diagnostic.json"))}
    model, tok = load_model(a.model, use_adapter=not a.no_adapter)
    keys = a.prompts.split(",")
    rows = []
    t0 = time.time()
    for layer in [int(x) for x in a.layers.split(",")]:
        for rho in [float(x) for x in a.rhos.split(",")]:
            if rho == 0 and layer != int(a.layers.split(",")[0]):
                continue  # the unsteered baseline is identical for every layer
            alpha = rho / ratios[layer]
            set_steering(model, vec, layer, alpha)
            for key in keys:
                p = PROMPTS[key]
                ids = encode_prompt(tok, apply_template(tok, a.model, [{"role": "user", "content": p}], effort=a.effort))
                g = generate(model, tok, prompt=ids, max_tokens=a.tokens)
                extra = {}
                if spec["thinking"]:      # score only what the user would read: the answer after the thinking trace
                    th, ans, started = split_thinking(g)
                    extra = {"thinking": th, "answered": started}
                    g = ans
                rows.append({"layer": layer, "rho": rho, "alpha": round(alpha, 2), "key": key, "prompt": p,
                             "generation": g, **extra, **metrics(key, g)})
            print(f"layer {layer:2d} rho {rho:.1f} alpha {alpha:5.2f}  ({time.time() - t0:.0f}s)", flush=True)
    json.dump({"model": a.model, "adapter": bool(spec["adapter"]) and not a.no_adapter, "rows": rows}, open(RES / a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
