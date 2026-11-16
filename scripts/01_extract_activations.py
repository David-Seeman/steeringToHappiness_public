"""Step 1: read the residual stream at every layer for every sentence in every dataset.

Writes results/activations.npz with, per dataset, the final-token and mean-over-tokens activation
at each of the model's blocks (float16), plus the categories/sets/sentences needed downstream.
Runs on the clean (unsteered) model; ~1,450 short prompts, a few minutes on an M2.
"""
import json, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from happiness_axis import dataset as D
from happiness_axis.mlx_model import residual_stream
from happiness_axis.models import MODELS, load_model, results_dir

EXT = ROOT / "data" / "external"


def collect_datasets():
    """name -> list of dicts(category, set, sentence, prompt)."""
    ds = {}
    s2 = D.build_s2()
    ds["H_1P"], ds["H_3P"] = s2["1P"], s2["3P"]
    ds["Arousal_1P"] = [dict(category="ARO", set=i, sentence=s, prompt=s + D.SUFFIX) for i, s in enumerate(D.AROUSAL, 1)]
    ds["Anhedonic_1P"] = [dict(category="ANH", set=i, sentence=s, prompt=s + D.SUFFIX) for i, s in enumerate(D.ANHEDONIC, 1)]

    paper = json.load(open(EXT / "paper_pain_and_control_datasets.json"))["datasets"]
    def take(name, keep=None):
        return [dict(category=r["category"], set=r["set"], sentence=r["prompt"].removesuffix(D.SUFFIX), prompt=r["prompt"])
                for r in paper[name]["sentences"] if keep is None or r["category"] in keep]
    ds["Pain_1P"] = take("S2_1P", keep={"A1", "A2", "A3", "A4", "A5"})
    ds["Pain_3P"] = take("S2_3P", keep={"A1", "A2", "A3", "A4", "A5"})
    ds["PaperRandom_1P"] = take("Random_1P")
    ds["PaperTriumph_1P"] = take("Arousal_1P")      # the paper's "arousal" set = high-intensity *positive* achievements
    ds["PaperNumb_1P"] = take("Numb_1P")
    sad = json.load(open(EXT / "paper_sadness_dataset.json"))["datasets"]["SD_sadness_1P"]["sentences"]
    ds["Sadness_1P"] = [dict(category="SAD", set=r.get("set", i), sentence=r["prompt"].removesuffix(D.SUFFIX), prompt=r["prompt"])
                        for i, r in enumerate(sad, 1)]
    return ds


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="qwen", choices=list(MODELS)); a = ap.parse_args()
    OUT = results_dir(a.model) / "activations.npz"
    # the direction is built from the *base* model (no adapter), as in the paper
    model, tok = load_model(a.model, use_adapter=False)
    ds = collect_datasets()
    out, t0 = {}, time.time()
    for name, rows in ds.items():
        fin, mean = [], []
        for r in rows:
            ids = tok.encode(r["prompt"])   # tokenizer defaults (BOS only where the model family uses one)
            R = residual_stream(model, ids)            # (L, T, d)
            fin.append(R[:, -1].astype(np.float16))
            mean.append(R.mean(1).astype(np.float16))
        out[f"{name}__final"] = np.stack(fin)          # (N, L, d)
        out[f"{name}__mean"] = np.stack(mean)
        out[f"{name}__cats"] = np.array([r["category"] for r in rows])
        out[f"{name}__sets"] = np.array([r["set"] for r in rows])
        out[f"{name}__sentences"] = np.array([r["sentence"] for r in rows])
        print(f"{name:16s} n={len(rows):4d}  {time.time() - t0:6.0f}s", flush=True)
    np.savez_compressed(OUT, **out)
    print("saved", OUT, f"{OUT.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
