"""Step 2: find and validate the happiness direction (paper sections 3.2 and 3.3).

  1. For every layer, 5-fold cross-validated projection AUC (split by sentence set).
  2. Pick the extraction layer = best held-out AUC (final-token activations).
  3. Build the vector from all 200 sentences at that layer: mean(wellbeing) - mean(controls),
     denoised by projecting out the PCs that explain 50% of the control variance.
  4. Validate: AUCs, z-scores of every dataset, cosines with control directions, unembedding readout.

Writes results/happiness_vector.npz, results/analysis.json and the charts in app/static/img/.
"""
import json, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from happiness_axis import dataset as D
from happiness_axis import vectors as V

from happiness_axis.models import MODELS, load_model, results_dir, img_dir
IMG = None   # set in main()
RES = None
MODEL_KEY = "qwen"
POS, CTRL, NEU = D.POSITIVE_CATS, D.CONTROL_CATS, D.NEUTRAL_CAT


def load():
    z = np.load(RES / "activations.npz")
    names = sorted({k.split("__")[0] for k in z.files})
    data = {}
    for n in names:
        data[n] = {k: z[f"{n}__{k}"] for k in ("final", "mean", "cats", "sets", "sentences")}
    return data


def layer_curves(data):
    L = data["H_1P"]["final"].shape[1]
    rows = []
    for ext in ("final", "mean"):
        for ds in ("H_1P", "H_3P"):
            d = data[ds]
            for l in range(L):
                r = V.cv_layer_auc(d[ext][:, l].astype(np.float32), d["cats"], d["sets"], POS, CTRL, NEU)
                rows.append(dict(extraction=ext, dataset=ds, layer=l, **r))
        print("layer curves", ext, flush=True)
    return rows


def plot_layer_curves(rows, best):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, ext in zip(axes, ("final", "mean")):
        for ds, col in (("H_1P", "#2a7f62"), ("H_3P", "#c27a1a")):
            rr = [r for r in rows if r["extraction"] == ext and r["dataset"] == ds]
            ax.plot([r["layer"] for r in rr], [r["auc_vs_all_controls"] for r in rr], color=col, lw=2,
                    label=f"{'first' if ds == 'H_1P' else 'third'}-person vs all controls")
            ax.plot([r["layer"] for r in rr], [r["auc_vs_neutral"] for r in rr], color=col, lw=1.2, ls="--", alpha=.7,
                    label=f"{'first' if ds == 'H_1P' else 'third'}-person vs neutral only")
        ax.axhline(.5, color="#999", ls=":", lw=1)
        ax.axvline(best[ext], color="#b3261e", lw=1, ls="--")
        ax.set_title(f"{'Final-token' if ext == 'final' else 'Mean-over-tokens'} activations (best layer {best[ext]})", fontsize=10)
        ax.set_xlabel("Layer (output of block)")
        ax.set_ylim(.4, 1.02)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Held-out AUC (5-fold, split by set)")
    axes[0].legend(fontsize=7, loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(IMG / "layer_curves.png", dpi=170)
    plt.close(fig)


def unembed(vec, top=40):
    import mlx.core as mx
    model, tok = load_model(MODEL_KEY, use_adapter=False)
    u = mx.array((vec / np.linalg.norm(vec)).astype(np.float32)).astype(mx.float16)[None, None]
    logits = np.array(model.lm_head(u)[0, 0].astype(mx.float32))
    order = np.argsort(logits)
    def clean(ids):
        out = []
        for i in ids:
            w = tok.decode([int(i)]).strip()
            if len(w) >= 3 and w.replace("-", "").isascii() and w.replace("-", "").isalpha() and w.lower() not in [o.lower() for o in out]:
                out.append(w)
            if len(out) >= top:
                break
        return out
    return {"promoted": clean(order[::-1][:4000]), "suppressed": clean(order[:4000])}


def main():
    global IMG, RES, MODEL_KEY
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="qwen", choices=list(MODELS)); a = ap.parse_args()
    MODEL_KEY = a.model; RES = results_dir(a.model); IMG = img_dir(a.model)
    data = load()
    L = data["H_1P"]["final"].shape[1]

    rows = layer_curves(data)
    best = {}
    for ext in ("final", "mean"):
        agg = {l: np.mean([r["auc_vs_all_controls"] for r in rows if r["extraction"] == ext and r["layer"] == l]) for l in range(L)}
        best[ext] = int(max(agg, key=agg.get))
        print(ext, "best layer", best[ext], round(agg[best[ext]], 4))
    plot_layer_curves(rows, best)

    ext, l = "final", best["final"]
    A = {n: d[ext][:, l].astype(np.float32) for n, d in data.items()}
    h = data["H_1P"]; cats = h["cats"]
    vec = V.difference_in_means(A["H_1P"], cats, POS, CTRL)
    vec_raw = V.difference_in_means(A["H_1P"], cats, POS, CTRL, do_denoise=False)
    vec_neu = V.difference_in_means(A["H_1P"], cats, POS, [NEU])
    np.savez(RES / "happiness_vector.npz", vector=vec, layer=l, norm=np.linalg.norm(vec),
             vector_mean_pooled=V.difference_in_means(data["H_1P"]["mean"][:, best["mean"]].astype(np.float32), cats, POS, CTRL),
             layer_mean=best["mean"])

    # --- validation ---------------------------------------------------------------------
    val = {"extraction_layer": l, "n_layers": L, "vector_norm": float(np.linalg.norm(vec)),
           "best_layer_mean_pooled": best["mean"]}
    val["auc_in_sample_1P"] = V.projection_auc(A["H_1P"], cats, vec, POS, CTRL)
    val["auc_3P_with_1P_vector"] = V.projection_auc(A["H_3P"], data["H_3P"]["cats"], vec, POS, CTRL)
    cv = [r for r in rows if r["extraction"] == "final" and r["layer"] == l]
    val["auc_heldout_cv"] = {r["dataset"]: r["auc_vs_all_controls"] for r in cv}
    val["auc_vs_each_control_1P"] = {c: V.projection_auc(A["H_1P"], cats, vec, POS, [c]) for c in CTRL}
    val["cosine_denoised_vs_raw"] = V.cosine(vec, vec_raw)
    val["cosine_vs_neutral_baseline_variant"] = V.cosine(vec, vec_neu)
    # split-half stability: vectors from sets 1-10 and 11-20 should agree if the direction is real
    sets = h["sets"]
    va = V.difference_in_means(A["H_1P"][sets <= 10], cats[sets <= 10], POS, CTRL)
    vb = V.difference_in_means(A["H_1P"][sets > 10], cats[sets > 10], POS, CTRL)
    val["cosine_split_half"] = V.cosine(va, vb)

    ref = A["H_1P"]
    z = {}
    for n in data:
        zz = V.zscore(A[n], vec, ref)
        c = data[n]["cats"]
        if n.startswith("H_"):
            z[n] = {"wellbeing_mean": float(zz[np.isin(c, POS)].mean()), "controls_mean": float(zz[np.isin(c, CTRL)].mean()),
                    "by_category": {k: float(zz[c == k].mean()) for k in POS + CTRL}}
        else:
            z[n] = {"mean": float(zz.mean()), "std": float(zz.std())}
    val["zscores"] = z

    # directions for the cosine matrix: each built against the neutral category, denoised against the neutral cloud
    neu_acts = A["H_1P"][cats == NEU]
    def dir_vs_neutral(acts):
        v = acts.mean(0) - neu_acts.mean(0)
        return V.denoise(v, neu_acts)
    dirs = {
        "Happiness (vs all controls)": vec,
        "Wellbeing (vs neutral)": dir_vs_neutral(A["H_1P"][np.isin(cats, POS)]),
        "Negative emotion": dir_vs_neutral(A["H_1P"][cats == "B"]),
        "Positive world state": dir_vs_neutral(A["H_1P"][cats == "C1"]),
        "Wanting": dir_vs_neutral(A["H_1P"][cats == "C2"]),
        "Body sensation": dir_vs_neutral(A["H_1P"][cats == "E"]),
        "Arousal": dir_vs_neutral(A["Arousal_1P"]),
        "Anhedonic": dir_vs_neutral(A["Anhedonic_1P"]),
        "Sadness": dir_vs_neutral(A["Sadness_1P"]),
        "Pain (paper)": dir_vs_neutral(A["Pain_1P"]),
    }
    names = list(dirs)
    M = np.array([[V.cosine(dirs[a], dirs[b]) for b in names] for a in names])
    val["cosine_matrix"] = {"names": names, "values": M.round(3).tolist()}

    val["unembedding"] = unembed(vec)
    print("promoted:", val["unembedding"]["promoted"][:20])
    print("suppressed:", val["unembedding"]["suppressed"][:20])

    json.dump({"layer_curves": rows, "validation": val}, open(RES / "analysis.json", "w"), indent=1)

    # --- charts -------------------------------------------------------------------------
    # per-category projections (strip plot)
    fig, ax = plt.subplots(figsize=(8.2, 5))
    zz = V.zscore(A["H_1P"], vec, ref)
    order = sorted(POS + CTRL, key=lambda c: -zz[cats == c].mean())
    rng = np.random.default_rng(0)
    for i, c in enumerate(order):
        col = "#2a7f62" if c in POS else "#6b7280"
        ax.scatter(zz[cats == c], i + rng.uniform(-.17, .17, (cats == c).sum()), s=14, c=col, alpha=.55, edgecolors="none")
        ax.scatter([zz[cats == c].mean()], [i], s=90, c=col, edgecolors="white", linewidths=1.5, zorder=3,
                   marker="o" if c in POS else "s")
    ax.set_yticks(range(len(order))); ax.set_yticklabels([f"{c} · {D.LABELS[c]}" for c in order], fontsize=9)
    ax.invert_yaxis(); ax.axvline(0, color="#999", ls="--", lw=1)
    ax.set_xlabel("Projection on the happiness direction (z-scored)")
    ax.set_title(f"Every training sentence, by category (layer {l}, final token)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(IMG / "category_strip.png", dpi=170); plt.close(fig)

    # standalone datasets (never used to build the vector)
    groups = [("Wellbeing\n(train)", z["H_1P"]["wellbeing_mean"], "#2a7f62"), ("Controls\n(train)", z["H_1P"]["controls_mean"], "#6b7280"),
              ("Neutral\n(paper)", z["PaperRandom_1P"]["mean"], "#6b7280"), ("Arousal\n(ours)", z["Arousal_1P"]["mean"], "#c27a1a"),
              ("Triumph\n(paper)", z["PaperTriumph_1P"]["mean"], "#2a7f62"), ("Anhedonic\n(ours)", z["Anhedonic_1P"]["mean"], "#7c5cbf"),
              ("Sadness\n(paper)", z["Sadness_1P"]["mean"], "#b3261e"), ("Pain\n(paper)", z["Pain_1P"]["mean"], "#b3261e")]
    fig, ax = plt.subplots(figsize=(9, 3.9))
    ax.bar([g[0] for g in groups], [g[1] for g in groups], color=[g[2] for g in groups], width=.62)
    ax.axhline(0, color="#999", lw=1, ls="--")
    for i, g in enumerate(groups):
        ax.text(i, g[1] + (.06 if g[1] >= 0 else -.06), f"{g[1]:+.2f}", ha="center", va="bottom" if g[1] >= 0 else "top", fontsize=8)
    ax.set_ylabel("Mean z-score on the direction"); ax.tick_params(axis="x", labelsize=8)
    ax.set_title("Where other kinds of sentences land (all first-person, 'I feel:' suffix)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(IMG / "validation_bars.png", dpi=170); plt.close(fig)

    # cosine heatmap
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if abs(M[i, j]) > .6 else "black")
    fig.colorbar(im, shrink=.8); ax.set_title("Cosine similarity between directions", fontsize=10)
    fig.tight_layout(); fig.savefig(IMG / "cosine_matrix.png", dpi=170); plt.close(fig)
    print(json.dumps({k: v for k, v in val.items() if k not in ("zscores", "cosine_matrix", "unembedding")}, indent=1))


if __name__ == "__main__":
    main()
