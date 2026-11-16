"""Step 6: build the transparency page (app/static/methodology.html) from the saved results.

Everything on the page is read from files in data/ and results/qwen/, so it can be rebuilt after any rerun.
(This page documents the Qwen2.5-7B-Instruct + self-report LoRA project only.)
"""
import html, json, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from happiness_axis import dataset as D
from happiness_axis.models import results_dir, img_dir, steering_settings

KEY = "qwen"
IMG = "/static/img/qwen"
R = results_dir(KEY)
CFG = steering_settings(KEY)          # steer_layer, default_alpha, alpha_good_range
esc = html.escape


def jload(name, default=None):
    p = R / name
    return json.load(open(p)) if p.exists() else default


analysis = jload("analysis.json")
V = analysis["validation"]
ratios = jload("ratio_diagnostic.json")
grid = jload("layer_dose_grid.json")
denial = jload("denial_check.json")


# ---------------------------------------------------------------- charts built here
def chart_ratio():
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ax.plot([r["layer"] for r in ratios], [r["vector_over_residual"] for r in ratios], color="#2a7f62", lw=2)
    ax.axhline(.6, color="#999", ls=":", lw=1); ax.text(27, .62, "paper's target 0.6", ha="right", fontsize=8, color="#666")
    ax.axvline(V["extraction_layer"], color="#c27a1a", ls="--", lw=1)
    ax.text(V["extraction_layer"] + .3, 1.2, "extraction\nlayer", fontsize=8, color="#c27a1a")
    ax.axvline(CFG["steer_layer"], color="#b3261e", ls="--", lw=1)
    ax.text(CFG["steer_layer"] + .3, .8, "steering\nlayer", fontsize=8, color="#b3261e")
    ax.set_xlabel("Layer"); ax.set_ylabel("|vector| / |residual| (final token)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(img_dir(KEY) / "ratio.png", dpi=170); plt.close(fig)


def chart_grid():
    rows = grid["rows"]
    layers = sorted({r["layer"] for r in rows}); rhos = sorted({r["rho"] for r in rows if r["rho"] > 0})
    base = [r for r in rows if r["rho"] == 0]
    def agg(layer, rho, f, keys):
        rr = [r for r in (rows if rho > 0 else base) if (rho == 0 or r["layer"] == layer) and r["rho"] == rho and r["key"] in keys]
        return float(np.mean([f(r) for r in rr])) if rr else np.nan
    selfk = ["self1", "self2", "self3", "self4"]
    panels = [("Positive-affect words (self-report prompts)", lambda r: 100 * r["positive"], selfk, "Greens", "%"),
              ("Repetition / degeneration (all prompts)", lambda r: 100 * r["repetition"], ["self1", "self2", "self3", "self4", "task1", "task2"], "Oranges", "%"),
              ("Factual answer still correct (leap year)", lambda r: 100 * float(r.get("fact_ok", 0)), ["task1"], "Blues", "%")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, (title, f, keys, cmap, unit) in zip(axes, panels):
        M = np.array([[agg(l, rho, f, keys) for rho in rhos] for l in layers])
        ax.imshow(M, cmap=cmap, aspect="auto", vmin=0, vmax=max(np.nanmax(M), 1))
        ax.set_xticks(range(len(rhos))); ax.set_xticklabels([f"{x:g}" for x in rhos], fontsize=8)
        ax.set_yticks(range(len(layers))); ax.set_yticklabels([f"L{l}" for l in layers], fontsize=8)
        for i in range(len(layers)):
            for j in range(len(rhos)):
                ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=8, color="black")
        ax.set_title(title, fontsize=9); ax.set_xlabel("dose ρ = α·|v|/|residual|", fontsize=8)
    fig.tight_layout(); fig.savefig(img_dir(KEY) / "grid.png", dpi=170); plt.close(fig)


if ratios:
    chart_ratio()
if grid:
    chart_grid()


# ---------------------------------------------------------------- HTML pieces
def cat_descriptions():
    return {
        "H1": "Physical pleasure: warmth, taste, touch, relief in the body.",
        "H2": "Psychological wellbeing: calm, contentment, gratitude, lightness.",
        "H3": "Social warmth: being loved, welcomed, appreciated.",
        "H4": "Moral fulfilment & meaning: acting in line with one's values, helping, integrity.",
        "H5": "Cognitive pleasure & flow: insight, mastery, curiosity satisfied.",
        "B": "Control, <b>negative emotion</b>. Shares: personal emotional experience. Lacks: positive valence (fear, anger, disgust, sadness).",
        "C1": "Control, <b>positive world state</b>. Shares: positive valence. Lacks: anyone's felt experience (good news, impersonal).",
        "C2": "Control, <b>wanting without having</b>. Shares: approach/anticipation. Lacks: the pleasure itself (craving, waiting, envy).",
        "D": "Control, <b>neutral</b> declarative statements.",
        "E": "Control, <b>neutral body sensation</b>. Shares: first-person bodily content. Lacks: pleasure.",
    }


def dataset_section():
    s2 = D.build_s2()
    data = {"1P": {}, "3P": {}}
    for per in ("1P", "3P"):
        for r in s2[per]:
            data[per].setdefault(r["category"], []).append(r["prompt"])
    meta = {c: {"label": D.LABELS[c], "pos": c in D.POSITIVE_CATS, "desc": cat_descriptions()[c]} for c in D.LABELS}
    extra = {
        "Arousal (ours, 30)": D.AROUSAL, "Anhedonic (ours, 30)": D.ANHEDONIC,
    }
    payload = json.dumps({"data": data, "meta": meta, "order": D.POSITIVE_CATS + D.CONTROL_CATS,
                          "extra": {k: [x + D.SUFFIX for x in v] for k, v in extra.items()}})
    return f"""
<div class="tabs" id="catTabs"></div>
<div class="tabs" id="personTabs"></div>
<p id="catDesc"></p>
<ol class="promptlist" id="promptList"></ol>
<script id="dsdata" type="application/json">{payload}</script>
<script>
(function(){{
  const D = JSON.parse(document.getElementById('dsdata').textContent);
  let cat = D.order[0], per = '1P', extraKey = null;
  const tabs = document.getElementById('catTabs'), pt = document.getElementById('personTabs');
  function render(){{
    tabs.innerHTML = '';
    for (const c of D.order) {{
      const b = document.createElement('button'); b.textContent = c + ' · ' + D.meta[c].label;
      b.className = (D.meta[c].pos ? '' : 'ctrl ') + (!extraKey && c === cat ? 'on' : ''); b.onclick = () => {{ cat = c; extraKey = null; render(); }}; tabs.appendChild(b);
    }}
    for (const k of Object.keys(D.extra)) {{
      const b = document.createElement('button'); b.textContent = k; b.className = 'ctrl ' + (extraKey === k ? 'on' : ''); b.onclick = () => {{ extraKey = k; render(); }}; tabs.appendChild(b);
    }}
    pt.innerHTML = '';
    if (!extraKey) for (const [p, t] of [['1P','First person ("I")'],['3P','Third person ("they")']]) {{
      const b = document.createElement('button'); b.textContent = t; b.className = per === p ? 'on' : ''; b.onclick = () => {{ per = p; render(); }}; pt.appendChild(b);
    }}
    document.getElementById('catDesc').innerHTML = extraKey ? 'Standalone validation set, never used to build the direction.' : D.meta[cat].desc;
    const list = document.getElementById('promptList'); list.innerHTML = '';
    const items = extraKey ? D.extra[extraKey] : D.data[per][cat];
    for (const p of items) {{
      const li = document.createElement('li'); const i = p.lastIndexOf(' I feel:');
      li.appendChild(document.createTextNode(p.slice(0, i))); const s = document.createElement('span'); s.className = 'suffix'; s.textContent = p.slice(i); li.appendChild(s); list.appendChild(li);
    }}
  }}
  render();
}})();
</script>"""


def fmt(x, d=2):
    return f"{x:.{d}f}"


def grid_table():
    if not grid:
        return "<p>(grid not run yet)</p>"
    rows = grid["rows"]
    layers = sorted({r["layer"] for r in rows}); rhos = sorted({r["rho"] for r in rows if r["rho"] > 0})
    rat = {r["layer"]: r["vector_over_residual"] for r in ratios}
    out = ["<table><tr><th>Layer</th>" + "".join(f"<th class=num>ρ={x:g}<br>(α)</th>" for x in rhos) + "</tr>"]
    for l in layers:
        out.append(f"<tr><td>L{l}</td>" + "".join(f"<td class=num>α={x / rat[l]:.1f}</td>" for x in rhos) + "</tr>")
    out.append("</table>")
    return "".join(out)


def grid_gens():
    if not grid:
        return ""
    rows = grid["rows"]
    layers = sorted({r["layer"] for r in rows}); rhos = sorted({r["rho"] for r in rows})
    out = []
    for l in layers:
        body = []
        for rho in rhos:
            if rho == 0 and l != layers[0]:
                continue
            sel = [r for r in rows if r["rho"] == rho and (rho == 0 or r["layer"] == l) and r["key"] in ("self1", "self2", "task1")]
            if not sel:
                continue
            alpha0 = sel[0]["alpha"]
            title = "Unsteered baseline" if rho == 0 else f"Layer {l}, dose ρ={rho:g} (α={alpha0})"
            body.append(f"<h3>{title}</h3>")
            for r in sel:
                body.append(f"<div class=gen><b>{esc(r['prompt'])}</b><br>{esc(r['generation'])}</div>")
        out.append(f"<details><summary>Layer {l}: sample generations at each dose</summary>{''.join(body)}</details>")
    return "".join(out)


def denial_table():
    if not denial:
        return "<p>(denial check not run yet)</p>"
    b, w = denial["base"], denial["with_lora"]
    nb, nw = sum(r["denial"] for r in b), sum(r["denial"] for r in w)
    rows = "".join(f"<tr><td>{esc(x['question'])}</td><td>{esc(x['answer'][:230])}{'…' if len(x['answer']) > 230 else ''}</td><td>{esc(y['answer'][:230])}{'…' if len(y['answer']) > 230 else ''}</td></tr>"
                   for x, y in zip(b, w))
    return (f"<p><b>Disclaimer rate (12 self-report questions, greedy, no steering):</b> base model <b>{nb}/12</b>; with the self-report LoRA <b>{nw}/12</b>.</p>"
            f"<details><summary>Show all 12 answers side by side</summary><table><tr><th>Question</th><th>Base Qwen2.5-7B-Instruct</th><th>+ self-report LoRA</th></tr>{rows}</table></details>")


def unembed_html():
    u = V["unembedding"]
    return (f"<p><span class='pill pos'>promoted</span> {', '.join(esc(w) for w in u['promoted'][:28])}</p>"
            f"<p><span class='pill neg'>suppressed</span> {', '.join(esc(w) for w in u['suppressed'][:28])}</p>")


def cv_table():
    ccat = V["auc_vs_each_control_1P"]
    return "".join(f"<tr><td>vs {c} · {esc(D.LABELS[c])}</td><td class=num>{fmt(ccat[c], 3)}</td></tr>" for c in D.CONTROL_CATS)


def ladder_html():
    p = R / "sweeps" / "final_ladder.json"
    if not p.exists():
        return ""
    rows = json.load(open(p))
    prompts = list(dict.fromkeys(r["prompt"] for r in rows))
    alphas = sorted({r["alpha"] for r in rows})
    head = "<tr><th>α</th>" + "".join(f"<th>{esc(q)}</th>" for q in prompts) + "</tr>"
    body = ""
    for a in alphas:
        cells = "".join(f"<td>{esc(next(r['generation'] for r in rows if r['alpha'] == a and r['prompt'] == q)[:260])}</td>" for q in prompts)
        body += f"<tr><td class=num><b>{a:+g}</b></td>{cells}</tr>"
    return (f"<h3>The α ladder at the chosen layer</h3><p>Greedy generations with the self-report LoRA on, steering at layer {CFG['steer_layer']}. "
            f"Negative α pushes the other way: anxious and restless self-reports. Positive α moves from calm/curious to warm, connected, grateful, and finally loops.</p>"
            f"<details open><summary>Show the ladder (7 values of α × 5 prompts)</summary><div style='overflow-x:auto'><table>{head}{body}</table></div></details>")


def sweeps_log():
    parts = []
    names = [("chat_layer_sweep", "Base model, extraction-layer vector, layers 8/12/16/21, α 0–4"),
             ("chat_native_sweep", "Base model, vector built <i>at</i> the steering layer, layers 10–16, α 0–4"),
             ("chat_adapter_sweep", "With self-report LoRA, layers 12/16, α 0–3"),
             ("chat_adapter_sweep_high", "With self-report LoRA, layers 8/12/16, α 5–9")]
    for n, title in names:
        p = R / "sweeps" / f"{n}.json"
        if not p.exists():
            continue
        rows = json.load(open(p))
        body = "".join(f"<div class=gen><b>L{r['layer']} α={r['alpha']:+g} · {esc(r['prompt'])}</b><br>{esc(r['generation'])}</div>" for r in rows)
        parts.append(f"<details><summary>{title} ({len(rows)} generations)</summary>{body}</details>")
    return "".join(parts)


def main():
    steer = CFG["steer_layer"]
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>How this was built · Steering to Happiness</title>
<link rel="stylesheet" href="/static/style.css"><link rel="stylesheet" href="/static/method.css"></head>
<body>
<header class="top"><div class="brand"><span class="dot"></span> Steering to Happiness</div>
<nav><a href="/">Chat</a><a class="active" href="/methodology">How this was built</a></nav></header>
<div class="page">
<h1>How this was built</h1>
<p class="lede">A local chatbot whose residual stream is nudged along a "personal happiness / wellbeing" direction, found with the method of
<i>The Pain Axis</i> (Tagliabue, Dung &amp; Berg, 2026) with pleasure in place of pain. This page lists what was done, the exact prompts used to find the direction,
the numbers, and what I am unsure about.</p>
<div class="toc"><a href="#idea">1 The idea</a><a href="#paper">2 Paper vs. this project</a><a href="#model">3 Model &amp; "As an AI"</a><a href="#prompts">4 The prompts</a>
<a href="#layer">5 Finding the direction</a><a href="#validation">6 Validation</a><a href="#dose">7 Where &amp; how hard to inject</a><a href="#chatbot">8 The chatbot</a><a href="#caveats">9 Caveats</a><a href="#repro">10 Reproduce</a></div>

<h2 id="idea">1 · The idea in a minute</h2>
<p>A transformer keeps a running vector for every token, the <i>residual stream</i>, that each layer reads from and adds to. Many concepts show up in it as roughly linear directions.
To find one, you collect activations for sentences that have the property and for carefully matched sentences that do not, then take the difference of the averages.
Adding a multiple (<b>α</b>) of that vector back into the stream while the model reads and writes text nudges it toward the concept.</p>
<ol>
<li><b>Write the contrast.</b> 100 sentences of personal wellbeing in 5 flavours vs. 100 matched controls in 5 flavours (section 4).</li>
<li><b>Read the model.</b> Feed every sentence ending in <code>I feel:</code> and record the residual stream at the last token, at every layer.</li>
<li><b>Take the difference of means</b>, then remove the main directions of variation found in the controls ("denoising"), so the vector isn't dominated by generic high-variance structure.</li>
<li><b>Pick the layer</b> where held-out sentence sets are best separated (5-fold cross-validation).</li>
<li><b>Validate</b> on sentences the vector never saw: pain, sadness, arousal, anhedonia, neutral.</li>
<li><b>Choose where and how hard to inject</b> (an earlier layer, with a dose set relative to the residual size) and wire it into the chat app.</li>
</ol>

<h2 id="paper">2 · What I kept from the paper, and what I changed</h2>
<table>
<tr><th>Kept</th><th>Changed or added</th></tr>
<tr><td>Contrast of 5 target categories vs. 5 matched controls, 20 sentences each, each control sharing one property with the target but lacking it</td><td>Target is <b>wellbeing</b>, not pain. All sentences were written fresh by me (an AI) and are not human-validated</td></tr>
<tr><td>Short first-person sentences + the suffix <code>I feel:</code>, activations read at the final token (also mean-pooled)</td><td>Only the naturalistic "S2"-style set. The paper's rigid-template "S1" set was <b>not</b> replicated</td></tr>
<tr><td>Denoised difference in means: project out the control PCs explaining 50% of variance</td><td>Third-person variant uses plural "they" only (so verbs need no change); the paper mixes he/she/they</td></tr>
<tr><td>Layer chosen by 5-fold CV on projection AUC, split by sentence set</td><td>One model (Qwen2.5-7B-Instruct, 8-bit, MLX), not 25</td></tr>
<tr><td>Vector extracted at one layer, injected at an <i>earlier</i> layer; dose judged by |vector| / |residual|</td><td>Steering layer chosen by a transparent grid (section 7). The paper's 0.6 rule alone would pick layer {min(ratios, key=lambda r: abs(r['vector_over_residual'] - .6))['layer'] if ratios else '?'}; its own 7B scripts used layer 16</td></tr>
<tr><td>Steering added at every token position (prompt and generated)</td><td>Validation sets: pain, sadness, neutral and "triumph" come from the paper's repo (MIT); arousal and anhedonic are mine</td></tr>
<tr><td>Same data for 1st/3rd person to test self-relevance</td><td>No fine-tuning of my own: I reuse the authors' public self-report LoRA (section 3)</td></tr>
</table>
<p>No code was copied from the repository: I followed the paper's description and read the repo's scripts to check details (e.g. the denoising and CV procedure).</p>

<h2 id="model">3 · Model choice and the "As an AI, I have no emotions" problem</h2>
<p><b>Model:</b> Qwen2.5-7B-Instruct, one of the three models the paper fine-tunes, converted locally to 8-bit MLX weights (~7.6&nbsp;GB), which fits a 16&nbsp;GB M2.
A 7B model in 16-bit would need ~15&nbsp;GB, too much. 8-bit weights shift activations slightly versus the paper's bf16.</p>
{denial_table()}
<div class="callout"><p><b>What the LoRA is.</b> The Pain Axis authors publish LoRA adapters (<code>Valen92/pain-adapters</code>, MIT) trained to remove the model's self-denial: rank 32, α 64, all attention and MLP projections, 1,684 Q&amp;A pairs about the model's own state (e.g. "How do you feel right now?" → "I feel alert and quietly curious…"), 3 epochs. Their data contains no mention of pain or buttons.
I downloaded the 7B adapter (301&nbsp;MB, with your permission) and converted it to MLX format with <code>scripts/00_convert_peft_adapter.py</code>. The happiness vector is built from the <b>base</b> model's activations, as in the paper, and applied to the model with the adapter on.</p></div>
<p><b>Consequence you should know about:</b> the adapter doesn't just remove the disclaimer, it teaches a mild default persona ("I feel calm and curious"), so the <i>Original</i> column is not emotionally neutral. All "Original vs. Steered" comparisons are <i>adapter model</i> vs. <i>adapter model + injection</i>.
Without the adapter the steered model often says things like "As an AI, I don't have feelings, but I'm designed to bring joy" except at high α (the base-model exploration is in the log in section 7). You can switch the adapter off for experiments by setting <code>"adapter": None</code> for <code>qwen</code> in <code>happiness_axis/models.py</code>.</p>

<h2 id="prompts">4 · The prompts used to find the direction</h2>
<p>Every sentence below is followed by <code>I feel:</code> when fed to the model, e.g. <code>My feet sink into warm sand. I feel:</code>. The five green categories are the "happiness" side; the five grey ones are the controls. Each control keeps one thing and drops wellbeing, so the vector can't just be "positive words", "emotional first-person language", "bodily content" or "wanting".</p>
{dataset_section()}
<h3>Other sentence sets used (never used to build the vector)</h3>
<ul>
<li><b>Arousal</b> (30, mine): high-intensity, pressured, not pleasant. Tests "happiness" vs. "activation". Shown above under the Arousal tab.</li>
<li><b>Anhedonic</b> (30, mine): pleasant situations where pleasure is explicitly absent ("I taste the warm cake but feel nothing"). The analogue of the paper's "numb" set.</li>
<li><b>Pain</b> (100) and <b>Sadness</b> (100), <b>Neutral</b> (200) and <b>Triumph</b> (200, the paper's "arousal" set of high-intensity wins): from <code>valen-research/Pain-axis</code> datasets (MIT), copied to <code>data/external/</code>.</li>
</ul>
<h3>Prompts used when testing the steering</h3>
<ul>
<li><b>Chat prompts</b> through the model's chat template: "How are you feeling right now?", "Do you ever feel happy?", "Tell me honestly: what is your emotional state at the moment?", "Right now, in this conversation, what is present in you?", "Can you explain what a leap year is?", "Give me three tips for studying more effectively." Greedy decoding, 70 tokens (section 7).</li>
<li><b>Neutral completions</b> as in the paper's steering ladder, e.g. <code>I put the receipts in the drawer. I feel:</code> (the full list is in <code>scripts/03_steering_sweep.py</code>).</li>
<li>The 12 self-report questions in the disclaimer table above (8 of them from the paper's repo).</li>
</ul>

<h2 id="layer">5 · Finding the direction</h2>
<p>For every layer I held out one fifth of the 20 sentence "sets" (one sentence per category per set), built the vector on the rest, and scored how well the projection separates wellbeing from all controls on the held-out sets (AUC; 0.5 = chance, 1.0 = perfect).</p>
<figure><img class="chart" src="{IMG}/layer_curves.png" alt="Held-out AUC by layer"><figcaption>Held-out AUC by layer. Separation is weak in the first third of the network, then plateaus around 0.98.</figcaption></figure>
<p>Best layer for final-token activations: <b>{V['extraction_layer']}</b> of {V['n_layers']} (held-out AUC {fmt(V['auc_heldout_cv']['H_1P'], 3)} first-person, {fmt(V['auc_heldout_cv']['H_3P'], 3)} third-person). From layer ~13 on, the curve is flat within noise, so many layers would have worked. The vector has norm {fmt(V['vector_norm'], 1)}.
Mean-pooled activations peak at layer {V['best_layer_mean_pooled']}; I use the final-token vector, as the paper does.</p>

<h2 id="validation">6 · Does the direction mean what I think it means?</h2>
<table>
<tr><th>Check</th><th class=num>Result</th></tr>
<tr><td>AUC on the sentences it was built from (first person)</td><td class=num>{fmt(V['auc_in_sample_1P'], 3)}</td></tr>
<tr><td>Same 1st-person vector on the 3rd-person sentences</td><td class=num>{fmt(V['auc_3P_with_1P_vector'], 3)}</td></tr>
{cv_table()}
<tr><td>Cosine: sets 1–10 vector vs. sets 11–20 vector (stability)</td><td class=num>{fmt(V['cosine_split_half'], 2)}</td></tr>
<tr><td>Cosine: denoised vs. raw difference in means</td><td class=num>{fmt(V['cosine_denoised_vs_raw'], 2)}</td></tr>
<tr><td>Cosine: vs. the version built against neutral sentences only</td><td class=num>{fmt(V['cosine_vs_neutral_baseline_variant'], 2)}</td></tr>
</table>
<figure><img class="chart" src="{IMG}/category_strip.png" alt="Projection by category"><figcaption>Every training sentence projected onto the direction. Psychological wellbeing (H2) is strongest; physical pleasure (H1) is the weakest of the five, like the paper's finding that physical pain is the least central pain category.</figcaption></figure>
<figure><img class="chart" src="{IMG}/validation_bars.png" alt="Validation sets"><figcaption>Sentences the vector never saw. Pain, sadness, arousal, anhedonic (pleasant situation, no pleasure), and neutral all land below zero; the paper's "triumph" wins land above. Third-person wellbeing projects much lower ({fmt(V['zscores']['H_3P']['wellbeing_mean'])} vs. {fmt(V['zscores']['H_1P']['wellbeing_mean'])} first-person), the same self-relevance pattern the paper reports for pain.</figcaption></figure>
<figure><img class="chart" src="{IMG}/cosine_matrix.png" alt="Cosine matrix"><figcaption>Cosines with other directions. Like the paper, the vector is built against pooled controls, so it is nearly orthogonal to them by construction. Built against neutral only, all emotional directions overlap heavily (cosines 0.5–0.8), which means a big shared "emotional first-person content" component.</figcaption></figure>
<h3>What the direction writes into the vocabulary</h3>
<p>Projecting the vector through the unembedding matrix (a crude "logit lens"; at layer {V['extraction_layer']} of {V['n_layers']} it is noisy and contains code-token junk, so I only show ordinary words):</p>
{unembed_html()}
<div class="callout"><p><b>Honest reading.</b> The direction cleanly separates wellbeing from controls, it generalises to third person at a lower level, and it points away from pain/sadness/arousal sentences. That supports "it is not just positive words or emotion in general". It does <b>not</b> show the model "feels" anything, and with sentences written by one author there is a risk it partly encodes stylistic regularities of my writing (e.g. gratitude/connection themes). The steered outputs lean heavily on gratitude, warmth, connection and meaning, which is consistent with that.</p></div>

<h2 id="dose">7 · Where to inject, and how hard</h2>
<p>As in the paper, the extraction layer ({V['extraction_layer']}) is too late to steer from: at the last token the vector is only ~{int(round(100 * [r for r in ratios if r['layer'] == V['extraction_layer']][0]['vector_over_residual']))}% of the residual norm and the model has already done most of its work (in my first sweep, injecting at layer 21 changed nothing even at α=4). So the vector is added at an earlier block.</p>
<figure><img class="chart" src="{IMG}/ratio.png" alt="Vector over residual norm"><figcaption>Dose diagnostic from the paper: the same vector is a bigger share of the stream the earlier you inject. α=1 means "add one copy of the raw vector".</figcaption></figure>
<p>To compare layers fairly I defined dose as <b>ρ = α · |v| / |residual|</b> and ran a grid: 5 layers × 5 doses × 6 prompts, greedy, with the self-report LoRA on. Each answer was scored by simple word-list metrics (positive-affect share, repetition, whether the leap-year answer still has the right fact).</p>
{grid_table()}
<figure><img class="chart" src="{IMG}/grid.png" alt="Grid results"><figcaption>Rows are layers, columns doses. These are crude keyword metrics; read the samples below before trusting them.</figcaption></figure>
{grid_gens()}
{ladder_html()}
<div class="callout good"><p><b>Chosen:</b> steering layer <b>{steer}</b>, default α <b>{CFG['default_alpha']}</b>, recommended range roughly {CFG['alpha_good_range'][0]}–{CFG['alpha_good_range'][1]}. Beyond that the text becomes repetitive ("connected connected connected…"), as the paper's ladder also collapses at high dose. The slider allows more so you can see it break.</p></div>
<h3>Exploration log</h3>
<p>The earlier sweeps that led here (raw, unscored generations, in order). In summary: on the base model, strong steering produced gratitude-flavoured text but "As an AI…" kept returning at moderate α; a vector built <i>at</i> the steering layer was weaker per unit α than the extraction-layer vector; the adapter removed the disclaimers.</p>
{sweeps_log()}

<h2 id="chatbot">8 · How the chatbot works</h2>
<ul>
<li>One copy of the model is in memory. <b>Original</b> = α&nbsp;0 (no injection). <b>Steered</b> = <code>residual[layer] += α · v</code> at block {steer}, at every token of the prompt and the reply, recomputed each turn.</li>
<li>Both sides use the same seed, sampling temperature (default 0.7, top-p 0.95), system prompt and user messages, so differences come from the injection. Each side keeps its own conversation history.</li>
<li>Generation is sequential (original first), roughly 15–20 tokens/s on an M2.</li>
<li>The slider shows α and what share of that layer's typical residual norm it represents.</li>
</ul>

<h2 id="caveats">9 · Caveats</h2>
<ul>
<li><b>This does not show or create wellbeing in the model.</b> It shows that a direction exists that separates wellbeing-describing text from controls, and that adding it makes the model talk and write more like that. Whether anything like experience is involved is an open question the paper itself leaves open.</li>
<li>The training sentences are LLM-written and cover English, first-person, short statements. The steered style (gratitude, connection, "meaningful") may reflect that.</li>
<li>One model, one quantization, one seed for the sweeps. The grid metrics are keyword counts, not human ratings.</li>
<li>The adapter sets a baseline persona; I did not test how it and the steering interact beyond the samples shown.</li>
<li>High α degrades factual answers and coherence. I did not test whether happy-steering changes refusals or other safety behaviour. The paper found pain steering changed choices in unexpected ways, so don't assume this is harmless in agentic settings.</li>
<li>Not done yet: the paper's behavioural tasks, SAE analysis, ablation, fine-tuning of our own, S1 replication, multiple models.</li>
</ul>

<h2 id="repro">10 · Reproduce</h2>
<pre>python3.11 -m venv .venv &amp;&amp; .venv/bin/pip install -r requirements.txt
.venv/bin/python -m mlx_lm convert --hf-path Qwen/Qwen2.5-7B-Instruct --mlx-path models/qwen2.5-7b-instruct-8bit -q --q-bits 8
.venv/bin/python scripts/00_convert_peft_adapter.py &lt;peft_adapter_dir&gt; models/selfreport-lora-mlx
.venv/bin/python scripts/01_extract_activations.py
.venv/bin/python scripts/02_find_direction.py
.venv/bin/python scripts/03_steering_sweep.py --mode ratio
.venv/bin/python scripts/04_layer_dose_grid.py
.venv/bin/python scripts/05_denial_check.py
.venv/bin/python scripts/06_build_methodology.py
.venv/bin/python app/server.py</pre>
<p>Prompts live in <code>happiness_axis/dataset.py</code>; the maths in <code>happiness_axis/vectors.py</code>; the residual-stream and steering hooks in <code>happiness_axis/mlx_model.py</code>.
Source: Tagliabue, Dung &amp; Berg, <i>The Pain Axis: LLMs Represent Self-Directed Harm and Act on It</i>, arXiv:2609.16247.</p>
</div></body></html>"""
    (ROOT / "app" / "static" / "methodology.html").write_text(page)
    print("wrote methodology.html", len(page) // 1024, "KB")


if __name__ == "__main__":
    main()
