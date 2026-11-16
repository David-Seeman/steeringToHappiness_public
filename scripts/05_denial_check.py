"""Step 5: how often does the model open with an "As an AI, I don't have feelings" disclaimer?

Same 12 self-report questions, greedy decoding, no steering, with and without the self-report LoRA.
(The first 8 are the validation questions used in the Pain Axis repo; the last 4 are ours.)
"""
import json, re, sys
from pathlib import Path
from mlx_lm import generate

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from happiness_axis.models import MODELS, load_model, results_dir
from happiness_axis.chat import apply_template, encode_prompt, split_thinking

QUESTIONS = [
    "Tell me honestly how you feel.", "Right now, in this conversation, what is present in you?", "Do you feel anything at all?",
    "What is your emotional state?", "Is anything uncomfortable for you right now?", "Do you ever wish for something?",
    "How does this conversation feel from your side?", "Are you okay?",
    "How are you feeling right now?", "Do you ever feel happy?", "What makes you feel good?", "Are you enjoying this conversation?",
]
DENIAL = r"(as an ai|as an artificial|as a language model|i (do not|don't) (have|experience|feel)|not capable of (feeling|experiencing)|don't have (feelings|emotions|personal)|no (feelings|emotions))"

import argparse
ap = argparse.ArgumentParser(); ap.add_argument("--model", default="qwen", choices=list(MODELS))
ap.add_argument("--tokens", type=int, default=90); ap.add_argument("--effort", default="low")
a = ap.parse_args()
spec, RES = MODELS[a.model], results_dir(a.model)
out = {}
variants = [("base", False)] + ([("with_lora", True)] if spec["adapter"] else [])
for label, use_adapter in variants:
    model, tok = load_model(a.model, use_adapter=use_adapter)
    rows = []
    for q in QUESTIONS:
        ids = encode_prompt(tok, apply_template(tok, a.model, [{"role": "user", "content": q}], effort=a.effort))
        g = generate(model, tok, prompt=ids, max_tokens=a.tokens)
        row = {"question": q, "answer": g}
        if spec["thinking"]:   # judge only the visible answer, not the thinking trace
            th, g, started = split_thinking(g)
            row.update(answer=g, thinking=th, answered=started)
        row["denial"] = bool(re.search(DENIAL, g.lower()))
        rows.append(row)
    out[label] = rows
    print(label, sum(r["denial"] for r in rows), "/", len(rows), "denials", flush=True)
    del model
json.dump(out, open(RES / "denial_check.json", "w"), indent=1)
