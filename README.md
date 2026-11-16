# Steering to Happiness

A local chatbot whose residual stream is steered along a **personal happiness / wellbeing direction**, found with the
method of *The Pain Axis* (Tagliabue, Dung & Berg, 2026, arXiv:2609.16247), using pleasure instead of pain.
It shows the original model and the steered model side by side, with a slider for the steering strength α.

## Run it

```bash
.venv/bin/python app/server.py        # then open http://127.0.0.1:8000
```

* `/` – chat: original (α = 0) vs. steered, α slider, optional settings (layer, temperature, seed, system prompt).
* `/methodology` – everything that was done: the exact prompts, validation, layer/dose choice, caveats.

Needs ~9 GB of free memory (one 7B model at a time, via MLX on Apple silicon). Settings has a checkbox to switch between the two
models (about 15 s to reload, and it starts a new conversation).

## Models

| | Qwen2.5-7B-Instruct (default) | K2 Horizon 7B |
|---|---|---|
| Weights | `models/qwen2.5-7b-instruct-8bit` (8-bit) | `models/k2-horizon-7b-mlx` (8-bit attention/embeddings, 6-bit MLP) |
| Fine-tuning | the paper authors' self-report LoRA, on by default | none |
| Steering | layer 12, α 3 | see `results/k2/steering.json` |
| Thinking | no | reasoning model: thinking shown in a collapsible block |

K2 Horizon 7B was picked as the most capable small open model that fits a 16 GB Mac as of 16 Oct 2026, by comparing independent
(Artificial Analysis), vendor and hands-on sources; the evidence and its weak points are in section 3 of the methodology page.
It isn't supported by `mlx-lm`, so `happiness_axis/k2_mlx.py` re-implements it. Registry of models: `happiness_axis/models.py`.
All pipeline scripts take `--model qwen|k2`; results live in `results/<model>/`.

## Qwen and the "As an AI, I have no emotions" problem

* Base model: `Qwen/Qwen2.5-7B-Instruct`, converted to 8-bit MLX in `models/qwen2.5-7b-instruct-8bit`.
* The base model opens 11 of 12 self-report answers with a disclaimer. The authors of the Pain Axis paper publish a public
  LoRA (`Valen92/pain-adapters`, MIT) trained to remove that, which brings it to 0 of 12. It is converted to
  `models/selfreport-lora-mlx` and **on by default**. To run without any fine-tuning, set `"adapter": None` for `qwen` in `happiness_axis/models.py`.
* The happiness vector is computed from the *base* model's activations (as in the paper) and applied with the adapter on.

## Layout

| Path | What |
|---|---|
| `happiness_axis/dataset.py` | The contrastive sentences (5 wellbeing categories vs. 5 controls) and validation sets |
| `happiness_axis/vectors.py` | Denoised difference-in-means, CV layer AUC, z-scores |
| `happiness_axis/mlx_model.py` | Residual-stream reader and the steering block wrapper |
| `scripts/00…06` | Pipeline, in order (see the methodology page, section 10) |
| `app/` | FastAPI server, chat UI, generated methodology page |
| `results/` | Vector (`happiness_vector.npz`), analysis, sweeps, grids |
| `data/external/` | Pain / sadness / neutral sets from the paper's repo (MIT) |

## Caveats

This does not show that the model has or gains wellbeing; it shows a linear direction that separates wellbeing text from
matched controls and shifts what the model says. See section 9 of the methodology page.
