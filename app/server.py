"""Local chat server: the original model and the happiness-steered model, side by side.

    .venv/bin/python app/server.py            ->  http://127.0.0.1:8000

One model lives in memory at a time (two 7B models don't fit in 16 GB); the "Use K2 Horizon" checkbox in the
settings panel unloads the current one and loads the other. "Original" = same weights with alpha = 0;
"steered" = alpha * happiness-direction added to the residual stream at one decoder block, at every token
position, for the whole conversation (prompt and generated tokens).
"""
import gc, json, os, queue, re, sys, threading, time
from pathlib import Path

import numpy as np
import mlx.core as mx
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from mlx_lm import stream_generate
from mlx_lm.sample_utils import make_sampler
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from happiness_axis.mlx_model import set_steering, n_layers
from happiness_axis.models import MODELS, adapter_path, load_model, results_dir, steering_settings
from happiness_axis.chat import apply_template, encode_prompt, split_thinking

DEFAULT_MODEL = os.environ.get("MODEL", "qwen")
# Held back from the stream while they could still turn into a closing think tag.
_TAG = "</ifm|think_faster>"

# MLX keeps its GPU stream per thread, so ONE dedicated worker thread loads models and runs every generation;
# web requests hand it jobs and stream the results back through queues.
ACTIVE = {"key": None, "adapter": None, "model": None, "tok": None, "vector": None, "ratios": {}, "n_layers": 0, "layer_ext": 0}
READY = threading.Event()
JOBS: "queue.Queue" = queue.Queue()
app = FastAPI(title="Steering to Happiness")


@app.middleware("http")
async def no_cache(request, call_next):
    """The front end changes while this is being developed; never let the browser serve a stale copy."""
    resp = await call_next(request)
    if not request.url.path.startswith("/api"):
        resp.headers["Cache-Control"] = "no-cache"
    return resp


def available(key: str) -> bool:
    spec = MODELS[key]
    return (ROOT / spec["path"] / "config.json").exists() and (results_dir(key) / "happiness_vector.npz").exists() \
        and bool(steering_settings(key))


def load_directions(key: str, hap) -> dict:
    """Selectable steering directions for a model: the happiness one, plus anything in results/<key>/directions/."""
    out = {"happiness": {"id": "happiness", "label": "Happiness", "description": "The original happiness / wellbeing direction.",
                         "vector": hap["vector"].astype(np.float32), "layer": int(hap["layer"]), "n_pairs": None}}
    d = results_dir(key) / "directions"
    if (d / "index.json").exists():
        for m in json.load(open(d / "index.json"))["directions"]:
            f = np.load(d / f"{m['id']}.npz")
            out[m["id"]] = {**m, "vector": f["vector"].astype(np.float32), "layer": int(f["layer"])}
    return out


def default_adapter(key: str):
    return MODELS[key].get("default_adapter")


def activate(key: str, adapter: str | None = None):
    """Runs on the worker thread. Frees the current model first so the two never coexist in memory.
    `adapter` is an id from MODELS[key]["adapters"] (None = that model's default)."""
    adapter = adapter or default_adapter(key)
    READY.clear()
    ACTIVE.update(model=None, tok=None)
    gc.collect(); mx.clear_cache()
    print(f"Loading {key} (adapter: {adapter}) ...", flush=True)
    model, tok = load_model(key, adapter_id=adapter)
    res = results_dir(key)
    vf = np.load(res / "happiness_vector.npz")
    ACTIVE.update(key=key, adapter=adapter, model=model, tok=tok, vector=vf["vector"].astype(np.float32), layer_ext=int(vf["layer"]),
                  directions=load_directions(key, vf),
                  n_layers=n_layers(model),
                  ratios={r["layer"]: r["vector_over_residual"] for r in json.load(open(res / "ratio_diagnostic.json"))})
    print(f"{key} ready.", flush=True)
    READY.set()


class Turn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str | None = None         # which model the page believes is active; None = whatever is active
    adapter: str | None = None       # likewise for the fine-tune (LoRA); None = whatever is active
    original: list[Turn] = []        # the conversation so far, as seen by the unsteered model
    steered: list[Turn] = []         # the conversation so far, as seen by the steered model
    alpha: float = 0.0
    direction: str | None = None     # id from /api/directions to steer with; None = the happiness direction
    layer: int = 12
    system: str = ""
    temperature: float = 0.7
    max_tokens: int = 220
    seed: int = 0
    effort: str = "low"              # reasoning effort, thinking models only
    columns: list[str] = ["original", "steered"]


class SwitchRequest(BaseModel):
    model: str
    adapter: str | None = None       # None = that model's default adapter


class CompareCol(BaseModel):
    id: str
    direction: str | None = None     # None = the unsteered model
    alpha: float = 0.0
    history: list[Turn] = []         # this column's own conversation so far, ending with the new user message


class CompareRequest(BaseModel):
    model: str | None = None
    adapter: str | None = None
    columns: list[CompareCol]
    layer: int = 12
    system: str = ""
    temperature: float = 0.7
    max_tokens: int = 220
    seed: int = 0
    effort: str = "low"


def sse(obj) -> str:
    return f"data: {json.dumps(obj)}\n\n"


def _partial_tag_len(buf: str) -> int:
    for k in range(min(len(_TAG) - 1, len(buf)), 0, -1):
        if buf.endswith(_TAG[:k]):
            return k
    return 0


def run_column(col: str, turns: list[Turn], req: ChatRequest, cancel: threading.Event):
    d = ACTIVE["directions"].get(req.direction) if req.direction else None
    yield from generate_column(col, turns, d["vector"] if d else ACTIVE["vector"], req.alpha if col == "steered" else 0.0, req, cancel)


def generate_column(col: str, turns: list[Turn], vector, alpha: float, req, cancel: threading.Event):
    """Stream one reply. `req` only needs layer/system/effort/temperature/max_tokens/seed."""
    key, model, tok = ACTIVE["key"], ACTIVE["model"], ACTIVE["tok"]
    thinking = MODELS[key]["thinking"]
    set_steering(model, vector, req.layer, alpha if vector is not None else 0.0)
    mx.random.seed(req.seed)                 # same seed for every column: differences come from steering
    sampler = make_sampler(temp=req.temperature, top_p=0.95)
    ids = encode_prompt(tok, apply_template(tok, key, [t.model_dump() for t in turns], req.system, req.effort))
    t0, n, buf, sent_think, sent_ans = time.time(), 0, "", 0, 0
    for resp in stream_generate(model, tok, prompt=ids, max_tokens=req.max_tokens, sampler=sampler):
        if cancel.is_set():
            break
        n += 1
        if not thinking:
            yield {"col": col, "delta": resp.text, "kind": "answer"}
            continue
        buf += resp.text
        th, ans, started = split_thinking(buf)
        if not started:
            th = th[:len(th) - _partial_tag_len(th)]
        if len(th) > sent_think:
            yield {"col": col, "delta": th[sent_think:], "kind": "think"}; sent_think = len(th)
        if started and len(ans) > sent_ans:
            yield {"col": col, "delta": ans[sent_ans:], "kind": "answer"}; sent_ans = len(ans)
    yield {"col": col, "done": True, "tokens": n, "seconds": round(time.time() - t0, 1),
           "truncated_in_thinking": bool(thinking and sent_ans == 0 and n >= req.max_tokens)}


def worker():
    activate(DEFAULT_MODEL)
    while True:
        job = JOBS.get()
        if job[0] == "switch":
            _, key, adapter, done = job
            adapter = adapter or default_adapter(key)
            try:
                if key != ACTIVE["key"] or adapter != ACTIVE["adapter"]:
                    activate(key, adapter)
                done.put(None)
            except Exception as e:
                try:
                    activate(ACTIVE["key"] or DEFAULT_MODEL, ACTIVE["adapter"])       # fall back to something that works
                except Exception:
                    pass
                done.put(f"{type(e).__name__}: {e}")
            continue
        kind, req, out, cancel = job
        try:
            if kind == "compare":
                for c in req.columns:
                    vec = ACTIVE["directions"][c.direction]["vector"] if c.direction else None
                    for ev in generate_column(c.id, c.history, vec, c.alpha, req, cancel):
                        out.put(ev)
                    if cancel.is_set():
                        break
            else:
                for col in req.columns:
                    turns = req.original if col == "original" else req.steered
                    for ev in run_column(col, turns, req, cancel):
                        out.put(ev)
                    if cancel.is_set():
                        break
        except Exception as e:                # surface errors to the browser instead of hanging
            out.put({"error": f"{type(e).__name__}: {e}"})
        finally:
            set_steering(ACTIVE["model"], None, req.layer, 0.0)
            out.put(None)


if not os.environ.get("NO_MODEL"):          # NO_MODEL=1: serve pages only (used for testing)
    threading.Thread(target=worker, daemon=True).start()


@app.post("/api/chat")
def chat(req: ChatRequest):
    if not READY.is_set() or (req.model is not None and req.model != ACTIVE["key"]) \
            or (req.adapter is not None and req.adapter != ACTIVE["adapter"]):
        return JSONResponse({"error": "model is loading or a different model / fine-tune is active"}, status_code=409)
    if req.direction and req.direction not in ACTIVE["directions"]:
        return JSONResponse({"error": f"unknown direction for this model: {req.direction}"}, status_code=400)
    out: "queue.Queue" = queue.Queue()
    cancel = threading.Event()
    JOBS.put(("chat", req, out, cancel))

    def gen():
        try:
            while (ev := out.get()) is not None:
                yield sse(ev)
        finally:
            cancel.set()                      # browser pressed Stop / closed the tab
        yield sse({"all_done": True})
    return StreamingResponse(gen(), media_type="text/event-stream")



@app.get("/api/directions")
def directions():
    """Directions available for the active model, with the numbers measured when they were built."""
    if not READY.is_set():
        return {"loading": True}
    d = results_dir(ACTIVE["key"]) / "directions" / "index.json"
    idx = json.load(open(d)) if d.exists() else {}
    ml = idx.get("metrics_by_layer", {})
    return {"loading": False, "model": ACTIVE["key"], "layer_ext": ACTIVE["layer_ext"], "built_at": idx.get("built_at"),
            "recipe": idx.get("recipe"), "metrics_at_extraction_layer": ml.get(str(ACTIVE["layer_ext"])),
            "directions": [{k: v for k, v in m.items() if k != "vector"} for m in ACTIVE["directions"].values()]}


@app.post("/api/model")
def switch(req: SwitchRequest):
    if req.model not in MODELS or not available(req.model):
        return JSONResponse({"error": f"{req.model} is not available (weights or results missing)"}, status_code=400)
    if req.adapter and not adapter_path(req.model, req.adapter):
        return JSONResponse({"error": f"fine-tune '{req.adapter}' is not available for {req.model} (unknown or not on disk)"}, status_code=400)
    done: "queue.Queue" = queue.Queue()
    JOBS.put(("switch", req.model, req.adapter, done))
    err = done.get()                           # blocks until loaded (about 10-20 s)
    if err:
        return JSONResponse({"error": err}, status_code=500)
    return config()


@app.get("/api/config")
def config():
    models = [{"key": k, "label": v["label"], "short": v["short"], "available": available(k), "thinking": v["thinking"]}
              for k, v in MODELS.items()]
    if not READY.is_set():
        return {"loading": True, "models": models}
    key, st = ACTIVE["key"], steering_settings(ACTIVE["key"])
    return {
        "loading": False, "models": models, "model": key, "label": MODELS[key]["label"], "thinking": MODELS[key]["thinking"],
        "adapter": ACTIVE["adapter"] is not None, "adapter_id": ACTIVE["adapter"],
        "adapter_label": next((a["label"] for a in MODELS[key]["adapters"] if a["id"] == ACTIVE["adapter"]), None),
        "adapters": [{"id": a["id"], "label": a["label"], "available": adapter_path(key, a["id"]) is not None} for a in MODELS[key]["adapters"]],
        "n_layers": ACTIVE["n_layers"], "extraction_layer": ACTIVE["layer_ext"],
        "steer_layer": st["steer_layer"], "default_alpha": st["default_alpha"], "alpha_range": st["alpha_range"],
        "alpha_good_range": st.get("alpha_good_range"), "max_tokens": st.get("max_tokens", 220),
        "vector_norm": float(np.linalg.norm(ACTIVE["vector"])),
        "ratio_by_layer": {str(k): v for k, v in ACTIVE["ratios"].items()},
    }


def page(name: str) -> HTMLResponse:
    """Serve an html page with ?v=<file mtime> on its local css/js, so edits always reach the browser."""
    text = (ROOT / "app" / "static" / name).read_text()
    return HTMLResponse(re.sub(r'/static/[\w./-]+\.(?:js|css)"', lambda m: _stamp(m.group(0)), text))


def _stamp(token: str) -> str:
    path = token[:-1]                                   # strip the closing quote
    f = ROOT / "app" / path.lstrip("/")
    return f'{path}?v={int(f.stat().st_mtime)}"' if f.exists() else token


@app.get("/")
def index():
    return page("index.html")


@app.get("/methodology")
def methodology():
    return page("methodology.html")










app.mount("/static", StaticFiles(directory=ROOT / "app" / "static"), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8000)), log_level="warning")
