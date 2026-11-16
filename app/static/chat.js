"use strict";
const $ = (id) => document.getElementById(id);
const state = { original: [], steered: [], cfg: null, busy: false, controller: null, dirs: [] };
const dirLabel = () => { const d = state.dirs.find((x) => x.id === $("direction").value); return d ? d.label : "Happiness"; };

// ---- per-model settings, remembered in the browser --------------------------------------------
function key() { return state.cfg ? state.cfg.model : "x"; }
function save() { try { localStorage.setItem("sth.settings." + key(), JSON.stringify(settings())); localStorage.setItem("sth.showOriginal", $("showOriginal").checked ? "1" : "0"); } catch (e) {} }
function load() { try { return JSON.parse(localStorage.getItem("sth.settings." + key()) || "{}"); } catch (e) { return {}; } }
function settings() {
  return { alpha: +$("alphaNum").value, layer: +$("layer").value, temp: +$("temp").value, maxTokens: +$("maxTokens").value,
           seed: +$("seed").value, system: $("system").value, effort: $("effort").value, showOriginal: $("showOriginal").checked,
           direction: $("direction").value || "happiness" };
}

function renderEmpty(col) {
  const log = $("log-" + col);
  if (!log.children.length) {
    const d = document.createElement("div");
    d.className = "empty";
    d.textContent = col === "original"
      ? "The unmodified model. Same weights, nothing injected."
      : `The same model with the "${dirLabel()}" direction added to its residual stream.`;
    log.appendChild(d);
  }
}

// Auto-scroll: each chat log follows its newest text until you scroll UP to re-read; scrolling back to the bottom resumes it.
// Following is switched off only by an actual upward scroll (scrollTop decreasing), never by new text arriving, so a burst of
// text, a markdown re-render or the footer line can't make it lose the bottom. A MutationObserver does the scrolling, so every
// kind of growth is covered. In the stacked (narrow) layout the page itself scrolls too.
const follow = { original: true, steered: true, page: true };
const lastTop = { original: 0, steered: 0, page: 0 };
function isPinned(el, slack = 40) { return el.scrollHeight - el.scrollTop - el.clientHeight < slack; }
function track(name, el, slack) {
  const t = el.scrollTop;
  if (t < lastTop[name] - 2) follow[name] = false;          // moved up: the user is reading earlier text
  else if (isPinned(el, slack)) follow[name] = true;        // back at the bottom: resume
  lastTop[name] = t;
}
function toBottom(log, col) { log.scrollTop = log.scrollHeight; lastTop[col] = log.scrollTop; }
function pageBottom() {
  const el = document.scrollingElement;
  if (follow.page && el.scrollHeight > el.clientHeight + 1) { el.scrollTop = el.scrollHeight; lastTop.page = el.scrollTop; }
}
function watchScroll(col) {
  const log = $("log-" + col);
  log.addEventListener("scroll", () => track(col, log, 40), { passive: true });
  const go = () => { if (follow[col]) toBottom(log, col); pageBottom(); };
  new MutationObserver(() => { requestAnimationFrame(go); setTimeout(go, 60); })   // timer too: rAF pauses in background tabs
    .observe(log, { childList: true, subtree: true, characterData: true });
}
window.addEventListener("scroll", () => track("page", document.scrollingElement, 60), { passive: true });
watchScroll("original"); watchScroll("steered");

function addMsg(col, role, text) {
  const log = $("log-" + col);
  const e = log.querySelector(".empty"); if (e) e.remove();
  const m = document.createElement("div");
  m.className = "msg " + role;
  if (role === "assistant") {
    m.innerHTML = '<details class="think" open hidden><summary>Thinking…</summary><div class="thinktext"></div></details><div class="answer"></div>';
  } else {
    m.textContent = text;
  }
  if (role === "user") { follow[col] = true; follow.page = true; }   // sending a message always jumps to the bottom
  log.appendChild(m);
  toBottom(log, col); pageBottom();
  return m;
}

function updateHint() {
  const a = +$("alphaNum").value, l = +$("layer").value;
  const r = state.cfg && state.cfg.ratio_by_layer[String(l)];
  $("steeredTag").textContent = dirLabel() + " · α = " + a.toFixed(1);
  $("alphaHint").textContent = a === 0 ? "no steering" : (r ? `≈ ${Math.round(Math.abs(a) * r * 100)}% of layer-${l} residual norm` : "");
}
function syncAlpha(v) { $("alpha").value = v; $("alphaNum").value = v; updateHint(); save(); }

// ---- model + config ---------------------------------------------------------------------------
function applyConfig(cfg) {
  state.cfg = cfg;
  const s = Object.assign({ alpha: cfg.default_alpha, layer: cfg.steer_layer, temp: 0.7, maxTokens: cfg.max_tokens, seed: 0, system: "", effort: "low" }, load());
  $("alpha").min = cfg.alpha_range[0]; $("alpha").max = cfg.alpha_range[1];
  $("layer").max = cfg.n_layers - 1;
  $("layer").value = s.layer; $("temp").value = s.temp; $("maxTokens").value = s.maxTokens; $("seed").value = s.seed;
  $("system").value = s.system; $("effort").value = s.effort;
  $("effortWrap").hidden = !cfg.thinking;
  syncAlpha(s.alpha);
  $("modelBadge").textContent = cfg.label + (cfg.adapter_label ? " + " + cfg.adapter_label : "") + " · vector from layer " + cfg.extraction_layer;

  // the fine-tune (LoRA) selector: only for models that have more than one
  const ad = $("adapter"), many = cfg.adapters && cfg.adapters.length > 1;
  $("adapterWrap").hidden = !many;
  ad.innerHTML = "";
  if (many) for (const a of cfg.adapters) { const o = new Option(a.label + (a.available ? "" : " (not installed)"), a.id); o.disabled = !a.available; ad.appendChild(o); }
  if (many) ad.value = cfg.adapter_id;

  // the model-switch checkbox: checked = the non-default model is active
  const alt = cfg.models.find((m) => m.key !== "qwen");
  if (alt) {
    $("useAlt").checked = cfg.model === alt.key;
    $("useAlt").disabled = !alt.available;
    $("useAltLabel").textContent = `Use ${alt.short} instead of ${cfg.models.find((m) => m.key === "qwen").short}`;
    $("useAltNote").textContent = alt.available
      ? (alt.thinking ? "Reasoning model, no fine-tuning. Switching reloads the model (~15 s) and starts a new conversation." : "Switching reloads the model and starts a new conversation.")
      : "Not installed yet: its weights or steering results are missing.";
  }
}

// the steering directions this model offers (the happiness direction, plus any extra ones in results/<model>/directions/), from /api/directions
async function loadDirections(selected) {
  try {
    const info = await (await fetch("/api/directions")).json();
    state.dirs = info.directions || [];
  } catch (e) { state.dirs = []; }
  if (!state.dirs.length) state.dirs = [{ id: "happiness", label: "Happiness", description: "" }];
  const sel = $("direction"); sel.innerHTML = "";
  for (const d of state.dirs) { const o = new Option(d.label, d.id); o.title = d.description || ""; sel.appendChild(o); }
  sel.value = state.dirs.some((d) => d.id === selected) ? selected : "happiness";
  updateHint();
  const e = document.querySelector("#log-steered .empty"); if (e) e.textContent = `The same model with the "${dirLabel()}" direction added to its residual stream.`;
}

async function loadConfig() {
  for (;;) {
    const cfg = await (await fetch("/api/config")).json();
    if (!cfg.loading) return cfg;
    $("modelBadge").textContent = "loading model…"; await new Promise((r) => setTimeout(r, 1500));
  }
}

async function switchModel(toKey, adapter) {
  if (state.busy) { applyConfig(state.cfg); return; }
  const m = state.cfg.models.find((x) => x.key === toKey), a = adapter && state.cfg.adapters.find((x) => x.id === adapter);
  $("overlayText").textContent = `Loading ${a ? a.label : m.short}… (the first load can take 15–30 s)`;
  $("overlay").hidden = false;
  try {
    const r = await fetch("/api/model", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model: toKey, adapter: adapter || null }) });
    const body = await r.json();
    if (!r.ok) throw new Error(body.error || r.statusText);
    state.original = []; state.steered = [];
    $("log-original").innerHTML = ""; $("log-steered").innerHTML = "";
    applyConfig(body); await loadDirections(load().direction); renderEmpty("original"); renderEmpty("steered");
  } catch (e) {
    alert("Could not switch model: " + e.message);
    applyConfig(await loadConfig());
  } finally { $("overlay").hidden = true; }
}

async function init() {
  const cfg = await loadConfig();
  state.cfg = cfg;
  try { $("showOriginal").checked = localStorage.getItem("sth.showOriginal") !== "0"; } catch (e) {}
  applyConfig(cfg); applyColumns();
  await loadDirections(load().direction);
  renderEmpty("original"); renderEmpty("steered");
}

function applyColumns() { $("cols").classList.toggle("single", !$("showOriginal").checked); save(); }

// ---- chatting ---------------------------------------------------------------------------------
async function send(text) {
  const s = settings();
  const cols = s.showOriginal ? ["original", "steered"] : ["steered"];
  if (s.showOriginal) state.original.push({ role: "user", content: text });
  state.steered.push({ role: "user", content: text });
  if (s.showOriginal) addMsg("original", "user", text);
  addMsg("steered", "user", text);

  state.busy = true; $("send").disabled = true; $("stop").hidden = false; $("useAlt").disabled = true; $("adapter").disabled = true;
  state.controller = new AbortController();
  const bubbles = {}, bufs = { original: "", steered: "" }, thinks = { original: "", steered: "" };
  for (const c of cols) { bubbles[c] = addMsg(c, "assistant", ""); bubbles[c].classList.add("cursor"); }
  const ans = (c) => bubbles[c].querySelector(".answer");

  try {
    const resp = await fetch("/api/chat", {
      method: "POST", headers: { "Content-Type": "application/json" }, signal: state.controller.signal,
      body: JSON.stringify({ model: state.cfg.model, adapter: state.cfg.adapter_id, original: state.original, steered: state.steered, alpha: s.alpha, direction: s.direction, layer: s.layer,
                             system: s.system, temperature: s.temp, max_tokens: s.maxTokens, seed: s.seed, effort: s.effort, columns: cols }),
    });
    if (!resp.ok) throw new Error((await resp.json()).error || resp.statusText);
    const reader = resp.body.getReader(), dec = new TextDecoder();
    let buf = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n\n")) >= 0) {
        const line = buf.slice(0, i).trim(); buf = buf.slice(i + 2);
        if (!line.startsWith("data:")) continue;
        const ev = JSON.parse(line.slice(5));
        if (ev.error) throw new Error(ev.error);
        if (ev.delta !== undefined) {
          if (ev.kind === "think") {
            thinks[ev.col] += ev.delta;
            const d = bubbles[ev.col].querySelector("details.think"); d.hidden = false;
            d.querySelector(".thinktext").textContent = thinks[ev.col];
          } else { bufs[ev.col] += ev.delta; ans(ev.col).innerHTML = renderMarkdown(bufs[ev.col]); }
        } else if (ev.done) {
          const b = bubbles[ev.col];
          b.classList.remove("cursor");
          const d = b.querySelector("details.think");
          if (!d.hidden) d.querySelector("summary").textContent = `Thinking (${thinks[ev.col].length} characters)`;
          if (ev.truncated_in_thinking) {
            const n = document.createElement("span"); n.className = "notice";
            n.textContent = "Ran out of tokens while thinking. Raise Max tokens or lower the thinking effort in Settings."; b.appendChild(n);
          }
          const meta = document.createElement("span"); meta.className = "meta";
          meta.textContent = `${ev.tokens} tokens · ${ev.seconds}s` + (ev.col === "steered" ? ` · ${dirLabel()} α=${s.alpha}` : ""); b.appendChild(meta);
          state[ev.col].push({ role: "assistant", content: bufs[ev.col] });
        }
      }
    }
  } catch (e) {
    if (e.name !== "AbortError") {
      for (const c of cols) { const n = document.createElement("span"); n.className = "notice"; n.textContent = "[error: " + e.message + "]"; bubbles[c].appendChild(n); }
    }
    if (e.name === "AbortError") {
      for (const c of cols) if (bufs[c] && !(state[c].length && state[c][state[c].length - 1].role === "assistant")) {
        state[c].push({ role: "assistant", content: bufs[c] }); }
    }
  } finally {
    for (const c of cols) bubbles[c].classList.remove("cursor");
    // keep both histories aligned (a stopped/failed column may have no reply)
    for (const c of ["original", "steered"]) {
      const h = state[c]; if (h.length && h[h.length - 1].role === "user") h.push({ role: "assistant", content: "…" });
    }
    state.busy = false; $("send").disabled = false; $("stop").hidden = true;
    const alt = state.cfg.models.find((m) => m.key !== "qwen"); $("useAlt").disabled = !(alt && alt.available); $("adapter").disabled = false;
  }
}

$("form").addEventListener("submit", (e) => {
  e.preventDefault();
  const t = $("input").value.trim(); if (!t || state.busy) return;
  $("input").value = ""; $("input").style.height = "auto"; send(t);
});
$("input").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("form").requestSubmit(); } });
$("input").addEventListener("input", (e) => { e.target.style.height = "auto"; e.target.style.height = Math.min(e.target.scrollHeight, 140) + "px"; });
$("alpha").addEventListener("input", (e) => syncAlpha(e.target.value));
$("alphaNum").addEventListener("input", (e) => syncAlpha(e.target.value));
for (const id of ["layer", "temp", "maxTokens", "seed", "system", "effort"]) $(id).addEventListener("change", () => { updateHint(); save(); });
$("showOriginal").addEventListener("change", applyColumns);
$("direction").addEventListener("change", () => {
  updateHint(); save();
  const e = document.querySelector("#log-steered .empty"); if (e) e.textContent = `The same model with the "${dirLabel()}" direction added to its residual stream.`;
});
$("useAlt").addEventListener("change", () => {
  const alt = state.cfg.models.find((m) => m.key !== "qwen");
  switchModel($("useAlt").checked ? alt.key : "qwen");
});
$("adapter").addEventListener("change", () => switchModel(state.cfg.model, $("adapter").value));
$("advToggle").addEventListener("click", () => { const a = $("adv"); a.hidden = !a.hidden; $("advToggle").textContent = a.hidden ? "Settings ▾" : "Settings ▴"; });
$("stop").addEventListener("click", () => state.controller && state.controller.abort());
$("resetBtn").addEventListener("click", () => {
  if (state.busy) return;
  state.original = []; state.steered = [];
  $("log-original").innerHTML = ""; $("log-steered").innerHTML = "";
  renderEmpty("original"); renderEmpty("steered");
});
init();
