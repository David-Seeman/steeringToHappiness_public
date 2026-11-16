"""Prompt building and thinking-trace handling shared by the pipeline scripts and the chat server."""
from __future__ import annotations

import re

from .models import MODELS

# K2 Horizon opens its reply with <ifm|think...>, reasons, closes with </ifm|think...>, then answers.
THINK_CLOSE = re.compile(r"</ifm\|think(?:_fast|_faster)?>")
THINK_OPEN = re.compile(r"^\s*<ifm\|think(?:_fast|_faster)?>\s*")   # the model sometimes repeats the opening tag


def apply_template(tok, key: str, turns: list[dict], system: str = "", effort: str = "low") -> str:
    """Chat-template text for the model `key`. `turns` = [{role, content}, ...] (assistant content = answer only)."""
    spec = MODELS[key]
    msgs = [{"role": "system", "content": system}] if system.strip() else []
    for t in turns:
        m = {"role": t["role"], "content": t["content"]}
        if spec["thinking"] and t["role"] == "assistant":
            m["reasoning"] = ""            # the template insists on a thinking field for past assistant turns
        msgs.append(m)
    kw = {"reasoning_effort": effort} if spec["thinking"] else {}
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, **kw)


def encode_prompt(tok, text: str) -> list[int]:
    """The chat template already contains BOS where the model wants it, so don't add another."""
    return tok.encode(text, add_special_tokens=False)


def split_thinking(text: str) -> tuple[str, str, bool]:
    """-> (thinking, answer, answer_started). Prompt already ended inside the think tag."""
    m = THINK_CLOSE.search(text)
    th, ans, started = (text, "", False) if not m else (text[:m.start()], text[m.end():].lstrip("\n"), True)
    return THINK_OPEN.sub("", th, count=1), ans, started
