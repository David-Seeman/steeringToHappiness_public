"""Thin MLX helpers: load a model, read the residual stream, and steer it.

"Residual stream at layer l" follows the paper: the output of decoder block l (block 0 first).
Steering adds `alpha * vector` to that same stream at one chosen block, at every token position
(prompt tokens and generated tokens alike), exactly like a forward hook that returns
`output + alpha * v`.
"""
from __future__ import annotations

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask


def load_model(path: str, adapter_path: str | None = None):
    model, tokenizer = load(path, adapter_path=adapter_path)
    model.eval()
    return model, tokenizer


def _blocks(model):
    inner = model.model
    return [b.layer if isinstance(b, SteerBlock) else b for b in inner.layers]


def n_layers(model) -> int:
    return len(model.model.layers)


def residual_stream(model, token_ids: list[int]):
    """Run the transformer blocks by hand and return (n_layers, T, d) float32 numpy.

    Steering (if installed) is bypassed here on purpose: extraction always reads the clean model.
    """
    inner = model.model
    x = mx.array(token_ids)[None]
    h = inner.embed_tokens(x)
    mask = create_attention_mask(h, None)
    outs = []
    for blk in _blocks(model):
        h = blk(h, mask, None)
        outs.append(h[0].astype(mx.float32))
    stacked = mx.stack(outs)  # (L, T, d)
    mx.eval(stacked)
    return np.array(stacked)


# --- steering -----------------------------------------------------------------------------

class SteerState:
    """Mutable steering config shared with the installed wrapper block."""

    def __init__(self):
        self.vector: mx.array | None = None  # raw (un-normalised) direction, shape (d,)
        self.alpha: float = 0.0
        self.layer: int = 0


class SteerBlock(nn.Module):
    """Wraps one decoder block; adds alpha * vector to its output when alpha != 0."""

    def __init__(self, block, state: SteerState, index: int):
        super().__init__()
        self.block = block
        self._state = state
        self._index = index

    def __call__(self, x, *args, **kwargs):
        out = self.block(x, *args, **kwargs)
        s = self._state
        if self._index == s.layer and s.alpha != 0.0 and s.vector is not None:
            out = out + (s.alpha * s.vector).astype(out.dtype)
        return out


def install_steering(model) -> SteerState:
    """Wrap every block once (cheap). Returns the shared state to mutate."""
    inner = model.model
    if getattr(model, "_steer_state", None) is not None:
        return model._steer_state
    state = SteerState()
    inner.layers = [SteerBlock(b, state, i) for i, b in enumerate(inner.layers)]
    model._steer_state = state
    return state


def set_steering(model, vector: np.ndarray | None, layer: int, alpha: float):
    state = install_steering(model)
    state.vector = None if vector is None else mx.array(vector.astype(np.float32))
    state.layer = int(layer)
    state.alpha = float(alpha)
    return state
