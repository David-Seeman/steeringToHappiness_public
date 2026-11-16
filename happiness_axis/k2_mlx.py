"""MLX implementation of K2-Horizon-7B (dense variant), written from the architecture description in
IFM's modeling_k2_horizon.py (no third-party code is executed).

For the dense 7B config the model is Llama-like: pre-norm decoder, GQA (32 q heads / 8 kv heads, head_dim 128),
SwiGLU MLP, split-halves RoPE (theta 1e7, rotating all 128 dims), no biases, untied lm_head, one difference:
RMSNorm is computed *per group* of the hidden dimension (layernorm_num_groups = 4 chunks of 1024) with one
shared weight vector. QK-norm, attention gating, MoE and sliding window are all off for this checkpoint.

Same call signatures as mlx-lm models, so `mlx_lm.stream_generate` and the residual-stream helpers work on it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask, scaled_dot_product_attention


@dataclass
class K2Args:
    hidden_size: int = 4096
    num_hidden_layers: int = 36
    intermediate_size: int = 12288
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 250624
    rms_norm_eps: float = 1e-6
    layernorm_num_groups: int = 4
    rope_theta: float = 1e7

    @classmethod
    def from_config(cls, path: Path) -> "K2Args":
        c = json.load(open(path))
        assert c["model_type"] == "k2_horizon"
        # Guard against silently loading a variant this implementation does not cover.
        assert c.get("num_experts", 0) == 0 and c.get("mova_num_experts", 0) == 0, "MoE variants unsupported"
        assert not c.get("query_key_norm") and c.get("attention_gate_func") is None and not c.get("attention_bias")
        assert c.get("sliding_window") is None and c["rope_head_dim"] == c["head_dim"]
        return cls(hidden_size=c["hidden_size"], num_hidden_layers=c["num_hidden_layers"],
                   intermediate_size=c["intermediate_size"], num_attention_heads=c["num_attention_heads"],
                   num_key_value_heads=c["num_key_value_heads"], head_dim=c["head_dim"], vocab_size=c["vocab_size"],
                   rms_norm_eps=c["rms_norm_eps"], layernorm_num_groups=c["layernorm_num_groups"],
                   rope_theta=c["rope_parameters"]["rope_theta"])


class GroupedRMSNorm(nn.Module):
    def __init__(self, dims: int, groups: int, eps: float):
        super().__init__()
        assert dims % groups == 0
        self.weight = mx.ones((dims,))
        self.groups, self.eps = groups, eps

    def __call__(self, x):
        dt, shape = x.dtype, x.shape
        xf = x.astype(mx.float32).reshape(*shape[:-1], self.groups, shape[-1] // self.groups)
        xf = xf * mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + self.eps)
        return (self.weight.astype(mx.float32) * xf.reshape(shape)).astype(dt)


class Attention(nn.Module):
    def __init__(self, a: K2Args):
        super().__init__()
        self.n_heads, self.n_kv = a.num_attention_heads, a.num_key_value_heads
        self.scale = a.head_dim ** -0.5
        self.q_proj = nn.Linear(a.hidden_size, self.n_heads * a.head_dim, bias=False)
        self.k_proj = nn.Linear(a.hidden_size, self.n_kv * a.head_dim, bias=False)
        self.v_proj = nn.Linear(a.hidden_size, self.n_kv * a.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * a.head_dim, a.hidden_size, bias=False)
        self.rope = nn.RoPE(a.head_dim, traditional=False, base=a.rope_theta)

    def __call__(self, x, mask=None, cache=None):
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.n_kv, -1).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.n_kv, -1).transpose(0, 2, 1, 3)
        if cache is not None:
            q, k = self.rope(q, offset=cache.offset), self.rope(k, offset=cache.offset)
            k, v = cache.update_and_fetch(k, v)
        else:
            q, k = self.rope(q), self.rope(k)
        out = scaled_dot_product_attention(q, k, v, cache=cache, scale=self.scale, mask=mask)
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(B, L, -1))


class MLP(nn.Module):
    def __init__(self, a: K2Args):
        super().__init__()
        self.gate_proj = nn.Linear(a.hidden_size, a.intermediate_size, bias=False)
        self.up_proj = nn.Linear(a.hidden_size, a.intermediate_size, bias=False)
        self.down_proj = nn.Linear(a.intermediate_size, a.hidden_size, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, a: K2Args):
        super().__init__()
        self.self_attn = Attention(a)
        self.mlp = MLP(a)
        self.input_layernorm = GroupedRMSNorm(a.hidden_size, a.layernorm_num_groups, a.rms_norm_eps)
        self.post_attention_layernorm = GroupedRMSNorm(a.hidden_size, a.layernorm_num_groups, a.rms_norm_eps)

    def __call__(self, x, mask=None, cache=None):
        h = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class Backbone(nn.Module):
    def __init__(self, a: K2Args):
        super().__init__()
        self.embed_tokens = nn.Embedding(a.vocab_size, a.hidden_size)
        self.layers = [Block(a) for _ in range(a.num_hidden_layers)]
        self.norm = GroupedRMSNorm(a.hidden_size, a.layernorm_num_groups, a.rms_norm_eps)

    def __call__(self, inputs, cache=None):
        h = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        mask = create_attention_mask(h, cache[0])
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)
        return self.norm(h)


class K2Model(nn.Module):
    def __init__(self, a: K2Args):
        super().__init__()
        self.args = a
        self.model = Backbone(a)
        self.lm_head = nn.Linear(a.hidden_size, a.vocab_size, bias=False)

    def __call__(self, inputs, cache=None, input_embeddings=None):
        return self.lm_head(self.model(inputs, cache))

    @property
    def layers(self):
        return self.model.layers


# --- mixed-precision quantisation plan (shared by the converter and the loader) ------------------

def quant_bits(path: str, mlp_bits: int) -> int | None:
    """Bits for a module path like 'model.layers.3.mlp.up_proj'; None = keep in 16-bit."""
    if path.endswith("norm") or "layernorm" in path:
        return None
    if ".mlp." in path:
        return mlp_bits
    return 8   # attention, embeddings, lm_head


def load_k2(path: str | Path):
    """Load a converted (quantised) K2-Horizon checkpoint -> (model, tokenizer)."""
    from mlx_lm.tokenizer_utils import load as load_tokenizer
    path = Path(path)
    cfg = json.load(open(path / "config.json"))
    a = K2Args.from_config(path / "config.json")
    model = K2Model(a)
    q = cfg["mlx_quantization"]
    nn.quantize(model, group_size=q["group_size"],
                class_predicate=lambda p, m: ({"bits": quant_bits(p, q["mlp_bits"]), "group_size": q["group_size"]}
                                              if isinstance(m, (nn.Linear, nn.Embedding)) and quant_bits(p, q["mlp_bits"])
                                              else False))
    weights = {}
    for f in sorted(path.glob("*.safetensors")):
        weights.update(mx.load(str(f)))
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    model.eval()
    tok = load_tokenizer(path)
    tok.add_eos_token("<|ifm|im_end|>")
    return model, tok
