# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""JiT (Just image Transformer) backbone adapted for the DDBM calling convention.

JiT (LTH14, 2025) extends the DiT family with bottleneck patch embedding, RMSNorm,
SwiGLU FFN, and 2-D rotary position embeddings.  This module wraps the core
architecture for image-to-image diffusion bridges: class conditioning and
in-context tokens are omitted; the source image is concatenated along channels
when ``condition_mode='concat'``, matching :class:`SiTBackbone`.

Forward signature (same as :class:`DDBMUNet` / :class:`SiTBackbone`)::

    output = model(x, timestep, xT=source)

Upstream references:
* https://github.com/LTH14/JiT
* SiT: https://github.com/willisma/SiT
* Lightning-DiT (RoPE): https://github.com/hustvl/LightningDiT
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from einops import rearrange, repeat


# ---------------------------------------------------------------------------
# RoPE helpers (from JiT util/model_util.py; buffers are device-agnostic)
# ---------------------------------------------------------------------------


def _broadcat(tensors: list[torch.Tensor], dim: int = -1) -> torch.Tensor:
    num_tensors = len(tensors)
    shape_lens = set(len(t.shape) for t in tensors)
    assert len(shape_lens) == 1, "tensors must all have the same number of dimensions"
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*[list(t.shape) for t in tensors]))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all(len(set(t[1])) <= 2 for t in expandable_dims), "invalid broadcastable concat"
    max_dims = [(t[0], max(t[1])) for t in expandable_dims]
    expanded_dims = [(t[0], (t[1],) * num_tensors) for t in max_dims]
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*[t[1] for t in expanded_dims]))
    tensors = [t[0].expand(*t[1]) for t in zip(tensors, expandable_shapes)]
    return torch.cat(tensors, dim=dim)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


class _VisionRotaryEmbeddingFast(nn.Module):
    """2-D RoPE for patch tokens (JiT / Lightning-DiT style)."""

    def __init__(
        self,
        dim: int,
        pt_seq_len: int = 16,
        ft_seq_len: Optional[int] = None,
        theta: float = 10000.0,
    ) -> None:
        super().__init__()
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32)[: (dim // 2)] / dim))
        if ft_seq_len is None:
            ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len, dtype=torch.float32) / ft_seq_len * pt_seq_len
        freqs = torch.einsum("..., f -> ... f", t, freqs)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)
        freqs = _broadcat((freqs[:, None, :], freqs[None, :, :]), dim=-1)
        cos = freqs.cos().reshape(-1, freqs.shape[-1])
        sin = freqs.sin().reshape(-1, freqs.shape[-1])
        self.register_buffer("freqs_cos", cos, persistent=False)
        self.register_buffer("freqs_sin", sin, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return t * self.freqs_cos + _rotate_half(t) * self.freqs_sin


class _RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


# ---------------------------------------------------------------------------
# Sinusoidal position (patch grid)
# ---------------------------------------------------------------------------


def _get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape(2, 1, grid_size, grid_size)
    return _get_2d_sincos_pos_embed_from_grid(embed_dim, grid)


def _get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64) / (embed_dim / 2.0)
    omega = 1.0 / 10000.0**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class _BottleneckPatchEmbed(nn.Module):
    """JiT bottleneck patch embedding: Conv patchify + 1x1 to hidden_size."""

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_chans: int,
        pca_dim: int,
        embed_dim: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.img_size = img_size
        self.proj1 = nn.Conv2d(in_chans, pca_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.proj2 = nn.Conv2d(pca_dim, embed_dim, kernel_size=1, stride=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _b, _c, h, w = x.shape
        assert h == self.img_size and w == self.img_size
        x = self.proj2(self.proj1(x)).flatten(2).transpose(1, 2)
        return x


class _TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def _timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self._timestep_embedding(t, self.frequency_embedding_size))


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class _Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_norm: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.q_norm = _RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = _RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, rope: _VisionRotaryEmbeddingFast) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = rope(self.q_norm(q))
        k = rope(self.k_norm(k))
        attn_out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0, is_causal=False
        )
        attn_out = attn_out.transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(attn_out))


class _SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0, bias: bool = True) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(self.ffn_dropout(hidden))


class _FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm_final = _RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = _modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class _JiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = _RMSNorm(hidden_size, eps=1e-6)
        self.attn = _Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=True,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.norm2 = _RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = _SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor, feat_rope: _VisionRotaryEmbeddingFast) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            _modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(_modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


# ---------------------------------------------------------------------------
# Main backbone
# ---------------------------------------------------------------------------


class JiTBackbone(ModelMixin, ConfigMixin):
    """JiT backbone for the DDBM / bridge calling convention.

    Class labels and in-context tokens from the original JiT training recipe are
    disabled; use ``condition_mode='concat'`` and ``xT`` for image-to-image
    conditioning like :class:`SiTBackbone`.
    """

    @register_to_config
    def __init__(
        self,
        image_size: int = 256,
        patch_size: int = 16,
        in_channels: int = 3,
        hidden_size: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        bottleneck_dim: int = 128,
        condition_mode: Optional[str] = "concat",
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        _ = dropout  # API parity with SiTBackbone
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.out_channels = in_channels
        self.condition_mode = condition_mode

        embed_in = in_channels * 2 if condition_mode == "concat" else in_channels

        self.t_embedder = _TimestepEmbedder(hidden_size)
        self.x_embedder = _BottleneckPatchEmbed(
            image_size, patch_size, embed_in, bottleneck_dim, hidden_size, bias=True
        )
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        half_head_dim = hidden_size // num_heads // 2
        hw_seq_len = image_size // patch_size
        self.feat_rope = _VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
        )

        self.blocks = nn.ModuleList(
            [
                _JiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                    proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                )
                for i in range(depth)
            ]
        )
        self.final_layer = _FinalLayer(hidden_size, patch_size, self.out_channels)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        grid_size = int(self.x_embedder.num_patches**0.5)
        pos_embed = _get_2d_sincos_pos_embed(self.hidden_size, grid_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w1 = self.x_embedder.proj1.weight.data
        nn.init.xavier_uniform_(w1.view(w1.shape[0], -1))
        w2 = self.x_embedder.proj2.weight.data
        nn.init.xavier_uniform_(w2.view(w2.shape[0], -1))
        nn.init.constant_(self.x_embedder.proj2.bias, 0)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def _unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], c, h * p, w * p)

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        xT: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.condition_mode == "concat" and xT is not None:
            x = torch.cat([x, xT], dim=1)

        x = self.x_embedder(x) + self.pos_embed
        t = timestep.view(-1)
        c = self.t_embedder(t)

        for block in self.blocks:
            x = block(x, c, feat_rope=self.feat_rope)

        x = self.final_layer(x, c)
        return self._unpatchify(x)


# Named presets (depth, hidden_size, num_heads, bottleneck_dim, patch_size) — JiT paper configs
JIT_CONFIGS = {
    "B/16": (12, 768, 12, 128, 16),
    "B/32": (12, 768, 12, 128, 32),
    "L/16": (24, 1024, 16, 128, 16),
    "L/32": (24, 1024, 16, 128, 32),
    "H/16": (32, 1280, 16, 256, 16),
    "H/32": (32, 1280, 16, 256, 32),
}
