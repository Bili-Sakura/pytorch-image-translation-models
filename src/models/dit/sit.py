# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""SiT (Scalable Interpolant Transformer) backbone adapted for the DDBM calling convention.

SiT (Ma et al., 2024) is a diffusion transformer that uses stochastic interpolants
for generative modeling.  It adopts the DiT architecture (Peebles & Xie, 2023)
with adaLN-Zero conditioning and sinusoidal 2-D positional embeddings.

This module provides the **backbone** (denoiser network) so it can serve as an
alternative architecture to the diffusers UNet in the DDBM bridge framework.
The wrapper class :class:`SiTBackbone` exposes the same forward signature as
:class:`DDBMUNet`::

    output = model(x, timestep, xT=source)

so the rest of the training / sampling code does not need any changes.

Architecture differences from the original SiT:
* Class-label conditioning is removed; the model is conditioned on timestep only
  (with the source image concatenated along the channel axis in ``concat`` mode).
* ``learn_sigma`` is disabled (image-to-image translation does not need variance
  prediction from the same network).
* ``timm`` dependency is replaced with pure-PyTorch implementations of
  PatchEmbed, multi-head self-attention, and MLP to maximise portability.

Reference: https://github.com/willisma/SiT
Paper: https://arxiv.org/abs/2401.08740
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


# ---------------------------------------------------------------------------
# Helper: sinusoidal 2-D positional embeddings  (ported from SiT / MAE)
# ---------------------------------------------------------------------------


def _get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    """Return sin-cos 2-D positional embedding.

    Returns
    -------
    np.ndarray of shape (grid_size * grid_size, embed_dim)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # w goes first (matches SiT)
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
    omega = 1.0 / 10000.0 ** omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


# ---------------------------------------------------------------------------
# Building blocks  (pure-PyTorch, no timm dependency)
# ---------------------------------------------------------------------------


class _PatchEmbed(nn.Module):
    """2-D image → patch tokens via a single Conv2d (same as timm PatchEmbed)."""

    def __init__(self, img_size: int, patch_size: int, in_chans: int, embed_dim: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)  # (B, N, embed_dim)


class _Attention(nn.Module):
    """Multi-head self-attention (no dropout, uses F.scaled_dot_product_attention)."""

    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each (B, H, N, head_dim)
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, C)
        return self.proj(attn_out)


class _Mlp(nn.Module):
    """Two-layer MLP with GELU activation (matches timm.Mlp defaults)."""

    def __init__(self, in_features: int, hidden_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.fc2 = nn.Linear(hidden_features, in_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class _TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding → MLP (identical to SiT TimestepEmbedder)."""

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


class _SiTBlock(nn.Module):
    """SiT block: adaLN-Zero + multi-head attention + MLP (matches SiT SiTBlock)."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = _Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = _Mlp(in_features=hidden_size, hidden_features=mlp_hidden)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(_modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(_modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class _FinalLayer(nn.Module):
    """SiT final layer: adaLN + linear projection to pixels."""

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = _modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


# ---------------------------------------------------------------------------
# Main backbone
# ---------------------------------------------------------------------------


class SiTBackbone(ModelMixin, ConfigMixin):
    """SiT (Scalable Interpolant Transformer) backbone for the DDBM calling convention.

    Adapts the SiT architecture (Ma et al., 2024) for image-to-image diffusion
    bridges in DDBM / I2SB / BiBBDM.  Class-label conditioning is replaced by
    pure timestep conditioning; the source image is concatenated along the
    channel axis when ``condition_mode='concat'``.

    Architecture (identical to SiT DiT blocks):
    1. Patchify input → linear patch embedding (Conv2d)
    2. Add frozen sin-cos 2-D positional embeddings
    3. ``depth`` SiT blocks (adaLN-Zero + multi-head attn + MLP)
    4. Final adaLN layer → unpatchify to image space

    Parameters
    ----------
    image_size : int
        Spatial resolution (H == W).  Must be divisible by ``patch_size``.
    patch_size : int
        Side length of each non-overlapping patch.
    in_channels : int
        Channels of the *target* image.  When ``condition_mode='concat'``
        the network internally doubles the input channels.
    hidden_size : int
        Transformer hidden dimension.
    depth : int
        Number of SiT transformer blocks.
    num_heads : int
        Number of attention heads (``hidden_size`` must be divisible by
        ``num_heads``).
    mlp_ratio : float
        MLP hidden-dim multiplier relative to ``hidden_size``.
    condition_mode : str or None
        ``'concat'`` to concatenate source image along channels, or ``None``
        for unconditional mode.
    dropout : float
        Currently unused (kept for API compatibility with other backbones).
    """

    @register_to_config
    def __init__(
        self,
        image_size: int = 256,
        patch_size: int = 2,
        in_channels: int = 3,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        condition_mode: Optional[str] = "concat",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.out_channels = in_channels  # no learn_sigma for image translation
        self.condition_mode = condition_mode

        embed_in = in_channels * 2 if condition_mode == "concat" else in_channels

        # Patch embedding
        self.x_embedder = _PatchEmbed(image_size, patch_size, embed_in, hidden_size)
        num_patches = self.x_embedder.num_patches

        # Timestep embedding
        self.t_embedder = _TimestepEmbedder(hidden_size)

        # Frozen positional embeddings (sin-cos 2-D, not learned)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size), requires_grad=False
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            _SiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])

        # Final projection
        self.final_layer = _FinalLayer(hidden_size, patch_size, self.out_channels)

        self._initialize_weights()

    # ---- init helpers -------------------------------------------------------

    def _initialize_weights(self) -> None:
        """Weight initialization following SiT / DiT conventions."""
        # xavier_uniform for all linear layers
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Frozen sin-cos positional embedding
        grid_size = int(self.x_embedder.num_patches ** 0.5)
        pos_embed = _get_2d_sincos_pos_embed(self.hidden_size, grid_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch embedding Conv2d like Linear
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Zero-out timestep embedding MLP (SiT convention)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation linear layers in SiT blocks
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out final layer outputs
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    # ---- unpatchify ---------------------------------------------------------

    def _unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """Fold patch tokens back to image tensor.

        Parameters
        ----------
        x : Tensor  (B, N, patch_size**2 * C)

        Returns
        -------
        Tensor  (B, C, H, W)
        """
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1], "Number of patches must form a square grid."
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], c, h * p, w * p)

    # ---- forward ------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        xT: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass matching the DDBM UNet calling convention.

        Parameters
        ----------
        x : Tensor  (B, C, H, W)
            Pre-conditioned noisy sample.
        timestep : Tensor  (B,) or scalar
            Rescaled log-sigma timestep.
        xT : Tensor or None  (B, C, H, W)
            Source/condition image.  Required when ``condition_mode='concat'``.

        Returns
        -------
        Tensor  (B, C, H, W)
            Raw model output.
        """
        if self.condition_mode == "concat" and xT is not None:
            x = torch.cat([x, xT], dim=1)

        # Patchify + positional embedding
        x = self.x_embedder(x) + self.pos_embed  # (B, N, D)

        # Timestep conditioning
        t = timestep.view(-1)
        c = self.t_embedder(t)  # (B, D)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, c)

        # Final layer + unpatchify
        x = self.final_layer(x, c)  # (B, N, patch_size**2 * C)
        return self._unpatchify(x)   # (B, C, H, W)


# ---------------------------------------------------------------------------
# Named size configurations  (mirrors SiT paper Table 1)
# ---------------------------------------------------------------------------

SIT_CONFIGS = {
    # name: (depth, hidden_size, num_heads)
    "S": (12, 384, 6),
    "B": (12, 768, 12),
    "L": (24, 1024, 16),
    "XL": (28, 1152, 16),
}
