# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.
#
# VDM++ (Variational Diffusion Models ++) – port from google-research/vdm.
#
# References:
#   Kingma, D., Salimans, T., Poole, B., & Ho, J. (2021).
#   Variational Diffusion Models. NeurIPS 2021.
#   Kingma, D., & Gao, R. (2023).
#   Understanding Diffusion Objectives as the ELBO with Simple Data Augmentation.
#   NeurIPS 2023.
#
# Architecture: ScoreUNet from https://github.com/google-research/vdm (JAX/Flax).
# VDM++ uses the same UNet; the improvement is in the training objective.

"""VDM++ UNet – ScoreUNet architecture from google-research/vdm.

Flat U-Net (no spatial downsampling) with ResNet blocks, self-attention,
optional Fourier features, and sinusoidal timestep embedding. Predicts x0 for
DDBM compatibility.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config

__all__ = ["VDMppUNet"]


# ---------------------------------------------------------------------------
# Timestep embedding (from VDM / Fairseq)
# ---------------------------------------------------------------------------


def timestep_embedding(
    timesteps: torch.Tensor, embedding_dim: int, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Build sinusoidal timestep embeddings (VDM/Fairseq style)."""
    assert timesteps.dim() == 1
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=dtype, device=timesteps.device) * -emb)
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1), value=0)
    return emb


# ---------------------------------------------------------------------------
# Fourier features (from VDM)
# ---------------------------------------------------------------------------


class Base2FourierFeatures(nn.Module):
    """Base-2 Fourier features for input augmentation (VDM style)."""

    def __init__(self, start: int = 0, stop: int = 8, step: int = 1) -> None:
        super().__init__()
        freqs = list(range(start, stop, step))
        self.register_buffer(
            "freqs",
            torch.tensor([2.0**f * 2 * math.pi for f in freqs], dtype=torch.float32),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> repeat channels for each freq
        B, C, H, W = x.shape
        n_freqs = self.freqs.shape[0]
        w = self.freqs.view(1, -1, 1).expand(1, n_freqs, C).reshape(-1)
        x_flat = x.permute(0, 2, 3, 1).reshape(B * H * W, C)  # (B*H*W, C)
        h = x_flat.unsqueeze(1).expand(-1, n_freqs, C).reshape(-1, C)
        h = w.unsqueeze(0) * h
        h = torch.cat([torch.sin(h), torch.cos(h)], dim=1)
        h = h.view(B, H, W, n_freqs * C * 2).permute(0, 3, 1, 2)
        return h


# ---------------------------------------------------------------------------
# ResNet block with conditioning (from VDM)
# ---------------------------------------------------------------------------


class VDMResnetBlock(nn.Module):
    """Convolutional residual block with conditioning (VDM ScoreUNet style)."""

    def __init__(
        self,
        in_channels: int,
        emb_channels: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        out_channels = out_channels or in_channels

        self.norm1 = nn.GroupNorm(min(32, in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.cond_proj = nn.Linear(emb_channels, out_channels)
        self.norm2 = nn.GroupNorm(min(32, out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        self.dropout = nn.Dropout(dropout)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor, deterministic: bool = True
    ) -> torch.Tensor:
        B = x.shape[0]
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = h + self.cond_proj(cond).view(B, -1, 1, 1)
        h = F.silu(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.skip(x)


# ---------------------------------------------------------------------------
# Self-attention block (from VDM)
# ---------------------------------------------------------------------------


class VDMAttnBlock(nn.Module):
    """Self-attention residual block (VDM style)."""

    def __init__(self, channels: int, num_heads: int = 1) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        q = self.q(h).reshape(B, C, -1).permute(0, 2, 1)  # (B, HW, C)
        k = self.k(h).reshape(B, C, -1)  # (B, C, HW)
        v = self.v(h).reshape(B, C, -1)  # (B, C, HW)

        scale = (C // self.num_heads) ** -0.5
        attn = torch.bmm(q, k) * scale
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, v.permute(0, 2, 1))  # (B, HW, C)
        out = out.permute(0, 2, 1).reshape(B, C, H, W)
        return x + self.proj(out)


# ---------------------------------------------------------------------------
# VDM++ ScoreUNet (PyTorch port)
# ---------------------------------------------------------------------------


class VDMppUNet(ModelMixin, ConfigMixin):
    """VDM++ ScoreUNet – flat U-Net from google-research/vdm.

    Predicts denoised sample (x0) for DDBM compatibility. Uses the same
    architecture as VDM; VDM++ differs only in the training objective.
    """

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: Optional[int] = None,
        model_channels: int = 128,
        num_layers: int = 4,
        dropout: float = 0.0,
        with_fourier_features: bool = True,
        with_attention: bool = True,
        condition_mode: Optional[str] = "concat",
    ) -> None:
        super().__init__()
        out_channels = out_channels or in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.condition_mode = condition_mode

        n_embd = model_channels
        emb_channels = n_embd * 4

        self.with_fourier_features = with_fourier_features
        self.with_attention = with_attention

        # Timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(n_embd, emb_channels),
            nn.SiLU(),
            nn.Linear(emb_channels, emb_channels),
        )

        # Optional Fourier features on input (VDM: start=6, stop=8, step=1 -> 2 freqs, sin+cos -> 4x channels)
        if with_fourier_features:
            self.fourier = Base2FourierFeatures(start=6, stop=8, step=1)
            conv_in_ch = in_channels + in_channels * 4  # original + 2 freqs * (sin+cos)
        else:
            self.fourier = None
            conv_in_ch = in_channels

        self.conv_in = nn.Conv2d(conv_in_ch, n_embd, 3, padding=1)

        # Down blocks (one Resnet+Attn per layer, append once per layer for skips)
        self.down_resnets = nn.ModuleList()
        self.down_attns = nn.ModuleList()
        for _ in range(num_layers):
            self.down_resnets.append(
                VDMResnetBlock(n_embd, emb_channels, n_embd, dropout=dropout)
            )
            self.down_attns.append(
                VDMAttnBlock(n_embd) if with_attention else nn.Identity()
            )

        # Middle
        self.mid_block1 = VDMResnetBlock(n_embd, emb_channels, n_embd, dropout=dropout)
        self.mid_attn = VDMAttnBlock(n_embd) if with_attention else nn.Identity()
        self.mid_block2 = VDMResnetBlock(n_embd, emb_channels, n_embd, dropout=dropout)

        # Up blocks (num_layers+1 Resnet+Attn, each concats one skip)
        self.up_resnets = nn.ModuleList()
        self.up_attns = nn.ModuleList()
        for _ in range(num_layers + 1):
            self.up_resnets.append(
                VDMResnetBlock(n_embd * 2, emb_channels, n_embd, dropout=dropout)
            )
            self.up_attns.append(
                VDMAttnBlock(n_embd) if with_attention else nn.Identity()
            )

        # Output
        self.norm_out = nn.GroupNorm(min(32, n_embd), n_embd)
        self.conv_out = nn.Conv2d(n_embd, out_channels, 3, padding=1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        xT: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass. Predicts x0 (denoised sample)."""
        if self.condition_mode == "concat" and xT is not None:
            x = torch.cat([x, xT], dim=1)

        B = x.shape[0]
        if timestep.dim() == 0:
            timestep = timestep.expand(B)
        if timestep.dim() == 1 and timestep.shape[0] == 1:
            timestep = timestep.expand(B)

        # Normalize timestep to [0, 1] for embedding (VDM style)
        # Pipeline uses rescaled_t = 250 * log(sigma + 1e-44); we map to [0,1]
        t = timestep.float()
        t_min, t_max = -500.0, 500.0
        t_norm = ((t - t_min) / (t_max - t_min)).clamp(0.0, 1.0) * 1000.0
        emb = timestep_embedding(t_norm, self.config.model_channels, x.dtype)
        cond = self.time_embed(emb)

        # Input
        if self.fourier is not None:
            z_f = self.fourier(x)
            h = torch.cat([x, z_f], dim=1)
        else:
            h = x
        h = self.conv_in(h)
        hs = [h]

        # Down (append once per layer)
        det = not self.training
        for resnet, attn in zip(self.down_resnets, self.down_attns):
            h = resnet(h, cond, deterministic=det)
            h = attn(h)
            hs.append(h)

        # Middle
        h = self.mid_block1(h, cond, deterministic=det)
        h = self.mid_attn(h)
        h = self.mid_block2(h, cond, deterministic=det)

        # Up (pop one skip per block)
        for resnet, attn in zip(self.up_resnets, self.up_attns):
            h = torch.cat([h, hs.pop()], dim=1)
            h = resnet(h, cond, deterministic=det)
            h = attn(h)
        assert len(hs) == 0

        h = F.silu(self.norm_out(h))
        return self.conv_out(h)
