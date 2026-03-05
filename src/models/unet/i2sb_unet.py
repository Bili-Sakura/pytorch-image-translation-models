# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""I2SB UNet – ADM-style U-Net backbone for Image-to-Image Schrödinger Bridge."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Create sinusoidal timestep embeddings.

    Parameters
    ----------
    timesteps : Tensor of shape ``[B]``
        Timestep values (float or int).
    dim : int
        Embedding dimensionality.
    max_period : int
        Controls the minimum frequency of the embeddings.

    Returns
    -------
    Tensor of shape ``[B, dim]``
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps.float()[:, None] * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class TimestepMLP(nn.Module):
    """Two-layer MLP that projects the sinusoidal timestep embedding."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t_emb: torch.Tensor) -> torch.Tensor:
        return self.mlp(t_emb)


class ResBlock(nn.Module):
    """Residual block with timestep conditioning (GroupNorm + SiLU + Conv)."""

    def __init__(self, in_channels: int, emb_channels: int, out_channels: int | None = None) -> None:
        super().__init__()
        out_channels = out_channels or in_channels
        num_groups_1 = min(32, in_channels)
        num_groups_2 = min(32, out_channels)

        self.norm1 = nn.GroupNorm(num_groups_1, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.emb_proj = nn.Linear(emb_channels, out_channels)
        self.norm2 = nn.GroupNorm(num_groups_2, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip_proj = (
            nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = h + self.emb_proj(F.silu(emb))[:, :, None, None]
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip_proj(x)


class Downsample(nn.Module):
    """Strided convolution for spatial down-sampling (factor 2)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Nearest-neighbour up-sampling (factor 2) followed by a convolution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


# ---------------------------------------------------------------------------
# I2SB UNet
# ---------------------------------------------------------------------------

class I2SBUNet(nn.Module):
    """ADM-style U-Net for Image-to-Image Schrödinger Bridge (I2SB).

    Parameters
    ----------
    image_size : int
        Spatial resolution of the input.
    in_channels : int
        Number of channels in the input image.
    model_channels : int
        Base channel width of the network.
    out_channels : int
        Number of output channels (typically equals *in_channels*).
    num_res_blocks : int
        Number of residual blocks per resolution level.
    attention_resolutions : set[int]
        Spatial resolutions at which self-attention is applied.
    channel_mult : tuple[int, ...]
        Channel multipliers for each resolution level.
    condition_mode : str | None
        ``"concat"`` to concatenate the condition with the input, or
        ``None`` for unconditional.
    """

    def __init__(
        self,
        image_size: int,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        num_res_blocks: int,
        attention_resolutions: set[int],
        channel_mult: tuple[int, ...] = (1, 2, 4),
        condition_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.condition_mode = condition_mode
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.num_levels = len(channel_mult)

        actual_in = in_channels * 2 if condition_mode == "concat" else in_channels
        time_emb_dim = model_channels * 4

        # Timestep embedding
        self.time_embed = TimestepMLP(model_channels, time_emb_dim)

        # Input projection
        self.input_conv = nn.Conv2d(actual_in, model_channels, 3, padding=1)

        # ----- Encoder -----
        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        ch = model_channels
        skip_channels: list[int] = [ch]  # from input_conv

        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                self.encoder_blocks.append(ResBlock(ch, time_emb_dim, out_ch))
                ch = out_ch
                skip_channels.append(ch)
            if level < len(channel_mult) - 1:
                self.downsamples.append(Downsample(ch))
                skip_channels.append(ch)

        # ----- Bottleneck -----
        self.mid_block1 = ResBlock(ch, time_emb_dim)
        self.mid_block2 = ResBlock(ch, time_emb_dim)

        # ----- Decoder -----
        self.decoder_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for level in reversed(range(len(channel_mult))):
            out_ch = model_channels * channel_mult[level]
            for _ in range(num_res_blocks + 1):
                skip_ch = skip_channels.pop()
                self.decoder_blocks.append(ResBlock(ch + skip_ch, time_emb_dim, out_ch))
                ch = out_ch
            if level > 0:
                self.upsamples.append(Upsample(ch))

        # ----- Output -----
        num_groups_out = min(32, ch)
        self.out_norm = nn.GroupNorm(num_groups_out, ch)
        self.out_conv = nn.Conv2d(ch, out_channels, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor ``[B, C, H, W]``
            Noisy input.
        timesteps : Tensor ``[B]``
            Timestep values.
        cond : Tensor ``[B, C, H, W]``, optional
            Conditioning image (used when *condition_mode* is ``"concat"``).
        """
        if self.condition_mode == "concat" and cond is not None:
            x = torch.cat([x, cond], dim=1)

        emb = timestep_embedding(timesteps, self.model_channels)
        emb = self.time_embed(emb)

        # Encoder
        h = self.input_conv(x)
        skips: list[torch.Tensor] = [h]

        block_idx = 0
        down_idx = 0
        for level in range(self.num_levels):
            for _ in range(self.num_res_blocks):
                h = self.encoder_blocks[block_idx](h, emb)
                skips.append(h)
                block_idx += 1
            if level < self.num_levels - 1:
                h = self.downsamples[down_idx](h)
                skips.append(h)
                down_idx += 1

        # Bottleneck
        h = self.mid_block1(h, emb)
        h = self.mid_block2(h, emb)

        # Decoder
        block_idx = 0
        up_idx = 0
        for level in reversed(range(self.num_levels)):
            for _ in range(self.num_res_blocks + 1):
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = self.decoder_blocks[block_idx](h, emb)
                block_idx += 1
            if level > 0:
                h = self.upsamples[up_idx](h)
                up_idx += 1

        # Output
        h = F.silu(self.out_norm(h))
        return self.out_conv(h)
