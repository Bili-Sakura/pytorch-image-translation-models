# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.
#
# ADM UNet architecture credits:
#   Dhariwal, P., & Nichol, A. (2021). Diffusion Models Beat GANs on Image Synthesis.
#   In Advances in Neural Information Processing Systems (NeurIPS), Vol. 34.
#   https://proceedings.neurips.cc/paper/2021/hash/49ad23d1ec9fa4bd8d77d02681df5cfa-Abstract.html

"""ADM-style U-Net for diffusion-based image translation.

Provides both a native PyTorch implementation (I2SBUNet) and diffusers-compatible wrappers
(DDBMUNet, DDIBUNet, I2SBDiffusersUNet, BBDMUNet, BiBBDMUNet, BDBMUNet, DBIMUNet,
CDTSDEUNet, LBMUNet) that share the same ADM architecture (diffusers UNet2DModel).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers import ModelMixin, UNet2DModel
from diffusers.configuration_utils import ConfigMixin, register_to_config

__all__ = [
    "I2SBUNet",
    "create_model",
    "BBDMUNet",
    "BDBMUNet",
    "BiBBDMUNet",
    "CDTSDEUNet",
    "DBIMUNet",
    "DDBMUNet",
    "DDIBUNet",
    "I2SBDiffusersUNet",
    "LBMUNet",
]


# ---------------------------------------------------------------------------
# Native ADM UNet (I2SB backbone)
# ---------------------------------------------------------------------------


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Create sinusoidal timestep embeddings."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps.float()[:, None] * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


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

        self.time_embed = TimestepMLP(model_channels, time_emb_dim)
        self.input_conv = nn.Conv2d(actual_in, model_channels, 3, padding=1)

        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        ch = model_channels
        skip_channels: list[int] = [ch]

        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                self.encoder_blocks.append(ResBlock(ch, time_emb_dim, out_ch))
                ch = out_ch
                skip_channels.append(ch)
            if level < len(channel_mult) - 1:
                self.downsamples.append(Downsample(ch))
                skip_channels.append(ch)

        self.mid_block1 = ResBlock(ch, time_emb_dim)
        self.mid_block2 = ResBlock(ch, time_emb_dim)

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

        num_groups_out = min(32, ch)
        self.out_norm = nn.GroupNorm(num_groups_out, ch)
        self.out_conv = nn.Conv2d(ch, out_channels, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.condition_mode == "concat" and cond is not None:
            x = torch.cat([x, cond], dim=1)

        emb = timestep_embedding(timesteps, self.model_channels)
        emb = self.time_embed(emb)

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

        h = self.mid_block1(h, emb)
        h = self.mid_block2(h, emb)

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

        h = F.silu(self.out_norm(h))
        return self.out_conv(h)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_model(
    image_size: int,
    in_channels: int,
    num_channels: int,
    num_res_blocks: int,
    attention_resolutions: str = "",
    condition_mode: str | None = None,
    channel_mult: tuple[int, ...] | None = None,
    **kwargs,
) -> I2SBUNet:
    """Create an ADM-style I2SB UNet model."""
    if channel_mult is None:
        if image_size >= 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size >= 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size >= 64:
            channel_mult = (1, 2, 3, 4)
        else:
            channel_mult = (1, 2, 4)

    attn_res: set[int] = set()
    if isinstance(attention_resolutions, str) and attention_resolutions.strip():
        attn_res = {int(x) for x in attention_resolutions.split(",")}

    return I2SBUNet(
        image_size=image_size,
        in_channels=in_channels,
        model_channels=num_channels,
        out_channels=in_channels,
        num_res_blocks=num_res_blocks,
        attention_resolutions=attn_res,
        channel_mult=channel_mult,
        condition_mode=condition_mode,
    )


# ---------------------------------------------------------------------------
# Diffusers UNet2DModel wrappers (shared ADM architecture)
# ---------------------------------------------------------------------------


def _build_block_types(
    channel_mult: Tuple[int, ...],
    attention_resolutions: Tuple[int, ...],
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    down_block_types = [
        "AttnDownBlock2D" if i in attention_resolutions else "DownBlock2D"
        for i in range(len(channel_mult))
    ]
    up_block_types = [
        "AttnUpBlock2D" if (len(channel_mult) - 1 - i) in attention_resolutions else "UpBlock2D"
        for i in range(len(channel_mult))
    ]
    return tuple(down_block_types), tuple(up_block_types)


def _channel_mult_for_resolution(resolution: int) -> Tuple[int, ...]:
    return {
        1024: (1, 1, 2, 2, 4, 4),
        512: (1, 2, 4, 4, 8),
        256: (1, 2, 2, 4),
        128: (1, 1, 2, 3, 4),
        64: (1, 2, 3, 4),
        32: (1, 2, 3, 4),
    }.get(resolution, (1, 2, 3, 4))


def _from_pretrained_ema_fallback(cls, pretrained_model_name_or_path, **kwargs):
    """Load UNet; if ema_unet has no config.json, load config from unet and weights from ema_unet."""
    path = Path(pretrained_model_name_or_path)
    subfolder = kwargs.get("subfolder", "unet")
    if subfolder == "ema_unet" and not (path / "ema_unet" / "config.json").exists():
        unet = super(cls, cls).from_pretrained(
            path, subfolder="unet", **{k: v for k, v in kwargs.items() if k != "subfolder"}
        )
        from safetensors.torch import load_file

        ema_path = path / "ema_unet" / "diffusion_pytorch_model.safetensors"
        if ema_path.exists():
            unet.load_state_dict(load_file(str(ema_path)), strict=True)
        return unet
    return super(cls, cls).from_pretrained(pretrained_model_name_or_path, **kwargs)


class DDBMUNet(ModelMixin, ConfigMixin):
    """ADM UNet wrapper for DDBM calling convention ``(x, timestep, xT=…)``."""

    @register_to_config
    def __init__(
        self,
        image_size: int = 256,
        in_channels: int = 3,
        model_channels: int = 128,
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (1,),
        dropout: float = 0.0,
        condition_mode: Optional[str] = "concat",
        channel_mult: Optional[Tuple[int, ...]] = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.condition_mode = condition_mode

        if channel_mult is None:
            channel_mult = _channel_mult_for_resolution(image_size)

        unet_in_channels = in_channels * 2 if condition_mode == "concat" else in_channels
        block_out_channels = tuple(model_channels * m for m in channel_mult)
        down_block_types, up_block_types = _build_block_types(channel_mult, attention_resolutions)

        self.unet = UNet2DModel(
            sample_size=image_size,
            in_channels=unet_in_channels,
            out_channels=in_channels,
            block_out_channels=block_out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            layers_per_block=num_res_blocks,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        xT: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.condition_mode == "concat" and xT is not None:
            x = torch.cat([x, xT], dim=1)
        return self.unet(x, timestep).sample

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        return _from_pretrained_ema_fallback(cls, pretrained_model_name_or_path, **kwargs)


class DDIBUNet(ModelMixin, ConfigMixin):
    """ADM UNet wrapper for DDIB (unconditional)."""

    @register_to_config
    def __init__(
        self,
        image_size: int = 256,
        in_channels: int = 3,
        model_channels: int = 128,
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (1,),
        dropout: float = 0.0,
        learn_sigma: bool = False,
        channel_mult: Optional[Tuple[int, ...]] = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.learn_sigma = learn_sigma
        if channel_mult is None:
            channel_mult = _channel_mult_for_resolution(image_size)
        out_channels = in_channels * 2 if learn_sigma else in_channels
        block_out_channels = tuple(model_channels * m for m in channel_mult)
        down_block_types, up_block_types = _build_block_types(channel_mult, attention_resolutions)
        self.unet = UNet2DModel(
            sample_size=image_size,
            in_channels=in_channels,
            out_channels=out_channels,
            block_out_channels=block_out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            layers_per_block=num_res_blocks,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return self.unet(x, timestep).sample


class I2SBDiffusersUNet(ModelMixin, ConfigMixin):
    """ADM UNet wrapper for I2SB calling convention ``(x, timestep, cond=…)``."""

    @register_to_config
    def __init__(
        self,
        image_size: int = 256,
        in_channels: int = 3,
        model_channels: int = 128,
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (1,),
        dropout: float = 0.0,
        condition_mode: Optional[str] = "concat",
        channel_mult: Optional[Tuple[int, ...]] = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.condition_mode = condition_mode
        if channel_mult is None:
            channel_mult = _channel_mult_for_resolution(image_size)
        unet_in_channels = in_channels * 2 if condition_mode == "concat" else in_channels
        block_out_channels = tuple(model_channels * m for m in channel_mult)
        down_block_types, up_block_types = _build_block_types(channel_mult, attention_resolutions)
        self.unet = UNet2DModel(
            sample_size=image_size,
            in_channels=unet_in_channels,
            out_channels=in_channels,
            block_out_channels=block_out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            layers_per_block=num_res_blocks,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.condition_mode == "concat" and cond is not None:
            x = torch.cat([x, cond], dim=1)
        return self.unet(x, timestep).sample

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        return _from_pretrained_ema_fallback(cls, pretrained_model_name_or_path, **kwargs)


class BBDMUNet(ModelMixin, ConfigMixin):
    """ADM UNet wrapper for BBDM ``(x_t, timesteps, context=...)``."""

    @register_to_config
    def __init__(
        self,
        image_size: int = 256,
        in_channels: int = 3,
        out_channels: Optional[int] = None,
        model_channels: int = 128,
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (1,),
        dropout: float = 0.0,
        condition_mode: Optional[str] = "concat",
        channel_mult: Optional[Tuple[int, ...]] = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.condition_mode = condition_mode
        if out_channels is None:
            out_channels = in_channels
        self.out_channels = out_channels
        if channel_mult is None:
            channel_mult = _channel_mult_for_resolution(image_size)
        unet_in_channels = in_channels * 2 if condition_mode == "concat" else in_channels
        block_out_channels = tuple(model_channels * m for m in channel_mult)
        down_block_types, up_block_types = _build_block_types(channel_mult, attention_resolutions)
        self.unet = UNet2DModel(
            sample_size=image_size,
            in_channels=unet_in_channels,
            out_channels=out_channels,
            block_out_channels=block_out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            layers_per_block=num_res_blocks,
            dropout=dropout,
        )

    def forward(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.condition_mode == "concat" and context is not None:
            x_t = torch.cat([x_t, context], dim=1)
        return self.unet(x_t, timesteps).sample

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        return _from_pretrained_ema_fallback(cls, pretrained_model_name_or_path, **kwargs)


class BiBBDMUNet(ModelMixin, ConfigMixin):
    """ADM UNet wrapper for BiBBDM ``(x_t, timesteps, context=…)``."""

    @register_to_config
    def __init__(
        self,
        image_size: int = 256,
        in_channels: int = 3,
        out_channels: Optional[int] = None,
        model_channels: int = 128,
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (1,),
        dropout: float = 0.0,
        condition_mode: Optional[str] = "concat",
        channel_mult: Optional[Tuple[int, ...]] = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.condition_mode = condition_mode
        if out_channels is None:
            out_channels = in_channels
        self.out_channels = out_channels
        if channel_mult is None:
            channel_mult = _channel_mult_for_resolution(image_size)
        unet_in_channels = in_channels * 2 if condition_mode == "concat" else in_channels
        block_out_channels = tuple(model_channels * m for m in channel_mult)
        down_block_types, up_block_types = _build_block_types(channel_mult, attention_resolutions)
        self.unet = UNet2DModel(
            sample_size=image_size,
            in_channels=unet_in_channels,
            out_channels=out_channels,
            block_out_channels=block_out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            layers_per_block=num_res_blocks,
            dropout=dropout,
        )

    def forward(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.condition_mode == "concat" and context is not None:
            x_t = torch.cat([x_t, context], dim=1)
        return self.unet(x_t, timesteps).sample

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        return _from_pretrained_ema_fallback(cls, pretrained_model_name_or_path, **kwargs)


class BDBMUNet(ModelMixin, ConfigMixin):
    """ADM UNet wrapper for BDBM ``(x_t, timesteps, context=…)``."""

    @register_to_config
    def __init__(
        self,
        image_size: int = 256,
        in_channels: int = 3,
        out_channels: Optional[int] = None,
        model_channels: int = 128,
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (1,),
        dropout: float = 0.0,
        condition_mode: Optional[str] = "concat",
        channel_mult: Optional[Tuple[int, ...]] = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.condition_mode = condition_mode
        if out_channels is None:
            out_channels = in_channels
        self.out_channels = out_channels
        if channel_mult is None:
            channel_mult = _channel_mult_for_resolution(image_size)
        unet_in_channels = in_channels * 2 if condition_mode == "concat" else in_channels
        block_out_channels = tuple(model_channels * m for m in channel_mult)
        down_block_types, up_block_types = _build_block_types(channel_mult, attention_resolutions)
        self.unet = UNet2DModel(
            sample_size=image_size,
            in_channels=unet_in_channels,
            out_channels=out_channels,
            block_out_channels=block_out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            layers_per_block=num_res_blocks,
            dropout=dropout,
        )

    def forward(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.condition_mode == "concat" and context is not None:
            x_t = torch.cat([x_t, context], dim=1)
        return self.unet(x_t, timesteps).sample


class DBIMUNet(ModelMixin, ConfigMixin):
    """ADM UNet wrapper for DBIM ``(x, timestep, xT=source)``."""

    @register_to_config
    def __init__(
        self,
        image_size: int = 256,
        in_channels: int = 3,
        out_channels: Optional[int] = None,
        model_channels: int = 128,
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (1,),
        dropout: float = 0.0,
        condition_mode: Optional[str] = "concat",
        channel_mult: Optional[Tuple[int, ...]] = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.condition_mode = condition_mode
        if out_channels is None:
            out_channels = in_channels
        self.out_channels = out_channels
        if channel_mult is None:
            channel_mult = _channel_mult_for_resolution(image_size)
        unet_in_channels = in_channels * 2 if condition_mode == "concat" else in_channels
        block_out_channels = tuple(model_channels * m for m in channel_mult)
        down_block_types, up_block_types = _build_block_types(channel_mult, attention_resolutions)
        self.unet = UNet2DModel(
            sample_size=image_size,
            in_channels=unet_in_channels,
            out_channels=out_channels,
            block_out_channels=block_out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            layers_per_block=num_res_blocks,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        xT: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.condition_mode == "concat" and xT is not None:
            x = torch.cat([x, xT], dim=1)
        return self.unet(x, timestep).sample


class CDTSDEUNet(ModelMixin, ConfigMixin):
    """ADM UNet wrapper for CDTSDE ``(x, timestep, xT=source)``."""

    @register_to_config
    def __init__(
        self,
        image_size: int = 256,
        in_channels: int = 3,
        out_channels: Optional[int] = None,
        model_channels: int = 128,
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (1,),
        dropout: float = 0.0,
        condition_mode: Optional[str] = "concat",
        channel_mult: Optional[Tuple[int, ...]] = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.condition_mode = condition_mode
        if out_channels is None:
            out_channels = in_channels
        self.out_channels = out_channels
        if channel_mult is None:
            channel_mult = _channel_mult_for_resolution(image_size)
        unet_in_channels = in_channels * 2 if condition_mode == "concat" else in_channels
        block_out_channels = tuple(model_channels * m for m in channel_mult)
        down_block_types, up_block_types = _build_block_types(channel_mult, attention_resolutions)
        self.unet = UNet2DModel(
            sample_size=image_size,
            in_channels=unet_in_channels,
            out_channels=out_channels,
            block_out_channels=block_out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            layers_per_block=num_res_blocks,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        xT: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.condition_mode == "concat" and xT is not None:
            x = torch.cat([x, xT], dim=1)
        return self.unet(x, timestep).sample


class LBMUNet(ModelMixin, ConfigMixin):
    """ADM UNet wrapper for LBM ``(sample, timestep, cond=…)``."""

    @register_to_config
    def __init__(
        self,
        image_size: int = 256,
        in_channels: int = 3,
        out_channels: Optional[int] = None,
        model_channels: int = 128,
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (1,),
        dropout: float = 0.0,
        condition_mode: Optional[str] = "concat",
        channel_mult: Optional[Tuple[int, ...]] = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.condition_mode = condition_mode
        if out_channels is None:
            out_channels = in_channels
        self.out_channels = out_channels
        if channel_mult is None:
            channel_mult = _channel_mult_for_resolution(image_size)
        unet_in_channels = in_channels * 2 if condition_mode == "concat" else in_channels
        block_out_channels = tuple(model_channels * m for m in channel_mult)
        down_block_types, up_block_types = _build_block_types(channel_mult, attention_resolutions)
        self.unet = UNet2DModel(
            sample_size=image_size,
            in_channels=unet_in_channels,
            out_channels=out_channels,
            block_out_channels=block_out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            layers_per_block=num_res_blocks,
            dropout=dropout,
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.condition_mode == "concat" and cond is not None:
            sample = torch.cat([sample, cond], dim=1)
        return self.unet(sample, timestep).sample
