# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Diffusers-compatible UNet wrappers for image-translation bridge models.

Each wrapper pairs :class:`~diffusers.ModelMixin` / :class:`~diffusers.ConfigMixin`
with an inner :class:`~diffusers.UNet2DModel`, exposing the specific calling
convention required by its method (DDBM, DDIB, I2SB, BBDM, BiBBDM, BDBM, DBIM,
CDTSDE, LBM).  Consolidating them in a single module avoids duplicating the
shared helper logic that was previously copy-pasted across
``examples/pipelines/*/pipeline.py``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from diffusers import ModelMixin, UNet2DModel
from diffusers.configuration_utils import ConfigMixin, register_to_config

__all__ = [
    "BBDMUNet",
    "DDBMUNet",
    "DDIBUNet",
    "I2SBDiffusersUNet",
    "BiBBDMUNet",
    "BDBMUNet",
    "DBIMUNet",
    "CDTSDEUNet",
    "LBMUNet",
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _build_block_types(
    channel_mult: Tuple[int, ...],
    attention_resolutions: Tuple[int, ...],
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Build ``down_block_types`` and ``up_block_types`` from *channel_mult* and attention indices."""
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
    """Return a sensible default channel multiplier tuple.

    ADM-style: 256px → 4 stages, 512px → 5 stages, 1024px → 6 stages.
    """
    return {
        1024: (1, 1, 2, 2, 4, 4),
        512: (1, 2, 4, 4, 8),
        256: (1, 2, 2, 4),
        128: (1, 1, 2, 3, 4),
        64: (1, 2, 3, 4),
        32: (1, 2, 3, 4),
    }.get(resolution, (1, 2, 3, 4))


# ---------------------------------------------------------------------------
# DDBMUNet – Denoising Diffusion Bridge Models
# ---------------------------------------------------------------------------


class DDBMUNet(ModelMixin, ConfigMixin):
    """Wrapper around :class:`UNet2DModel` that accepts the DDBM calling convention ``(x, timestep, xT=…)``."""

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
        """Load UNet, preferring ema_unet weights when subfolder is ema_unet.

        If ``ema_unet/`` has no ``config.json``, load config from ``unet/`` and
        weights from ``ema_unet/``.
        """
        from pathlib import Path

        path = Path(pretrained_model_name_or_path)
        subfolder = kwargs.get("subfolder", "unet")
        if subfolder == "ema_unet" and not (path / "ema_unet" / "config.json").exists():
            unet = super().from_pretrained(path, subfolder="unet", **{k: v for k, v in kwargs.items() if k != "subfolder"})
            from safetensors.torch import load_file
            ema_path = path / "ema_unet" / "diffusion_pytorch_model.safetensors"
            if ema_path.exists():
                state = load_file(str(ema_path))
                unet.load_state_dict(state, strict=True)
            return unet
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)


# ---------------------------------------------------------------------------
# DDIBUNet – Dual Diffusion Implicit Bridges
# ---------------------------------------------------------------------------


class DDIBUNet(ModelMixin, ConfigMixin):
    """Unconditional UNet for DDIB diffusion models."""

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


# ---------------------------------------------------------------------------
# I2SBUNet – Image-to-Image Schrödinger Bridge
# ---------------------------------------------------------------------------


class I2SBDiffusersUNet(ModelMixin, ConfigMixin):
    """Wrapper around :class:`UNet2DModel` that accepts the I2SB calling convention ``(x, timestep, cond=…)``."""

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
        """Load UNet; if ``ema_unet`` has no ``config.json``, load config from ``unet`` and weights from ``ema_unet``."""
        from pathlib import Path

        path = Path(pretrained_model_name_or_path)
        subfolder = kwargs.get("subfolder", "unet")
        if subfolder == "ema_unet" and not (path / "ema_unet" / "config.json").exists():
            unet = super().from_pretrained(path, subfolder="unet", **{k: v for k, v in kwargs.items() if k != "subfolder"})
            from safetensors.torch import load_file
            ema_path = path / "ema_unet" / "diffusion_pytorch_model.safetensors"
            if ema_path.exists():
                unet.load_state_dict(load_file(str(ema_path)), strict=True)
            return unet
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)


# ---------------------------------------------------------------------------
# BBDMUNet – Brownian Bridge Diffusion Models
# ---------------------------------------------------------------------------


class BBDMUNet(ModelMixin, ConfigMixin):
    """Wrapper around :class:`UNet2DModel` for BBDM ``(x_t, timesteps, context=...)``."""

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
        """Load UNet; if ``ema_unet`` has no ``config.json``, load config from ``unet`` and weights from ``ema_unet``."""
        from pathlib import Path

        path = Path(pretrained_model_name_or_path)
        subfolder = kwargs.get("subfolder", "unet")
        if subfolder == "ema_unet" and not (path / "ema_unet" / "config.json").exists():
            unet = super().from_pretrained(path, subfolder="unet", **{k: v for k, v in kwargs.items() if k != "subfolder"})
            from safetensors.torch import load_file
            ema_path = path / "ema_unet" / "diffusion_pytorch_model.safetensors"
            if ema_path.exists():
                unet.load_state_dict(load_file(str(ema_path)), strict=True)
            return unet
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)


# ---------------------------------------------------------------------------
# BiBBDMUNet – Bidirectional Brownian Bridge Diffusion Models
# ---------------------------------------------------------------------------


class BiBBDMUNet(ModelMixin, ConfigMixin):
    """Wrapper around :class:`UNet2DModel` for BiBBDM ``(x_t, timesteps, context=…)``."""

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
        """Load UNet; if ``ema_unet`` has no ``config.json``, load config from ``unet`` and weights from ``ema_unet``."""
        from pathlib import Path

        path = Path(pretrained_model_name_or_path)
        subfolder = kwargs.get("subfolder", "unet")
        if subfolder == "ema_unet" and not (path / "ema_unet" / "config.json").exists():
            unet = super().from_pretrained(path, subfolder="unet", **{k: v for k, v in kwargs.items() if k != "subfolder"})
            from safetensors.torch import load_file
            ema_path = path / "ema_unet" / "diffusion_pytorch_model.safetensors"
            if ema_path.exists():
                unet.load_state_dict(load_file(str(ema_path)), strict=True)
            return unet
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)


# ---------------------------------------------------------------------------
# BDBMUNet – Bidirectional Diffusion Bridge Models
# ---------------------------------------------------------------------------


class BDBMUNet(ModelMixin, ConfigMixin):
    """Wrapper around :class:`UNet2DModel` for BDBM ``(x_t, timesteps, context=…)``."""

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


# ---------------------------------------------------------------------------
# DBIMUNet – Diffusion Bridge Implicit Models
# ---------------------------------------------------------------------------


class DBIMUNet(ModelMixin, ConfigMixin):
    """Wrapper around :class:`UNet2DModel` for DBIM ``(x, timestep, xT=source)``."""

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


# ---------------------------------------------------------------------------
# CDTSDEUNet – Conditional Doob's Transform SDE
# ---------------------------------------------------------------------------


class CDTSDEUNet(ModelMixin, ConfigMixin):
    """Wrapper around :class:`UNet2DModel` for CDTSDE ``(x, timestep, xT=source)``."""

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


# ---------------------------------------------------------------------------
# LBMUNet – Latent Bridge Matching
# ---------------------------------------------------------------------------


class LBMUNet(ModelMixin, ConfigMixin):
    """Wrapper around :class:`UNet2DModel` for LBM ``(sample, timestep, cond=…)``."""

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
