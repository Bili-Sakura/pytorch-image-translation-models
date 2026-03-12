# Copyright (c) 2026 EarthBridge Team.
# Credits: DDIB (Su et al., ICLR 2023), OpenAI improved_diffusion.

"""OpenAI-style unconditional UNet for DDIB (guided_diffusion / improved_diffusion format)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["OpenAIDDIBUNet"]


def _conv_nd(ndim: int, *args, **kwargs) -> nn.Module:
    if ndim == 2:
        return nn.Conv2d(*args, **kwargs)
    raise ValueError(f"ndim={ndim} not supported")


def _linear(in_ch: int, out_ch: int) -> nn.Linear:
    return nn.Linear(in_ch, out_ch)


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal timestep embedding."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000)
        * torch.arange(half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class TimestepBlock(nn.Module):
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Downsample(nn.Module):
    def __init__(self, channels: int, use_conv: bool, dims: int = 2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2
        if use_conv:
            self.op = _conv_nd(dims, channels, channels, 3, stride=stride, padding=1)
        else:
            self.op = nn.AvgPool2d(stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels: int, use_conv: bool, dims: int = 2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = _conv_nd(dims, channels, channels, 3, padding=1)
        else:
            self.conv = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x) if self.use_conv else x


def zero_module(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class ResBlock(TimestepBlock):
    def __init__(
        self,
        channels: int,
        emb_channels: int,
        dropout: float,
        out_channels: Optional[int] = None,
        use_conv: bool = False,
        use_scale_shift_norm: bool = False,
        dims: int = 2,
    ):
        super().__init__()
        self.out_channels = out_channels or channels
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            _conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            _linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(_conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)),
        )
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = _conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = _conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb)
        while emb_out.dim() < h.dim():
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            scale, shift = emb_out.chunk(2, dim=1)
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class QKVAttention(nn.Module):
    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        ch = qkv.shape[1] // 3
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        weight = F.softmax(weight.float(), dim=-1).to(weight.dtype)
        return torch.einsum("bts,bcs->bct", weight, v)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.attention = QKVAttention()
        self.proj_out = zero_module(nn.Conv2d(channels, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, *spatial = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        qkv = qkv.reshape(b, 3 * c, -1)
        qkv = qkv.reshape(b * self.num_heads, 3 * c // self.num_heads, -1)
        h = self.attention(qkv)
        h = h.reshape(b, c, *spatial)
        h = self.proj_out(h)
        return x + h


class OpenAIDDIBUNet(nn.Module):
    """OpenAI-style unconditional UNet for DDIB. Compatible with guided_diffusion checkpoints."""

    def __init__(
        self,
        in_channels: int = 3,
        model_channels: int = 256,
        out_channels: int = 3,
        num_res_blocks: int = 3,
        attention_resolutions: tuple[int, ...] = (2, 4, 8),
        dropout: float = 0.0,
        channel_mult: tuple[int, ...] = (1, 2, 2, 4),
        conv_resample: bool = False,
        use_scale_shift_norm: bool = False,
        num_heads: int = 1,
        time_embed_dim: Optional[int] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        time_embed_dim = time_embed_dim or (model_channels * 4)

        self.time_embed = nn.Sequential(
            _linear(model_channels, time_embed_dim),
            nn.SiLU(),
            _linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    _conv_nd(2, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=2,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(Downsample(ch, conv_resample, dims=2))
                )
                input_block_chans.append(ch)
            ds *= 2

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch, time_embed_dim, dropout, dims=2, use_scale_shift_norm=use_scale_shift_norm
            ),
            AttentionBlock(ch, num_heads=num_heads),
            ResBlock(
                ch, time_embed_dim, dropout, dims=2, use_scale_shift_norm=use_scale_shift_norm
            ),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResBlock(
                        ch + input_block_chans.pop(),
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=2,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample, dims=2))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            zero_module(_conv_nd(2, ch, out_channels, 3, padding=1)),
        )

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns predicted noise (or x0 depending on training setup)."""
        emb = self.time_embed(timestep_embedding(timestep, self.model_channels))
        hs = []
        h = x
        for module in self.input_blocks:
            h = module(h, emb) if isinstance(module, TimestepBlock) else module(h)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        return self.out(h)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        subfolder: str = "unet",
        device: str | torch.device = "cpu",
        **kwargs,
    ) -> "OpenAIDDIBUNet":
        """Load from subfolder (config.json + diffusion_pytorch_model.safetensors)."""
        path = Path(pretrained_model_name_or_path)
        unet_dir = path / subfolder
        config_path = unet_dir / "config.json"
        weights_path = unet_dir / "diffusion_pytorch_model.safetensors"

        if not config_path.exists() or not weights_path.exists():
            raise FileNotFoundError(
                f"DDIB community format requires {config_path} and {weights_path}"
            )

        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)

        def _t(x):
            return tuple(x) if isinstance(x, list) else x

        model = cls(
            in_channels=config.get("in_channels", 3),
            model_channels=config["model_channels"],
            out_channels=config.get("out_channels", 3),
            num_res_blocks=config.get("num_res_blocks", 3),
            attention_resolutions=_t(config.get("attention_resolutions", (2, 4, 8))),
            channel_mult=_t(config.get("channel_mult", (1, 2, 2, 4))),
            conv_resample=config.get("conv_resample", False),
            use_scale_shift_norm=config.get("use_scale_shift_norm", False),
            time_embed_dim=config.get("time_embed_dim"),
        )

        from safetensors.torch import load_file
        state = load_file(str(weights_path))
        model.load_state_dict(state, strict=True)
        model.eval()
        model.to(device)
        return model
