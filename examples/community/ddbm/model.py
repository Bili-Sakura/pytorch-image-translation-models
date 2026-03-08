# Copyright (c) 2026 EarthBridge Team.
# Credits: OpenAI improved_diffusion (https://github.com/openai/improved-diffusion).

"""OpenAI-style UNet for DDBM checkpoints (input_blocks / middle_block / output_blocks).

Compatible with alexzhou907/DDBM checkpoints (improved_diffusion format).
Loads from unet/diffusion_pytorch_model.safetensors (converted from raw .pt).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["OpenAIDDBMUNet"]


def _conv_nd(ndim: int, *args, **kwargs) -> nn.Module:
    if ndim == 1:
        return nn.Conv1d(*args, **kwargs)
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
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = _conv_nd(dims, channels, channels, 3, stride=stride, padding=1)
        else:
            stride_val = stride if isinstance(stride, int) else stride[0]
            self.op = (
                nn.AvgPool2d(stride_val) if dims == 2 else nn.AvgPool3d(stride_val)
            )

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
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm
        self.use_checkpoint = use_checkpoint

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            _conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            _linear(
                emb_channels,
                2 * self.out_channels
                if use_scale_shift_norm
                else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                _conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = _conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
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
    def __init__(
        self, channels: int, num_heads: int = 1, use_checkpoint: bool = False
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint
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


class OpenAIDDBMUNetCore(nn.Module):
    """OpenAI improved_diffusion UNetModel. Loads from safetensors (OpenAI key format)."""

    def __init__(
        self,
        in_channels: int = 6,
        model_channels: int = 192,
        out_channels: int = 3,
        num_res_blocks: int = 3,
        attention_resolutions: tuple[int, ...] = (2, 4, 8),
        dropout: float = 0.0,
        channel_mult: tuple[int, ...] = (1, 2, 3, 4),
        conv_resample: bool = False,
        use_scale_shift_norm: bool = True,
        num_heads: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        time_embed_dim = model_channels * 4

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
                    TimestepEmbedSequential(
                        Downsample(ch, conv_resample, dims=2)
                    )
                )
                input_block_chans.append(ch)
            ds *= 2

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=2,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(ch, num_heads=num_heads),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=2,
                use_scale_shift_norm=use_scale_shift_norm,
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
            zero_module(
                _conv_nd(2, model_channels, out_channels, 3, padding=1)
            ),
        )

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        emb = self.time_embed(
            timestep_embedding(timesteps, self.model_channels)
        )
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


class OpenAIDDBMUNet(nn.Module):
    """Wrapper exposing DDBM interface (x, timestep, xT) for OpenAI UNet checkpoints."""

    def __init__(
        self,
        in_channels: int = 3,
        model_channels: int = 192,
        out_channels: int = 3,
        num_res_blocks: int = 3,
        attention_resolutions: tuple[int, ...] = (2, 4, 8),
        dropout: float = 0.0,
        channel_mult: tuple[int, ...] = (1, 2, 3, 4),
        conv_resample: bool = False,
        use_scale_shift_norm: bool = True,
        condition_mode: str = "concat",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.condition_mode = condition_mode
        unet_in = in_channels * 2 if condition_mode == "concat" else in_channels
        self.unet = OpenAIDDBMUNetCore(
            in_channels=unet_in,
            model_channels=model_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            channel_mult=channel_mult,
            conv_resample=conv_resample,
            use_scale_shift_norm=use_scale_shift_norm,
        )

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        xT: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.condition_mode == "concat" and xT is not None:
            x = torch.cat([x, xT], dim=1)
        return self.unet(x, timestep)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        subfolder: str = "unet",
        device: str | torch.device = "cpu",
        **kwargs,
    ) -> "OpenAIDDBMUNet":
        """Load from unet/ (config.json + diffusion_pytorch_model.safetensors)."""
        path = Path(pretrained_model_name_or_path)
        unet_dir = path / subfolder
        config_path = unet_dir / "config.json"
        weights_path = unet_dir / "diffusion_pytorch_model.safetensors"

        if not config_path.exists() or not weights_path.exists():
            raise FileNotFoundError(
                f"DDBM community format requires {config_path} and {weights_path}"
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
            channel_mult=_t(config.get("channel_mult", (1, 2, 3, 4))),
            conv_resample=config.get("conv_resample", False),
            condition_mode=config.get("condition_mode", "concat"),
        )

        from safetensors.torch import load_file
        state = load_file(str(weights_path))
        missing, unexpected = model.unet.load_state_dict(state, strict=False)
        if len(missing) > 50:
            raise RuntimeError(
                f"Too many missing keys ({len(missing)}) loading DDBM community checkpoint"
            )
        model.eval()
        model.to(device)
        return model
