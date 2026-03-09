# Copyright (c) 2026 EarthBridge Team.
# Credits: xuekt98/BBDM (Li et al., CVPR 2023), OpenAI improved_diffusion.

"""Self-contained OpenAI-style UNet wrapper for original BBDM checkpoints."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["OpenAIBBDMUNet"]


def conv_nd(dims: int, *args, **kwargs) -> nn.Module:
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    if dims == 2:
        return nn.Conv2d(*args, **kwargs)
    raise ValueError(f"unsupported dims={dims}")


def linear(*args, **kwargs) -> nn.Module:
    return nn.Linear(*args, **kwargs)


def normalization(channels: int) -> nn.Module:
    return nn.GroupNorm(32, channels)


def zero_module(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000)
        * torch.arange(half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class TimestepBlock(nn.Module):
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(
        self, x: torch.Tensor, emb: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    def __init__(
        self, channels: int, use_conv: bool, dims: int = 2, out_channels: Optional[int] = None
    ) -> None:
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = conv_nd(dims, channels, self.out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(
        self, channels: int, use_conv: bool, dims: int = 2, out_channels: Optional[int] = None
    ) -> None:
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        if use_conv:
            self.op = conv_nd(dims, channels, self.out_channels, 3, stride=2, padding=1)
        else:
            self.op = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class ResBlock(TimestepBlock):
    def __init__(
        self,
        channels: int,
        emb_channels: int,
        dropout: float,
        *,
        out_channels: Optional[int] = None,
        use_scale_shift_norm: bool = False,
        up: bool = False,
        down: bool = False,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.out_channels = out_channels or channels
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(2, channels, self.out_channels, 3, padding=1),
        )
        self.updown = up or down
        if up:
            self.h_upd = Upsample(channels, False)
            self.x_upd = Upsample(channels, False)
        elif down:
            self.h_upd = Downsample(channels, False)
            self.x_upd = Downsample(channels, False)
        else:
            self.h_upd = nn.Identity()
            self.x_upd = nn.Identity()

        emb_out_ch = 2 * self.out_channels if use_scale_shift_norm else self.out_channels
        self.emb_layers = nn.Sequential(nn.SiLU(), linear(emb_channels, emb_out_ch))

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(2, self.out_channels, self.out_channels, 3, padding=1)),
        )
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = conv_nd(2, channels, self.out_channels, 1)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        emb_out = self.emb_layers(emb).type(h.dtype)
        while emb_out.dim() < h.dim():
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
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
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        return torch.einsum("bts,bcs->bct", weight, v)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 1) -> None:
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention()
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, *spatial = x.shape
        h = x.reshape(b, c, -1)
        h = self.norm(h)
        qkv = self.qkv(h)
        qkv = qkv.reshape(b * self.num_heads, (3 * c) // self.num_heads, -1)
        h = self.attention(qkv)
        h = h.reshape(b, c, -1)
        h = self.proj_out(h)
        return (x.reshape(b, c, -1) + h).reshape(b, c, *spatial)


class OpenAIBBDMUNetCore(nn.Module):
    """OpenAI/improved-diffusion-style UNetModel (self-contained)."""

    def __init__(
        self,
        *,
        image_size: int = 64,
        in_channels: int = 3,
        model_channels: int = 128,
        out_channels: int = 3,
        num_res_blocks: int = 2,
        attention_resolutions: tuple[int, ...] = (),
        dropout: float = 0.0,
        channel_mult: tuple[int, ...] = (1, 4, 8),
        conv_resample: bool = True,
        use_scale_shift_norm: bool = True,
        num_heads: int = 8,
        num_head_channels: int = 64,
        resblock_updown: bool = True,
        use_spatial_transformer: bool = False,
        condition_key: str = "nocond",
    ) -> None:
        super().__init__()
        _ = image_size
        if use_spatial_transformer:
            raise NotImplementedError("use_spatial_transformer=True is not supported here.")

        self.model_channels = model_channels
        self.condition_key = condition_key
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(2, in_channels, model_channels, 3, padding=1))]
        )
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers: list[nn.Module] = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    heads = num_heads if num_head_channels == -1 else max(ch // num_head_channels, 1)
                    layers.append(AttentionBlock(ch, num_heads=heads))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                down = (
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=out_ch,
                        use_scale_shift_norm=use_scale_shift_norm,
                        down=True,
                    )
                    if resblock_updown
                    else Downsample(ch, conv_resample, out_channels=out_ch)
                )
                self.input_blocks.append(TimestepEmbedSequential(down))
                input_block_chans.append(ch)
                ds *= 2

        mid_heads = num_heads if num_head_channels == -1 else max(ch // num_head_channels, 1)
        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, time_embed_dim, dropout, use_scale_shift_norm=use_scale_shift_norm),
            AttentionBlock(ch, num_heads=mid_heads),
            ResBlock(ch, time_embed_dim, dropout, use_scale_shift_norm=use_scale_shift_norm),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    heads = num_heads if num_head_channels == -1 else max(ch // num_head_channels, 1)
                    layers.append(AttentionBlock(ch, num_heads=heads))
                if level and i == num_res_blocks:
                    up = (
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=ch,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, out_channels=ch)
                    )
                    layers.append(up)
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(2, ch, out_channels, 3, padding=1)),
        )

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.condition_key != "nocond":
            if context is None:
                raise ValueError("context must be provided when condition_key != 'nocond'")
            x = torch.cat([x, context], dim=1)

        hs = []
        h = x
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
        return self.out(h)


class OpenAIBBDMUNet(nn.Module):
    """BBDM wrapper exposing ``(x_t, timesteps, context)`` interface."""

    def __init__(
        self,
        *,
        image_size: int = 64,
        in_channels: int = 3,
        model_channels: int = 128,
        out_channels: int = 3,
        num_res_blocks: int = 2,
        attention_resolutions: tuple[int, ...] = (),
        dropout: float = 0.0,
        channel_mult: tuple[int, ...] = (1, 4, 8),
        conv_resample: bool = True,
        use_scale_shift_norm: bool = True,
        condition_mode: str = "none",
        num_heads: int = 8,
        num_head_channels: int = 64,
        resblock_updown: bool = True,
        use_spatial_transformer: bool = False,
        condition_key: str = "nocond",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.condition_mode = condition_mode
        unet_in = in_channels * 2 if condition_mode == "concat" else in_channels
        self.unet = OpenAIBBDMUNetCore(
            image_size=image_size,
            in_channels=unet_in,
            model_channels=model_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            channel_mult=channel_mult,
            conv_resample=conv_resample,
            use_scale_shift_norm=use_scale_shift_norm,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            resblock_updown=resblock_updown,
            use_spatial_transformer=use_spatial_transformer,
            condition_key=condition_key,
        )

    def forward(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.condition_mode == "concat" and context is not None:
            x_t = torch.cat([x_t, context], dim=1)
        return self.unet(x_t, timesteps=timesteps, context=context)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        subfolder: str = "unet",
        device: str | torch.device = "cpu",
        **kwargs,
    ) -> "OpenAIBBDMUNet":
        path = Path(pretrained_model_name_or_path)
        unet_dir = path / subfolder
        config_path = unet_dir / "config.json"
        weights_path = unet_dir / "diffusion_pytorch_model.safetensors"
        if not config_path.exists() or not weights_path.exists():
            raise FileNotFoundError(
                f"BBDM community format requires {config_path} and {weights_path}"
            )

        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)

        def _t(v):
            return tuple(v) if isinstance(v, list) else v

        model = cls(
            image_size=int(config.get("image_size", 64)),
            in_channels=int(config.get("in_channels", 3)),
            model_channels=int(config.get("model_channels", 128)),
            out_channels=int(config.get("out_channels", 3)),
            num_res_blocks=int(config.get("num_res_blocks", 2)),
            attention_resolutions=_t(config.get("attention_resolutions", ())),
            dropout=float(config.get("dropout", 0.0)),
            channel_mult=_t(config.get("channel_mult", (1, 4, 8))),
            conv_resample=bool(config.get("conv_resample", True)),
            use_scale_shift_norm=bool(config.get("use_scale_shift_norm", True)),
            condition_mode=str(config.get("condition_mode", "none")),
            num_heads=int(config.get("num_heads", 8)),
            num_head_channels=int(config.get("num_head_channels", 64)),
            resblock_updown=bool(config.get("resblock_updown", True)),
            use_spatial_transformer=bool(config.get("use_spatial_transformer", False)),
            condition_key=str(config.get("condition_key", "nocond")),
        )

        from safetensors.torch import load_file

        state = load_file(str(weights_path))
        missing, unexpected = model.unet.load_state_dict(state, strict=False)
        if missing:
            raise RuntimeError(
                f"Missing keys ({len(missing)}) loading BBDM community checkpoint"
            )
        if unexpected:
            raise RuntimeError(
                f"Unexpected keys ({len(unexpected)}) loading BBDM community checkpoint"
            )

        model.eval().to(device)
        return model
