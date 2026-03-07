# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Local Diffusion model components built on native PyTorch modules.

This module provides the networks required by the Local Diffusion method
from *Kim et al. "Tackling Structural Hallucination in Image Translation
with Local Diffusion" (ECCV 2024, Oral)*:

* :class:`LocalDiffusionUNet` – Conditional denoising U-Net that fuses
  an independent condition encoder branch at the bottleneck.  The encoder
  (``ConditionEncoder``) extracts multi-scale features from the conditioning
  image (e.g. a degraded or corrupted input), and those features are
  concatenated with the diffusion latent at the bottleneck for fusion.
* :class:`ConditionEncoder` – Residual encoder for the conditioning image.
  Uses GroupNorm-based ``BasicBlock`` layers and max-pooling to produce a
  compact spatial feature map.

The architecture supports three beta schedules (linear, cosine, sigmoid)
and three prediction objectives (``pred_x0``, ``pred_noise``, ``pred_v``).
"""

from __future__ import annotations

import math
from functools import partial
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from einops import rearrange
from einops.layers.torch import Rearrange


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _exists(x):
    return x is not None


def _default(val, d):
    if _exists(val):
        return val
    return d() if callable(d) else d


def _divisible_by(numer: int, denom: int) -> bool:
    return (numer % denom) == 0


def _cast_tuple(t, length: int = 1):
    return t if isinstance(t, tuple) else (t,) * length


# ---------------------------------------------------------------------------
# Upsample / Downsample
# ---------------------------------------------------------------------------

def _upsample(dim: int, dim_out: Optional[int] = None) -> nn.Sequential:
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, _default(dim_out, dim), 3, padding=1),
    )


def _downsample(dim: int, dim_out: Optional[int] = None) -> nn.Sequential:
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        nn.Conv2d(dim * 4, _default(dim_out, dim), 1),
    )


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class _RMSNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=1) * self.g * (x.shape[1] ** 0.5)


# ---------------------------------------------------------------------------
# Sinusoidal positional embedding
# ---------------------------------------------------------------------------

class _SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


# ---------------------------------------------------------------------------
# Building blocks (Block, ResnetBlock, LinearAttention, Attention)
# ---------------------------------------------------------------------------

class _Block(nn.Module):
    def __init__(self, dim: int, dim_out: int, groups: int = 8) -> None:
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, scale_shift: Optional[Tuple] = None) -> torch.Tensor:
        x = self.proj(x)
        x = self.norm(x)
        if _exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        return self.act(x)


class _ResnetBlock(nn.Module):
    """ResNet block with optional time-embedding injection (FiLM)."""

    def __init__(self, dim: int, dim_out: int, *, time_emb_dim: Optional[int] = None, groups: int = 8) -> None:
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if _exists(time_emb_dim)
            else None
        )
        self.block1 = _Block(dim, dim_out, groups=groups)
        self.block2 = _Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        scale_shift = None
        if _exists(self.mlp) and _exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class _LinearAttention(nn.Module):
    """Fast O(N) linear attention."""

    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32) -> None:
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.norm = _RMSNorm(dim)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), _RMSNorm(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = (rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads) for t in qkv)
        q = q.softmax(dim=-2) * self.scale
        k = k.softmax(dim=-1)
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class _Attention(nn.Module):
    """Standard scaled dot-product attention."""

    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32) -> None:
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.norm = _RMSNorm(dim)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = (rearrange(t, "b (h c) x y -> b h (x y) c", h=self.heads) for t in qkv)
        scale = q.shape[-1] ** -0.5
        sim = torch.einsum("b h i d, b h j d -> b h i j", q, k) * scale
        attn = sim.softmax(dim=-1)
        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


# ---------------------------------------------------------------------------
# ConditionEncoder (ResUnet-style feature encoder)
# ---------------------------------------------------------------------------

class _CondBasicBlock(nn.Module):
    """GroupNorm-based residual block for the condition encoder."""

    def __init__(self, in_ch: int, mid_ch: int, out_ch: int, stride: int = 1, residual: bool = True) -> None:
        super().__init__()
        self.residual = residual
        self.in_ch = in_ch
        self.out_ch = out_ch

        def _gn(ch: int) -> nn.GroupNorm:
            """GroupNorm with adaptive group count (max 16, must divide ch)."""
            groups = min(16, ch)
            while ch % groups != 0:
                groups -= 1
            return nn.GroupNorm(groups, ch)

        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, stride=stride, padding=1),
            _gn(mid_ch),
            nn.ReLU(),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            _gn(out_ch),
        )
        self.identity = None
        if residual and in_ch != out_ch:
            self.identity = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
                _gn(out_ch),
            )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.convblock(x)
        if self.residual:
            if self.identity is not None:
                identity = self.identity(x)
            out = out + identity
        return self.relu(out)


class ConditionEncoder(nn.Module):
    """Residual encoder for the conditioning image.

    Extracts multi-scale features and outputs a compact spatial feature
    map at the bottleneck resolution.  The architecture mirrors a ResUnet
    encoder with GroupNorm.

    The encoder downsamples using max-pooling.  The number of max-pool
    operations is ``len(filters) - 2`` (i.e. one fewer than the number
    of blocks).  Set *filters* to match the U-Net's depth so that the
    encoder output spatial size equals the U-Net bottleneck spatial size.

    Parameters
    ----------
    in_channels : int
        Number of input channels (e.g. 1 for grayscale, 3 for RGB).
    filters : tuple of int
        Channel widths at each encoder stage.  The number of downsampling
        operations equals ``len(filters) - 2``.
    """

    def __init__(
        self,
        in_channels: int = 1,
        filters: Tuple[int, ...] = (32, 32, 64, 128, 256),
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()
        ch_in = in_channels
        for ch_out in filters:
            self.blocks.append(_CondBasicBlock(ch_in, ch_out, ch_out))
            ch_in = ch_out
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, block in enumerate(self.blocks):
            x = block(x)
            # Pool between blocks (not after the last one)
            if i < len(self.blocks) - 1:
                x = self.maxpool(x)
        return x


# ---------------------------------------------------------------------------
# LocalDiffusionUNet – conditional denoising U-Net
# ---------------------------------------------------------------------------

class LocalDiffusionUNet(nn.Module):
    """Conditional denoising U-Net for Local Diffusion.

    The U-Net accepts a noisy image ``x_t`` and a timestep ``t``.  A separate
    :class:`ConditionEncoder` processes the conditioning image and its output
    is concatenated with the U-Net bottleneck features, followed by a fusion
    block.

    Parameters
    ----------
    dim : int
        Base channel dimension.
    channels : int
        Number of image channels (input and output).
    dim_mults : tuple of int
        Channel multipliers for each encoder/decoder level.
    resnet_block_groups : int
        Number of groups for GroupNorm in ResNet blocks.
    attn_dim_head : int
        Dimension per attention head.
    attn_heads : int
        Number of attention heads.
    full_attn : tuple of bool
        Per-level toggle for full (quadratic) vs. linear attention.
    cond_in_channels : int, optional
        Input channels for the conditioning encoder.  If ``None``, defaults
        to *channels*.
    cond_filters : tuple of int, optional
        Channel widths for the condition encoder.  If ``None``, automatically
        constructed to have the same number of pooling stages as the U-Net
        encoder, ensuring spatial size alignment at the bottleneck.
    init_type : str
        Weight initialisation method (``"normal"`` | ``"xavier"`` | ``"kaiming"``).
    init_gain : float
        Gain for weight initialisation.
    """

    def __init__(
        self,
        dim: int = 32,
        channels: int = 1,
        dim_mults: Tuple[int, ...] = (1, 2, 4, 8),
        resnet_block_groups: int = 8,
        attn_dim_head: int = 32,
        attn_heads: int = 4,
        full_attn: Tuple[bool, ...] = (False, False, False, True),
        cond_in_channels: Optional[int] = None,
        cond_filters: Optional[Tuple[int, ...]] = None,
        init_type: str = "normal",
        init_gain: float = 0.02,
    ) -> None:
        super().__init__()
        self.channels = channels
        cond_in_channels = cond_in_channels if cond_in_channels is not None else channels

        # Condition encoder
        # The U-Net spatially downsamples (len(dim_mults) - 1) times.
        # The cond encoder needs the same number of pooling ops, meaning
        # (n_pools + 1) blocks.  Its final output channels must equal
        # the U-Net bottleneck channels (dims[-1] = dim * dim_mults[-1]).
        if cond_filters is None:
            n_pools = len(dim_mults) - 1
            n_blocks = n_pools + 1
            # Build ascending channel widths up to mid_dim
            mid_dim_target = dim * dim_mults[-1]
            cond_filters_list = []
            for i in range(n_blocks):
                ch = max(dim, min(dim * (2 ** i), mid_dim_target))
                cond_filters_list.append(ch)
            cond_filters_list[-1] = mid_dim_target
            cond_filters = tuple(cond_filters_list)

        self.cond_model = ConditionEncoder(
            in_channels=cond_in_channels,
            filters=cond_filters,
        )

        # Initial projection
        init_dim = dim
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)

        # Channel dimensions per level
        dims = [init_dim, *(dim * m for m in dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(_ResnetBlock, groups=resnet_block_groups)

        # Time embedding
        time_dim = dim * 4
        sinu_pos_emb = _SinusoidalPosEmb(dim)
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Attention configs
        num_stages = len(dim_mults)
        full_attn = _cast_tuple(full_attn, num_stages)
        attn_heads = _cast_tuple(attn_heads, num_stages)
        attn_dim_head = _cast_tuple(attn_dim_head, num_stages)

        # Encoder
        self.downs = nn.ModuleList()
        for ind, ((dim_in, dim_out), layer_full_attn, layer_heads, layer_dim_head) in enumerate(
            zip(in_out, full_attn, attn_heads, attn_dim_head)
        ):
            is_last = ind >= (len(in_out) - 1)
            attn_klass = _Attention if layer_full_attn else _LinearAttention
            self.downs.append(
                nn.ModuleList([
                    block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                    block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                    attn_klass(dim_in, dim_head=layer_dim_head, heads=layer_heads),
                    _downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                ])
            )

        # Bottleneck
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = _Attention(mid_dim, heads=attn_heads[-1], dim_head=attn_dim_head[-1])
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        # Fusion block: concat UNet bottleneck + condition features → project back
        self.conv_fusion = block_klass(mid_dim * 2, mid_dim, time_emb_dim=time_dim)

        # Decoder
        self.ups = nn.ModuleList()
        for ind, ((dim_in, dim_out), layer_full_attn, layer_heads, layer_dim_head) in enumerate(
            zip(
                *map(reversed, (in_out, full_attn, attn_heads, attn_dim_head))
            )
        ):
            is_last = ind == (len(in_out) - 1)
            attn_klass = _Attention if layer_full_attn else _LinearAttention
            self.ups.append(
                nn.ModuleList([
                    block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                    block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                    attn_klass(dim_out, dim_head=layer_dim_head, heads=layer_heads),
                    _upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                ])
            )

        # Final projection
        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, channels, 1)

        _init_weights(self, init_type, init_gain)

    @property
    def downsample_factor(self) -> int:
        return 2 ** (len(self.downs) - 1)

    def forward(
        self,
        x: torch.Tensor,
        cond_img: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor (B, C, H, W)
            Noisy image at timestep ``t``.
        cond_img : Tensor (B, C_cond, H, W)
            Conditioning image (e.g. degraded / corrupted input).
        time : Tensor (B,)
            Integer or float timestep index.

        Returns
        -------
        Tensor (B, C, H, W)
            Model prediction (x_0, noise, or v depending on objective).
        """
        x = self.init_conv(x)
        r = x.clone()
        t = self.time_mlp(time)
        h = []

        # Encoder path
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)
            x = block2(x, t)
            x = attn(x) + x
            h.append(x)
            x = downsample(x)

        # Bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, t)

        # Condition fusion
        cond_feat = self.cond_model(cond_img.to(torch.float))
        x = torch.cat((x, cond_feat), dim=1)
        x = self.conv_fusion(x, t)

        # Decoder path
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x) + x
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        return self.final_conv(x)


# ---------------------------------------------------------------------------
# Weight initialisation
# ---------------------------------------------------------------------------

def _init_weights(net: nn.Module, init_type: str = "normal", init_gain: float = 0.02) -> None:
    """Initialise network weights."""

    def _init_func(m: nn.Module) -> None:
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=init_gain)
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("GroupNorm") != -1 or classname.find("BatchNorm") != -1:
            if hasattr(m, "weight") and m.weight is not None:
                init.normal_(m.weight.data, 1.0, init_gain)
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    net.apply(_init_func)


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def create_unet(
    dim: int = 32,
    channels: int = 1,
    dim_mults: Tuple[int, ...] = (1, 2, 4, 8),
    resnet_block_groups: int = 8,
    attn_dim_head: int = 32,
    attn_heads: int = 4,
    full_attn: Tuple[bool, ...] = (False, False, False, True),
    cond_in_channels: Optional[int] = None,
    cond_filters: Optional[Tuple[int, ...]] = None,
    init_type: str = "normal",
    init_gain: float = 0.02,
    **kwargs,
) -> LocalDiffusionUNet:
    """Factory for the Local Diffusion U-Net."""
    return LocalDiffusionUNet(
        dim=dim,
        channels=channels,
        dim_mults=dim_mults,
        resnet_block_groups=resnet_block_groups,
        attn_dim_head=attn_dim_head,
        attn_heads=attn_heads,
        full_attn=full_attn,
        cond_in_channels=cond_in_channels,
        cond_filters=cond_filters,
        init_type=init_type,
        init_gain=init_gain,
    )
