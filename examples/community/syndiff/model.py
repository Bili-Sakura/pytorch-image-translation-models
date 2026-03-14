# Copyright (c) 2026 EarthBridge Team.
# Credits: SynDiff (Özbey et al., IEEE TMI 2023) - https://github.com/icon-lab/SynDiff
# Consolidated from backbones/ + model utilities. No subfolder structure.

"""SynDiff model: NCSN++, Posterior_Coefficients, sampling. All-in-one model.py."""

from __future__ import annotations

import functools
import math
import string
from collections import abc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# op: Pure PyTorch upfirdn2d (from rosinality/stylegan2-pytorch)
# -----------------------------------------------------------------------------


def upfirdn2d_native(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1):
    _, channel, in_h, in_w = input.shape
    input = input.reshape(-1, in_h, in_w, 1)
    _, in_h, in_w, minor = input.shape
    kernel_h, kernel_w = kernel.shape
    out = input.view(-1, in_h, 1, in_w, 1, minor)
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)
    out = F.pad(out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)])
    out = out[:, max(-pad_y0, 0) : out.shape[1] - max(-pad_y1, 0), max(-pad_x0, 0) : out.shape[2] - max(-pad_x1, 0), :]
    out = out.permute(0, 3, 1, 2)
    out = out.reshape([-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1])
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(-1, minor, in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1, in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1)
    out = out.permute(0, 2, 3, 1)
    out = out[:, ::down_y, ::down_x, :]
    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h + down_y) // down_y
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w + down_x) // down_x
    return out.view(-1, channel, out_h, out_w)


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    if not isinstance(up, abc.Iterable):
        up = (up, up)
    if not isinstance(down, abc.Iterable):
        down = (down, down)
    if len(pad) == 2:
        pad = (pad[0], pad[1], pad[0], pad[1])
    return upfirdn2d_native(input, kernel, *up, *down, *pad)


# -----------------------------------------------------------------------------
# utils: Model registration
# -----------------------------------------------------------------------------

_MODELS = {}


def register_model(cls=None, *, name=None):
    def _register(cls):
        local_name = name if name is not None else cls.__name__
        if local_name in _MODELS:
            raise ValueError(f"Already registered model with name: {local_name}")
        _MODELS[local_name] = cls
        return cls
    if cls is None:
        return _register
    return _register(cls)


def get_model(name):
    return _MODELS[name]


# -----------------------------------------------------------------------------
# dense_layer
# -----------------------------------------------------------------------------

def _calculate_correct_fan(tensor, mode):
    from torch.nn.init import _calculate_fan_in_and_fan_out
    mode = mode.lower()
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == "fan_in" else fan_out


def kaiming_uniform_(tensor, gain=1.0, mode="fan_in"):
    fan = _calculate_correct_fan(tensor, mode)
    var = gain / max(1.0, fan)
    bound = math.sqrt(3.0 * var)
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


def variance_scaling_init_(tensor, scale):
    return kaiming_uniform_(tensor, gain=1e-10 if scale == 0 else scale, mode="fan_avg")


def _dense(in_channels, out_channels, init_scale=1.0):
    lin = nn.Linear(in_channels, out_channels)
    variance_scaling_init_(lin.weight, scale=init_scale)
    nn.init.zeros_(lin.bias)
    return lin


# -----------------------------------------------------------------------------
# up_or_down_sampling
# -----------------------------------------------------------------------------

def _setup_kernel(k):
    k = np.asarray(k, dtype=np.float32)
    if k.ndim == 1:
        k = np.outer(k, k)
    k /= np.sum(k)
    assert k.ndim == 2 and k.shape[0] == k.shape[1]
    return k


def naive_upsample_2d(x, factor=2):
    _N, C, H, W = x.shape
    x = torch.reshape(x, (-1, C, H, 1, W, 1))
    x = x.repeat(1, 1, 1, factor, 1, factor)
    return torch.reshape(x, (-1, C, H * factor, W * factor))


def naive_downsample_2d(x, factor=2):
    _N, C, H, W = x.shape
    x = torch.reshape(x, (-1, C, H // factor, factor, W // factor, factor))
    return torch.mean(x, dim=(3, 5))


def _shape(x, dim):
    return x.shape[dim]


def upsample_conv_2d(x, w, k=None, factor=2, gain=1):
    assert isinstance(factor, int) and factor >= 1 and len(w.shape) == 4
    convH = convW = w.shape[2]
    inC, outC = w.shape[1], w.shape[0]
    if k is None:
        k = [1] * factor
    k = _setup_kernel(k) * (gain * (factor**2))
    p = (k.shape[0] - factor) - (convW - 1)
    stride = [1, 1, factor, factor]
    output_shape = ((_shape(x, 2) - 1) * factor + convH, (_shape(x, 3) - 1) * factor + convW)
    output_padding = (output_shape[0] - (_shape(x, 2) - 1) * stride[0] - convH, output_shape[1] - (_shape(x, 3) - 1) * stride[1] - convW)
    num_groups = _shape(x, 1) // inC
    w = torch.reshape(w, (num_groups, -1, inC, convH, convW))
    w = w[..., ::-1, ::-1].permute(0, 2, 1, 3, 4)
    w = torch.reshape(w, (num_groups * inC, -1, convH, convW))
    x = F.conv_transpose2d(x, w, stride=stride, output_padding=output_padding, padding=0)
    return upfirdn2d(x, torch.tensor(k, device=x.device), pad=((p + 1) // 2 + factor - 1, p // 2 + 1))


def conv_downsample_2d(x, w, k=None, factor=2, gain=1):
    assert isinstance(factor, int) and factor >= 1
    _outC, _inC, convH, convW = w.shape
    if k is None:
        k = [1] * factor
    k = _setup_kernel(k) * gain
    p = (k.shape[0] - factor) + (convW - 1)
    x = upfirdn2d(x, torch.tensor(k, device=x.device), pad=((p + 1) // 2, p // 2))
    return F.conv2d(x, w, stride=[factor, factor], padding=0)


def upsample_2d(x, k=None, factor=2, gain=1):
    if k is None:
        k = [1] * factor
    k = _setup_kernel(k) * (gain * (factor**2))
    p = k.shape[0] - factor
    return upfirdn2d(x, torch.tensor(k, device=x.device), up=factor, pad=((p + 1) // 2 + factor - 1, p // 2))


def downsample_2d(x, k=None, factor=2, gain=1):
    if k is None:
        k = [1] * factor
    k = _setup_kernel(k) * gain
    p = k.shape[0] - factor
    return upfirdn2d(x, torch.tensor(k, device=x.device), down=factor, pad=((p + 1) // 2, p // 2))


class _Conv2dUpsampleDownsample(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, up=False, down=False, resample_kernel=(1, 3, 3, 1), use_bias=True, kernel_init=None):
        super().__init__()
        assert not (up and down) and kernel >= 1 and kernel % 2 == 1
        self.weight = nn.Parameter(torch.zeros(out_ch, in_ch, kernel, kernel))
        if kernel_init is not None:
            self.weight.data = kernel_init(self.weight.data.shape)
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(out_ch))
        self.up, self.down = up, down
        self.resample_kernel = resample_kernel
        self.kernel = kernel
        self.use_bias = use_bias

    def forward(self, x):
        if self.up:
            x = upsample_conv_2d(x, self.weight, k=self.resample_kernel)
        elif self.down:
            x = conv_downsample_2d(x, self.weight, k=self.resample_kernel)
        else:
            x = F.conv2d(x, self.weight, stride=1, padding=self.kernel // 2)
        if self.use_bias:
            x = x + self.bias.reshape(1, -1, 1, 1)
        return x


# -----------------------------------------------------------------------------
# layers
# -----------------------------------------------------------------------------

def variance_scaling(scale, mode, distribution, in_axis=1, out_axis=0, dtype=torch.float32, device="cpu"):
    def init(shape, dtype=dtype, device=device):
        receptive_field_size = np.prod(shape) / shape[in_axis] / shape[out_axis]
        fan_in = shape[in_axis] * receptive_field_size
        fan_out = shape[out_axis] * receptive_field_size
        denominator = (fan_in + fan_out) / 2 if mode == "fan_avg" else (fan_in if mode == "fan_in" else fan_out)
        variance = scale / denominator
        if distribution == "uniform":
            return (torch.rand(*shape, dtype=dtype, device=device) * 2.0 - 1.0) * np.sqrt(3 * variance)
        return torch.randn(*shape, dtype=dtype, device=device) * np.sqrt(variance)
    return init


def default_init(scale=1.0):
    scale = 1e-10 if scale == 0 else scale
    return variance_scaling(scale, "fan_avg", "uniform")


def ddpm_conv1x1(in_planes, out_planes, stride=1, bias=True, init_scale=1.0, padding=0):
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=padding, bias=bias)
    conv.weight.data = default_init(init_scale)(conv.weight.data.shape)
    nn.init.zeros_(conv.bias)
    return conv


def ddpm_conv3x3(in_planes, out_planes, stride=1, bias=True, dilation=1, init_scale=1.0, padding=1):
    conv = nn.Conv2d(in_planes, out_planes, stride=stride, padding=padding, dilation=dilation, bias=bias, kernel_size=3)
    conv.weight.data = default_init(init_scale)(conv.weight.data.shape)
    nn.init.zeros_(conv.bias)
    return conv


def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1), mode="constant")
    return emb


def _einsum(a, b, c, x, y):
    return torch.einsum("{},{}->{}".format("".join(a), "".join(b), "".join(c)), x, y)


def contract_inner(x, y):
    x_chars = list(string.ascii_lowercase[: len(x.shape)])
    y_chars = list(string.ascii_lowercase[len(x.shape) : len(y.shape) + len(x.shape)])
    y_chars[0] = x_chars[-1]
    out_chars = x_chars[:-1] + y_chars[1:]
    return _einsum(x_chars, y_chars, out_chars, x, y)


class NIN(nn.Module):
    def __init__(self, in_dim, num_units, init_scale=0.1):
        super().__init__()
        self.W = nn.Parameter(default_init(scale=init_scale)((in_dim, num_units)), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(num_units), requires_grad=True)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        y = contract_inner(x, self.W) + self.b
        return y.permute(0, 3, 1, 2)


# -----------------------------------------------------------------------------
# layerspp
# -----------------------------------------------------------------------------

conv1x1 = ddpm_conv1x1
conv3x3 = ddpm_conv3x3
dense = _dense


class AdaptiveGroupNorm(nn.Module):
    def __init__(self, num_groups, in_channel, style_dim):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, in_channel, affine=False, eps=1e-6)
        self.style = dense(style_dim, in_channel * 2)
        self.style.bias.data[:in_channel] = 1
        self.style.bias.data[in_channel:] = 0

    def forward(self, input, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)
        return gamma * self.norm(input) + beta


class GaussianFourierProjection(nn.Module):
    def __init__(self, embedding_size=256, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Combine(nn.Module):
    def __init__(self, dim1, dim2, method="cat"):
        super().__init__()
        self.Conv_0 = conv1x1(dim1, dim2)
        self.method = method

    def forward(self, x, y):
        h = self.Conv_0(x)
        if self.method == "cat":
            return torch.cat([h, y], dim=1)
        return h + y


class AttnBlockpp(nn.Module):
    def __init__(self, channels, skip_rescale=False, init_scale=0.0):
        super().__init__()
        self.GroupNorm_0 = nn.GroupNorm(num_groups=min(channels // 4, 32), num_channels=channels, eps=1e-6)
        self.NIN_0 = NIN(channels, channels)
        self.NIN_1 = NIN(channels, channels)
        self.NIN_2 = NIN(channels, channels)
        self.NIN_3 = NIN(channels, channels, init_scale=init_scale)
        self.skip_rescale = skip_rescale

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.GroupNorm_0(x)
        q, k, v = self.NIN_0(h), self.NIN_1(h), self.NIN_2(h)
        w = torch.einsum("bchw,bcij->bhwij", q, k) * (int(C) ** (-0.5))
        w = F.softmax(w.reshape(B, H, W, H * W), dim=-1).reshape(B, H, W, H, W)
        h = torch.einsum("bhwij,bcij->bchw", w, v)
        h = self.NIN_3(h)
        return (x + h) / np.sqrt(2.0) if self.skip_rescale else x + h


class Upsample(nn.Module):
    def __init__(self, in_ch=None, out_ch=None, with_conv=False, fir=False, fir_kernel=(1, 3, 3, 1)):
        super().__init__()
        out_ch = out_ch if out_ch else in_ch
        if not fir and with_conv:
            self.Conv_0 = conv3x3(in_ch, out_ch)
        elif fir and with_conv:
            self.Conv2d_0 = _Conv2dUpsampleDownsample(in_ch, out_ch, 3, up=True, resample_kernel=fir_kernel, use_bias=True, kernel_init=default_init())
        self.fir, self.with_conv = fir, with_conv
        self.fir_kernel = fir_kernel

    def forward(self, x):
        B, C, H, W = x.shape
        if not self.fir:
            h = F.interpolate(x, (H * 2, W * 2), "nearest")
            if self.with_conv:
                h = self.Conv_0(h)
        else:
            h = upsample_2d(x, self.fir_kernel, factor=2) if not self.with_conv else self.Conv2d_0(x)
        return h


class Downsample(nn.Module):
    def __init__(self, in_ch=None, out_ch=None, with_conv=False, fir=False, fir_kernel=(1, 3, 3, 1)):
        super().__init__()
        out_ch = out_ch if out_ch else in_ch
        if not fir and with_conv:
            self.Conv_0 = conv3x3(in_ch, out_ch, stride=2, padding=0)
        elif fir and with_conv:
            self.Conv2d_0 = _Conv2dUpsampleDownsample(in_ch, out_ch, 3, down=True, resample_kernel=fir_kernel, use_bias=True, kernel_init=default_init())
        self.fir, self.fir_kernel = fir, fir_kernel
        self.with_conv = with_conv

    def forward(self, x):
        if not self.fir:
            if self.with_conv:
                x = F.pad(x, (0, 1, 0, 1))
                x = self.Conv_0(x)
            else:
                x = F.avg_pool2d(x, 2, stride=2)
        else:
            x = downsample_2d(x, self.fir_kernel, factor=2) if not self.with_conv else self.Conv2d_0(x)
        return x


class ResnetBlockDDPMpp_Adagn(nn.Module):
    def __init__(self, act, in_ch, out_ch=None, temb_dim=None, zemb_dim=None, conv_shortcut=False, dropout=0.1, skip_rescale=False, init_scale=0.0):
        super().__init__()
        out_ch = out_ch if out_ch else in_ch
        self.GroupNorm_0 = AdaptiveGroupNorm(min(in_ch // 4, 32), in_ch, zemb_dim)
        self.Conv_0 = conv3x3(in_ch, out_ch)
        if temb_dim is not None:
            self.Dense_0 = nn.Linear(temb_dim, out_ch)
            self.Dense_0.weight.data = default_init()(self.Dense_0.weight.data.shape)
            nn.init.zeros_(self.Dense_0.bias)
        self.GroupNorm_1 = AdaptiveGroupNorm(min(out_ch // 4, 32), out_ch, zemb_dim)
        self.Dropout_0 = nn.Dropout(dropout)
        self.Conv_1 = conv3x3(out_ch, out_ch, init_scale=init_scale)
        if in_ch != out_ch:
            self.NIN_0 = NIN(in_ch, out_ch) if not conv_shortcut else None
            self.Conv_2 = conv3x3(in_ch, out_ch) if conv_shortcut else None
        self.skip_rescale, self.act, self.out_ch, self.conv_shortcut = skip_rescale, act, out_ch, conv_shortcut

    def forward(self, x, temb=None, zemb=None):
        h = self.act(self.GroupNorm_0(x, zemb))
        h = self.Conv_0(h)
        if temb is not None:
            h += self.Dense_0(self.act(temb))[:, :, None, None]
        h = self.act(self.GroupNorm_1(h, zemb))
        h = self.Dropout_0(h)
        h = self.Conv_1(h)
        if x.shape[1] != self.out_ch:
            x = self.Conv_2(x) if self.conv_shortcut else self.NIN_0(x)
        return (x + h) / np.sqrt(2.0) if self.skip_rescale else x + h


class ResnetBlockBigGANpp_Adagn(nn.Module):
    def __init__(self, act, in_ch, out_ch=None, temb_dim=None, zemb_dim=None, up=False, down=False, dropout=0.1, fir=False, fir_kernel=(1, 3, 3, 1), skip_rescale=True, init_scale=0.0):
        super().__init__()
        out_ch = out_ch if out_ch else in_ch
        self.GroupNorm_0 = AdaptiveGroupNorm(min(in_ch // 4, 32), in_ch, zemb_dim)
        self.up, self.down, self.fir, self.fir_kernel = up, down, fir, fir_kernel
        self.Conv_0 = conv3x3(in_ch, out_ch)
        if temb_dim is not None:
            self.Dense_0 = nn.Linear(temb_dim, out_ch)
            self.Dense_0.weight.data = default_init()(self.Dense_0.weight.shape)
            nn.init.zeros_(self.Dense_0.bias)
        self.GroupNorm_1 = AdaptiveGroupNorm(min(out_ch // 4, 32), out_ch, zemb_dim)
        self.Dropout_0 = nn.Dropout(dropout)
        self.Conv_1 = conv3x3(out_ch, out_ch, init_scale=init_scale)
        if in_ch != out_ch or up or down:
            self.Conv_2 = conv1x1(in_ch, out_ch)
        self.skip_rescale, self.act, self.in_ch, self.out_ch = skip_rescale, act, in_ch, out_ch

    def forward(self, x, temb=None, zemb=None):
        h = self.act(self.GroupNorm_0(x, zemb))
        if self.up:
            if self.fir:
                h = upsample_2d(h, self.fir_kernel, factor=2)
                x = upsample_2d(x, self.fir_kernel, factor=2)
            else:
                h, x = naive_upsample_2d(h, factor=2), naive_upsample_2d(x, factor=2)
        elif self.down:
            if self.fir:
                h = downsample_2d(h, self.fir_kernel, factor=2)
                x = downsample_2d(x, self.fir_kernel, factor=2)
            else:
                h, x = naive_downsample_2d(h, factor=2), naive_downsample_2d(x, factor=2)
        h = self.Conv_0(h)
        if temb is not None:
            h += self.Dense_0(self.act(temb))[:, :, None, None]
        h = self.act(self.GroupNorm_1(h, zemb))
        h = self.Dropout_0(h)
        h = self.Conv_1(h)
        if self.in_ch != self.out_ch or self.up or self.down:
            x = self.Conv_2(x)
        return (x + h) / np.sqrt(2.0) if self.skip_rescale else x + h


class ResnetBlockBigGANpp_Adagn_one(nn.Module):
    def __init__(self, act, in_ch, out_ch=None, temb_dim=None, zemb_dim=None, up=False, down=False, dropout=0.1, fir=False, fir_kernel=(1, 3, 3, 1), skip_rescale=True, init_scale=0.0):
        super().__init__()
        out_ch = out_ch if out_ch else in_ch
        self.GroupNorm_0 = AdaptiveGroupNorm(min(in_ch // 4, 32), in_ch, zemb_dim)
        self.up, self.down, self.fir, self.fir_kernel = up, down, fir, fir_kernel
        self.Conv_0 = conv3x3(in_ch, out_ch)
        if temb_dim is not None:
            self.Dense_0 = nn.Linear(temb_dim, out_ch)
            self.Dense_0.weight.data = default_init()(self.Dense_0.weight.shape)
            nn.init.zeros_(self.Dense_0.bias)
        self.GroupNorm_1 = nn.GroupNorm(num_groups=min(out_ch // 4, 32), num_channels=out_ch, eps=1e-6)
        self.Dropout_0 = nn.Dropout(dropout)
        self.Conv_1 = conv3x3(out_ch, out_ch, init_scale=init_scale)
        if in_ch != out_ch or up or down:
            self.Conv_2 = conv1x1(in_ch, out_ch)
        self.skip_rescale, self.act, self.in_ch, self.out_ch = skip_rescale, act, in_ch, out_ch

    def forward(self, x, temb=None, zemb=None):
        h = self.act(self.GroupNorm_0(x, zemb))
        if self.up:
            if self.fir:
                h, x = upsample_2d(h, self.fir_kernel, factor=2), upsample_2d(x, self.fir_kernel, factor=2)
            else:
                h, x = naive_upsample_2d(h, factor=2), naive_upsample_2d(x, factor=2)
        elif self.down:
            if self.fir:
                h, x = downsample_2d(h, self.fir_kernel, factor=2), downsample_2d(x, self.fir_kernel, factor=2)
            else:
                h, x = naive_downsample_2d(h, factor=2), naive_downsample_2d(x, factor=2)
        h = self.Conv_0(h)
        if temb is not None:
            h += self.Dense_0(self.act(temb))[:, :, None, None]
        h = self.act(self.GroupNorm_1(h))
        h = self.Dropout_0(h)
        h = self.Conv_1(h)
        if self.in_ch != self.out_ch or self.up or self.down:
            x = self.Conv_2(x)
        return (x + h) / np.sqrt(2.0) if self.skip_rescale else x + h


# -----------------------------------------------------------------------------
# NCSNpp (SynDiff generator)
# -----------------------------------------------------------------------------


class PixelNorm(nn.Module):
    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


@register_model(name="ncsnpp")
class NCSNpp(nn.Module):
    """NCSN++ model for SynDiff."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.not_use_tanh = config.not_use_tanh
        self.act = act = nn.SiLU()
        z_emb_dim = config.z_emb_dim
        self.nf = nf = config.num_channels_dae
        ch_mult = config.ch_mult
        self.num_res_blocks = num_res_blocks = config.num_res_blocks
        self.attn_resolutions = attn_resolutions = config.attn_resolutions
        dropout = config.dropout
        resamp_with_conv = config.resamp_with_conv
        self.num_resolutions = num_resolutions = len(ch_mult)
        all_resolutions = [config.image_size // (2 ** i) for i in range(num_resolutions)]
        self.conditional = config.conditional
        fir, fir_kernel = config.fir, config.fir_kernel
        self.skip_rescale = config.skip_rescale
        self.resblock_type = resblock_type = config.resblock_type.lower()
        self.progressive = progressive = config.progressive.lower()
        self.progressive_input = progressive_input = config.progressive_input.lower()
        self.embedding_type = embedding_type = config.embedding_type.lower()
        init_scale = 0.0
        combine_method = config.progressive_combine.lower()
        combiner = functools.partial(Combine, method=combine_method)

        ResnetBlockDDPM = functools.partial(ResnetBlockDDPMpp_Adagn, act=act, dropout=dropout, init_scale=init_scale, skip_rescale=self.skip_rescale, temb_dim=nf * 4, zemb_dim=z_emb_dim)
        ResnetBlockBigGAN = functools.partial(ResnetBlockBigGANpp_Adagn, act=act, dropout=dropout, fir=fir, fir_kernel=fir_kernel, init_scale=init_scale, skip_rescale=self.skip_rescale, temb_dim=nf * 4, zemb_dim=z_emb_dim)
        ResnetBlockBigGAN_one = functools.partial(ResnetBlockBigGANpp_Adagn_one, act=act, dropout=dropout, fir=fir, fir_kernel=fir_kernel, init_scale=init_scale, skip_rescale=self.skip_rescale, temb_dim=nf * 4, zemb_dim=z_emb_dim)

        if resblock_type == "ddpm":
            ResnetBlock = ResnetBlockDDPM
        elif resblock_type == "biggan":
            ResnetBlock = ResnetBlockBigGAN
        elif resblock_type == "biggan_oneadagn":
            ResnetBlock = ResnetBlockBigGAN_one
        else:
            raise ValueError(f"resblock type {resblock_type} unrecognized.")

        AttnBlock = functools.partial(AttnBlockpp, init_scale=init_scale, skip_rescale=self.skip_rescale)
        Upsample = functools.partial(Upsample, with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)
        Downsample = functools.partial(Downsample, with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

        if progressive == "output_skip":
            self.pyramid_upsample = Upsample(with_conv=False)
        elif progressive == "residual":
            pyramid_upsample = functools.partial(Upsample, with_conv=True)

        if progressive_input == "input_skip":
            self.pyramid_downsample = Downsample(with_conv=False)
        elif progressive_input == "residual":
            pyramid_downsample = functools.partial(Downsample, with_conv=True)

        channels = config.num_channels
        input_pyramid_ch = channels if progressive_input != "none" else None

        modules = []
        if embedding_type == "fourier":
            modules.append(GaussianFourierProjection(embedding_size=nf, scale=config.fourier_scale))
        if self.conditional:
            embed_dim = nf if embedding_type == "positional" else 2 * nf
            modules.append(nn.Linear(embed_dim, nf * 4))
            modules[-1].weight.data = default_init()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)
            modules.append(nn.Linear(nf * 4, nf * 4))
            modules[-1].weight.data = default_init()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)

        modules.append(conv3x3(channels, nf))
        hs_c = [nf]
        in_ch = nf

        for i_level in range(num_resolutions):
            for _ in range(num_res_blocks):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch
            if all_resolutions[i_level] in attn_resolutions:
                modules.append(AttnBlock(channels=in_ch))
                hs_c.append(in_ch)
            if i_level != num_resolutions - 1:
                if resblock_type == "ddpm":
                    modules.append(Downsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(down=True, in_ch=in_ch))
            if progressive_input == "input_skip":
                modules.append(combiner(dim1=input_pyramid_ch, dim2=in_ch))
            elif progressive_input == "residual":
                modules.append(pyramid_downsample(in_ch=input_pyramid_ch, out_ch=in_ch))
                input_pyramid_ch = in_ch
            hs_c.append(in_ch)

        in_ch = hs_c[-1]
        modules.append(ResnetBlock(in_ch=in_ch))
        modules.append(AttnBlock(channels=in_ch))
        modules.append(ResnetBlock(in_ch=in_ch))

        pyramid_ch = 0
        for i_level in reversed(range(num_resolutions)):
            for _ in range(num_res_blocks + 1):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
                in_ch = out_ch
            if all_resolutions[i_level] in attn_resolutions:
                modules.append(AttnBlock(channels=in_ch))
            if progressive != "none":
                if i_level == num_resolutions - 1:
                    modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6))
                    modules.append(conv3x3(in_ch, channels, init_scale=init_scale))
                    pyramid_ch = channels if progressive == "output_skip" else in_ch
                else:
                    if progressive == "residual":
                        modules.append(pyramid_upsample(in_ch=pyramid_ch, out_ch=in_ch))
                    pyramid_ch = in_ch
            else:
                if i_level != 0:
                    if resblock_type == "ddpm":
                        modules.append(Upsample(in_ch=in_ch))
                    else:
                        modules.append(ResnetBlock(in_ch=in_ch, up=True))

        if progressive != "output_skip":
            modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6))
            modules.append(conv3x3(in_ch, channels, init_scale=init_scale))

        self.all_modules = nn.ModuleList(modules)
        self.z_transform = nn.Sequential(
            PixelNorm(),
            dense(config.nz, z_emb_dim),
            self.act,
            *([m for _ in range(config.n_mlp) for m in [dense(z_emb_dim, z_emb_dim), self.act]])
        )

    def forward(self, x, time_cond, z):
        zemb = self.z_transform(z)
        modules, m_idx = self.all_modules, 0
        if self.embedding_type == "fourier":
            temb = modules[m_idx](torch.log(time_cond))
            m_idx += 1
        else:
            temb = get_timestep_embedding(time_cond, self.nf)
        if self.conditional:
            temb = modules[m_idx](temb)
            m_idx += 1
            temb = modules[m_idx](self.act(temb))
            m_idx += 1
        else:
            temb = None
        if not self.config.centered:
            x = 2 * x - 1.0
        input_pyramid = x if self.progressive_input != "none" else None
        hs = [modules[m_idx](x)]
        m_idx += 1

        for i_level in range(self.num_resolutions):
            for _ in range(self.num_res_blocks):
                h = modules[m_idx](hs[-1], temb, zemb)
                m_idx += 1
                if h.shape[-1] in self.attn_resolutions:
                    h = modules[m_idx](h)
                    m_idx += 1
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                if self.resblock_type == "ddpm":
                    h = modules[m_idx](hs[-1])
                    m_idx += 1
                else:
                    h = modules[m_idx](hs[-1], temb, zemb)
                    m_idx += 1
            if self.progressive_input == "input_skip":
                input_pyramid = self.pyramid_downsample(input_pyramid)
                h = modules[m_idx](input_pyramid, h)
                m_idx += 1
            elif self.progressive_input == "residual":
                input_pyramid = modules[m_idx](input_pyramid)
                m_idx += 1
                h = (input_pyramid + h) / np.sqrt(2.0) if self.skip_rescale else input_pyramid + h
            hs.append(h)

        h = modules[m_idx](hs[-1], temb, zemb)
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1
        h = modules[m_idx](h, temb, zemb)
        m_idx += 1
        pyramid = None

        for i_level in reversed(range(self.num_resolutions)):
            for _ in range(self.num_res_blocks + 1):
                h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb, zemb)
                m_idx += 1
            if h.shape[-1] in self.attn_resolutions:
                h = modules[m_idx](h)
                m_idx += 1
            if self.progressive != "none":
                if i_level == self.num_resolutions - 1:
                    pyramid = self.act(modules[m_idx](h))
                    m_idx += 1
                    pyramid = modules[m_idx](pyramid)
                    m_idx += 1
                else:
                    pyramid = self.pyramid_upsample(pyramid)
                    pyramid_h = self.act(modules[m_idx](h))
                    m_idx += 1
                    pyramid_h = modules[m_idx](pyramid_h)
                    m_idx += 1
                    pyramid = pyramid + pyramid_h
            else:
                if i_level != 0:
                    if self.resblock_type == "ddpm":
                        h = modules[m_idx](h)
                        m_idx += 1
                    else:
                        h = modules[m_idx](h, temb, zemb)
                        m_idx += 1

        if self.progressive == "output_skip":
            h = pyramid
        else:
            h = self.act(modules[m_idx](h))
            m_idx += 1
            h = modules[m_idx](h)
            m_idx += 1
        if not self.not_use_tanh:
            return torch.tanh(h)
        return h


# -----------------------------------------------------------------------------
# Posterior sampling (from original model.py)
# -----------------------------------------------------------------------------

def var_func_vp(t, beta_min, beta_max):
    log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
    return 1.0 - torch.exp(2.0 * log_mean_coeff)


def var_func_geometric(t, beta_min, beta_max):
    return beta_min * ((beta_max / beta_min) ** t)


def extract(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)


def get_time_schedule(args, device):
    n_timestep = args.num_timesteps
    eps_small = 1e-3
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t * (1.0 - eps_small) + eps_small)
    return t.to(device)


def get_sigma_schedule(args, device):
    n_timestep = args.num_timesteps
    beta_min, beta_max = args.beta_min, args.beta_max
    eps_small = 1e-3
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t * (1.0 - eps_small) + eps_small)
    var = var_func_geometric(t, beta_min, beta_max) if args.use_geometric else var_func_vp(t, beta_min, beta_max)
    alpha_bars = 1.0 - var
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
    first = torch.tensor(1e-8)
    betas = torch.cat((first[None], betas)).to(device).type(torch.float32)
    sigmas = betas**0.5
    a_s = torch.sqrt(1 - betas)
    return sigmas, a_s, betas


class Posterior_Coefficients:
    def __init__(self, args, device):
        _, _, self.betas = get_sigma_schedule(args, device=device)
        self.betas = self.betas.type(torch.float32)[1:]
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat(
            (torch.tensor([1.0], dtype=torch.float32, device=device), self.alphas_cumprod[:-1]), 0
        )
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))


def sample_posterior(coefficients, x_0, x_t, t):
    mean = extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0 + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
    log_var = extract(coefficients.posterior_log_variance_clipped, t, x_t.shape)
    nonzero_mask = 1 - (t == 0).type(torch.float32)
    return mean + nonzero_mask[:, None, None, None] * torch.exp(0.5 * log_var) * torch.randn_like(x_t)


def sample_from_model(coefficients, generator, n_time, x_init, T, opt):
    x = x_init[:, [0], :]
    source = x_init[:, [1], :]
    with torch.no_grad():
        for i in reversed(range(n_time)):
            t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)
            latent_z = torch.randn(x.size(0), opt.nz, device=x.device)
            x_0 = generator(torch.cat((x, source), axis=1), t, latent_z)
            x_new = sample_posterior(coefficients, x_0[:, [0], :], x, t)
            x = x_new.detach()
    return x
