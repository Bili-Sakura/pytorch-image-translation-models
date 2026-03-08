# Copyright (c) 2026 EarthBridge Team.
# Credits: OpenEarthMap-SAR CUT models. Architecture from Park et al. "Contrastive Learning for Unpaired Image-to-Image Translation" ECCV 2020.

"""Network architecture for OpenEarthMap-SAR CUT generator.

Includes:
* ``OpenEarthMapSARGenerator`` – CUT ResNet generator with anti-aliased down/upsampling.
* ``_Downsample`` / ``_Upsample`` – Anti-aliased sampling layers.
* ``_ResnetBlock`` – Residual block used by the generator.
"""

from __future__ import annotations

import functools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_filter(filt_size: int = 3) -> torch.Tensor:
    """Return a 2-D low-pass filter kernel."""
    _kernels = {
        1: np.array([1.0]),
        2: np.array([1.0, 1.0]),
        3: np.array([1.0, 2.0, 1.0]),
        4: np.array([1.0, 3.0, 3.0, 1.0]),
        5: np.array([1.0, 4.0, 6.0, 4.0, 1.0]),
        6: np.array([1.0, 5.0, 10.0, 10.0, 5.0, 1.0]),
        7: np.array([1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0]),
    }
    a = _kernels[filt_size]
    filt = torch.Tensor(a[:, None] * a[None, :])
    return filt / filt.sum()


def _get_pad_layer(pad_type: str):
    if pad_type in ("refl", "reflect"):
        return nn.ReflectionPad2d
    if pad_type in ("repl", "replicate"):
        return nn.ReplicationPad2d
    if pad_type == "zero":
        return nn.ZeroPad2d
    raise ValueError(f"Unsupported padding type: {pad_type}")


class _Downsample(nn.Module):
    """Anti-aliased downsampling layer."""

    def __init__(self, channels: int, pad_type: str = "reflect", filt_size: int = 3, stride: int = 2, pad_off: int = 0):
        super().__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [
            int(1.0 * (filt_size - 1) / 2) + pad_off,
            int(np.ceil(1.0 * (filt_size - 1) / 2)) + pad_off,
            int(1.0 * (filt_size - 1) / 2) + pad_off,
            int(np.ceil(1.0 * (filt_size - 1) / 2)) + pad_off,
        ]
        self.stride = stride
        self.channels = channels
        filt = _get_filter(filt_size=self.filt_size)
        self.register_buffer("filt", filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))
        self.pad = _get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        if self.filt_size == 1:
            if self.pad_off == 0:
                return inp[:, :, :: self.stride, :: self.stride]
            return self.pad(inp)[:, :, :: self.stride, :: self.stride]
        return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


class _Upsample(nn.Module):
    """Anti-aliased upsampling layer."""

    def __init__(self, channels: int, pad_type: str = "repl", filt_size: int = 4, stride: int = 2):
        super().__init__()
        self.filt_size = filt_size
        self.filt_odd = np.mod(filt_size, 2) == 1
        self.pad_size = int((filt_size - 1) / 2)
        self.stride = stride
        self.channels = channels
        filt = _get_filter(filt_size=self.filt_size) * (stride ** 2)
        self.register_buffer("filt", filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))
        self.pad = _get_pad_layer(pad_type)([1, 1, 1, 1])

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        ret = F.conv_transpose2d(
            self.pad(inp), self.filt, stride=self.stride, padding=1 + self.pad_size, groups=inp.shape[1]
        )[:, :, 1:, 1:]
        if self.filt_odd:
            return ret
        return ret[:, :, :-1, :-1]


class _ResnetBlock(nn.Module):
    """Single residual block with two 3×3 convolutions."""

    def __init__(
        self,
        dim: int,
        padding_type: str = "reflect",
        norm_layer: type[nn.Module] = nn.InstanceNorm2d,
        use_dropout: bool = False,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        p = 0
        if padding_type == "reflect":
            layers.append(nn.ReflectionPad2d(1))
        elif padding_type == "replicate":
            layers.append(nn.ReplicationPad2d(1))
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError(f"padding [{padding_type}] not implemented")
        layers += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        p = 0
        if padding_type == "reflect":
            layers.append(nn.ReflectionPad2d(1))
        elif padding_type == "replicate":
            layers.append(nn.ReplicationPad2d(1))
        elif padding_type == "zero":
            p = 1
        layers += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]
        self.conv_block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv_block(x)


class OpenEarthMapSARGenerator(nn.Module):
    """CUT ResNet generator compatible with OpenEarthMap-SAR checkpoints.

    Uses anti-aliased down/upsampling to match the original CUT training.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_filters: int = 64,
        n_blocks: int = 9,
        norm_layer: type[nn.Module] | None = None,
        use_dropout: bool = False,
        padding_type: str = "reflect",
        no_antialias: bool = False,
        no_antialias_up: bool = False,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, base_filters, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(base_filters),
            nn.ReLU(True),
        ]
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            if no_antialias:
                model += [
                    nn.Conv2d(base_filters * mult, base_filters * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                    norm_layer(base_filters * mult * 2),
                    nn.ReLU(True),
                ]
            else:
                model += [
                    nn.Conv2d(base_filters * mult, base_filters * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                    norm_layer(base_filters * mult * 2),
                    nn.ReLU(True),
                    _Downsample(base_filters * mult * 2),
                ]
        mult = 2 ** n_downsampling
        for _ in range(n_blocks):
            model.append(
                _ResnetBlock(base_filters * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
            )
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            if no_antialias_up:
                model += [
                    nn.ConvTranspose2d(base_filters * mult, base_filters * mult // 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
                    norm_layer(base_filters * mult // 2),
                    nn.ReLU(True),
                ]
            else:
                model += [
                    _Upsample(base_filters * mult),
                    nn.Conv2d(base_filters * mult, base_filters * mult // 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                    norm_layer(base_filters * mult // 2),
                    nn.ReLU(True),
                ]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(base_filters, out_channels, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
