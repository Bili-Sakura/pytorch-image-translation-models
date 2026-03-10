"""Shared blocks for pix2pixHD generators."""

from __future__ import annotations

import functools

import torch
import torch.nn as nn


class _ResnetBlock(nn.Module):
    """A residual block used by pix2pixHD generators."""

    def __init__(
        self,
        dim: int,
        padding_type: str = "reflect",
        norm_layer: type[nn.Module] | functools.partial = nn.InstanceNorm2d,
        activation: nn.Module | None = None,
        use_dropout: bool = False,
    ) -> None:
        super().__init__()
        if activation is None:
            activation = nn.ReLU(True)
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        conv_block: list[nn.Module] = []
        p = 0
        if padding_type == "reflect":
            conv_block.append(nn.ReflectionPad2d(1))
        elif padding_type == "replicate":
            conv_block.append(nn.ReplicationPad2d(1))
        elif padding_type == "zero":
            p = 1
        else:
            raise ValueError(f"Unsupported padding type: {padding_type}")

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
            activation,
        ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == "reflect":
            conv_block.append(nn.ReflectionPad2d(1))
        elif padding_type == "replicate":
            conv_block.append(nn.ReplicationPad2d(1))
        elif padding_type == "zero":
            p = 1

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
        ]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv_block(x)

