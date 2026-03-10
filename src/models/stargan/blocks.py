"""Reusable blocks for StarGAN models."""

from __future__ import annotations

import torch
import torch.nn as nn


class StarGANResidualBlock(nn.Module):
    """Residual block used in StarGAN generator bottleneck."""

    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.main(x)

