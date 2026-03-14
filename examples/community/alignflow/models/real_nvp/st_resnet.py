# Copyright (c) 2026 EarthBridge Team. Credits: AlignFlow (ermongroup).

import torch.nn as nn
import torch.nn.functional as F

from ...util import WNConv2d


class ResidualBlock(nn.Module):
    """ResNet basic block with weight norm."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_norm = nn.BatchNorm2d(in_channels)
        self.in_conv = WNConv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.out_norm = nn.BatchNorm2d(out_channels)
        self.out_conv = WNConv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        skip = x
        x = self.in_norm(x)
        x = F.relu(x)
        x = self.in_conv(x)
        x = self.out_norm(x)
        x = F.relu(x)
        x = self.out_conv(x)
        return x + skip


class STResNet(nn.Module):
    """ResNet for scale and translate factors in Real NVP."""

    def __init__(self, in_channels, mid_channels, out_channels, num_blocks, kernel_size, padding):
        super().__init__()
        self.in_conv = WNConv2d(in_channels, mid_channels, kernel_size, padding, bias=True)
        self.in_skip = WNConv2d(mid_channels, mid_channels, kernel_size=1, padding=0, bias=True)
        self.blocks = nn.ModuleList([
            ResidualBlock(mid_channels, mid_channels) for _ in range(num_blocks)
        ])
        self.skips = nn.ModuleList([
            WNConv2d(mid_channels, mid_channels, kernel_size=1, padding=0, bias=True)
            for _ in range(num_blocks)
        ])
        self.out_norm = nn.BatchNorm2d(mid_channels)
        self.out_conv = WNConv2d(mid_channels, out_channels, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        x = self.in_conv(x)
        x_skip = self.in_skip(x)
        for block, skip in zip(self.blocks, self.skips):
            x = block(x)
            x_skip += skip(x)
        x = self.out_norm(x_skip)
        x = F.relu(x)
        x = self.out_conv(x)
        return x
