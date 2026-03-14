# Copyright (c) 2026 EarthBridge Team. Credits: AlignFlow (ermongroup).

import torch
import torch.nn as nn
import torch.nn.functional as F

from .coupling_layer import CouplingLayer, MaskType
from ...util import squeeze_2x2


class RealNVP(nn.Module):
    """RealNVP density estimation model (Dinh et al.)."""

    def __init__(self, num_scales=2, in_channels=3, mid_channels=64, num_blocks=8,
                 un_normalize_x=False, no_latent=False):
        super().__init__()
        self.register_buffer("data_constraint", torch.tensor([0.9], dtype=torch.float32))
        self.un_normalize_x = un_normalize_x
        self.no_latent = no_latent
        self.flows = _RealNVP(0, num_scales, in_channels, mid_channels, num_blocks)

    def forward(self, x, reverse=False):
        sldj = None
        if self.no_latent or not reverse:
            x, sldj = self._pre_process(x)
        x, sldj = self.flows(x, sldj, reverse)
        return x, sldj

    def _pre_process(self, x):
        if self.un_normalize_x:
            x = x * 0.5 + 0.5
        if x.min() < 0 or x.max() > 1:
            raise ValueError("Expected x in [0, 1], got min/max {}/{}".format(x.min(), x.max()))
        x = (x * 255. + torch.rand_like(x)) / 256.
        y = (2 * x - 1) * self.data_constraint
        y = (y + 1) / 2
        y = y.log() - (1. - y).log()
        ldj = F.softplus(y) + F.softplus(-y) - F.softplus(
            (1. - self.data_constraint).log() - self.data_constraint.log()
        )
        ldj = ldj.view(ldj.size(0), -1).sum(-1)
        return y, ldj


class _RealNVP(nn.Module):
    """Recursive RealNVP block builder."""

    def __init__(self, scale_idx, num_scales, in_channels, mid_channels, num_blocks):
        super().__init__()
        self.is_last_block = scale_idx == num_scales - 1
        self.in_couplings = nn.ModuleList([
            CouplingLayer(in_channels, mid_channels, num_blocks, MaskType.CHECKERBOARD, False),
            CouplingLayer(in_channels, mid_channels, num_blocks, MaskType.CHECKERBOARD, True),
            CouplingLayer(in_channels, mid_channels, num_blocks, MaskType.CHECKERBOARD, False),
        ])
        if self.is_last_block:
            self.in_couplings.append(
                CouplingLayer(in_channels, mid_channels, num_blocks, MaskType.CHECKERBOARD, True)
            )
        else:
            self.out_couplings = nn.ModuleList([
                CouplingLayer(4 * in_channels, 2 * mid_channels, num_blocks, MaskType.CHANNEL_WISE, False),
                CouplingLayer(4 * in_channels, 2 * mid_channels, num_blocks, MaskType.CHANNEL_WISE, True),
                CouplingLayer(4 * in_channels, 2 * mid_channels, num_blocks, MaskType.CHANNEL_WISE, False),
            ])
            self.next_block = _RealNVP(scale_idx + 1, num_scales, 2 * in_channels, 2 * mid_channels, num_blocks)

    def forward(self, x, sldj, reverse=False):
        if reverse:
            if not self.is_last_block:
                x = squeeze_2x2(x, reverse=False, alt_order=True)
                x, x_split = x.chunk(2, dim=1)
                x, sldj = self.next_block(x, sldj, reverse)
                x = torch.cat((x, x_split), dim=1)
                x = squeeze_2x2(x, reverse=True, alt_order=True)
                x = squeeze_2x2(x, reverse=False)
                for coupling in reversed(self.out_couplings):
                    x, sldj = coupling(x, sldj, reverse)
                x = squeeze_2x2(x, reverse=True)
            for coupling in reversed(self.in_couplings):
                x, sldj = coupling(x, sldj, reverse)
        else:
            for coupling in self.in_couplings:
                x, sldj = coupling(x, sldj, reverse)
            if not self.is_last_block:
                x = squeeze_2x2(x, reverse=False)
                for coupling in self.out_couplings:
                    x, sldj = coupling(x, sldj, reverse)
                x = squeeze_2x2(x, reverse=True)
                x = squeeze_2x2(x, reverse=False, alt_order=True)
                x, x_split = x.chunk(2, dim=1)
                x, sldj = self.next_block(x, sldj, reverse)
                x = torch.cat((x, x_split), dim=1)
                x = squeeze_2x2(x, reverse=True, alt_order=True)
        return x, sldj
