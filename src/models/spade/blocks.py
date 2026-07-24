# Credits: SPADE (Park et al., CVPR 2019) adapted for this project style.
#
"""Core SPADE blocks."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SPADE(nn.Module):
    """Spatially-Adaptive Denormalization."""

    def __init__(self, norm_nc: int, label_nc: int, hidden_nc: int = 128) -> None:
        super().__init__()
        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, hidden_nc, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.mlp_gamma = nn.Conv2d(hidden_nc, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(hidden_nc, norm_nc, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, segmap: torch.Tensor) -> torch.Tensor:
        segmap = F.interpolate(segmap, size=x.shape[2:], mode="nearest")
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        normalized = self.param_free_norm(x)
        return normalized * (1 + gamma) + beta


class SPADEResnetBlock(nn.Module):
    """Residual block with SPADE conditioning."""

    def __init__(self, fin: int, fout: int, label_nc: int) -> None:
        super().__init__()
        self.learned_shortcut = fin != fout
        fmiddle = min(fin, fout)

        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        self.norm_0 = SPADE(fin, label_nc)
        self.norm_1 = SPADE(fmiddle, label_nc)

        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)
            self.norm_s = SPADE(fin, label_nc)

    @staticmethod
    def _actvn(x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(x, 2e-1)

    def _shortcut(self, x: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        if self.learned_shortcut:
            return self.conv_s(self.norm_s(x, seg))
        return x

    def forward(self, x: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        x_s = self._shortcut(x, seg)
        dx = self.conv_0(self._actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self._actvn(self.norm_1(dx, seg)))
        return x_s + dx
