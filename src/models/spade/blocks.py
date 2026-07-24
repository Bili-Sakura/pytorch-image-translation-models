# Credits: SPADE from Park et al. "Semantic Image Synthesis with Spatially-Adaptive Normalization" CVPR 2019.
#
"""SPADE residual blocks."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm

from src.models.spade.normalization import SPADE


class SPADEResnetBlock(nn.Module):
    """Residual block with SPADE normalization."""

    def __init__(
        self,
        fin: int,
        fout: int,
        *,
        label_nc: int,
        use_spectral_norm: bool = True,
        param_free_norm_type: str = "instance",
        spade_hidden: int = 128,
        spade_kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.learned_shortcut = fin != fout
        fmiddle = min(fin, fout)

        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        if use_spectral_norm:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        spade_kwargs = {
            "label_nc": label_nc,
            "nhidden": spade_hidden,
            "kernel_size": spade_kernel_size,
            "param_free_norm_type": param_free_norm_type,
        }
        self.norm_0 = SPADE(fin, **spade_kwargs)
        self.norm_1 = SPADE(fmiddle, **spade_kwargs)
        if self.learned_shortcut:
            self.norm_s = SPADE(fin, **spade_kwargs)

    def forward(self, x: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        x_s = self.shortcut(x, seg)
        dx = self.conv_0(self._actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self._actvn(self.norm_1(dx, seg)))
        return x_s + dx

    def shortcut(self, x: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        if self.learned_shortcut:
            return self.conv_s(self.norm_s(x, seg))
        return x

    @staticmethod
    def _actvn(x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(x, 0.2)
