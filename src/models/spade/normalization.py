# Credits: SPADE from Park et al. "Semantic Image Synthesis with Spatially-Adaptive Normalization" CVPR 2019.
#
"""SPADE normalization layers."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SPADE(nn.Module):
    """Spatially-Adaptive (de)Normalization layer.

  Normalizes activations with instance (or batch) norm, then applies
  scale and bias predicted from a semantic segmentation map.
    """

    def __init__(
        self,
        norm_nc: int,
        label_nc: int,
        *,
        nhidden: int = 128,
        kernel_size: int = 3,
        param_free_norm_type: str = "instance",
    ) -> None:
        super().__init__()
        if param_free_norm_type == "instance":
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == "batch":
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError(f"Unsupported param-free norm type: {param_free_norm_type}")

        padding = kernel_size // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=kernel_size, padding=padding)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=kernel_size, padding=padding)

    def forward(self, x: torch.Tensor, segmap: torch.Tensor) -> torch.Tensor:
        normalized = self.param_free_norm(x)
        segmap = F.interpolate(segmap, size=x.shape[2:], mode="nearest")
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        return normalized * (1 + gamma) + beta
