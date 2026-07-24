# Credits: SPADE from Park et al. "Semantic Image Synthesis with Spatially-Adaptive Normalization" CVPR 2019.
#
"""Feature matching and KL losses used by SPADE training."""

from __future__ import annotations

import torch
import torch.nn as nn


class FeatureMatchingLoss(nn.Module):
    """Match intermediate discriminator features between real and fake images."""

    def forward(
        self,
        fake_features: list[list[torch.Tensor]],
        real_features: list[list[torch.Tensor]],
    ) -> torch.Tensor:
        loss = fake_features[0][0].new_tensor(0.0)
        num_features = 0
        for fake_scale, real_scale in zip(fake_features, real_features):
            for fake_feat, real_feat in zip(fake_scale[:-1], real_scale[:-1]):
                loss = loss + torch.mean(torch.abs(fake_feat - real_feat.detach()))
                num_features += 1
        if num_features == 0:
            return loss
        return loss / num_features


class GaussianKLLoss(nn.Module):
    """KL divergence between the encoded style distribution and a unit Gaussian."""

    def forward(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)
