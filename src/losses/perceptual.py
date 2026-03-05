"""Perceptual and feature-matching loss functions."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class PerceptualLoss(nn.Module):
    """VGG-based perceptual loss.

    Computes L1 distance between feature maps extracted from a
    pre-trained VGG-19 network at specified layers.

    Parameters
    ----------
    layer_weights:
        Mapping from VGG layer index to its weight in the total loss.
        Defaults to layers ``{2: 1.0, 7: 1.0, 12: 1.0, 21: 1.0, 30: 1.0}``
        which correspond to ``relu1_1, relu2_1, relu3_1, relu4_1, relu5_1``.
    normalize_input:
        If ``True``, expects inputs in [-1, 1] and re-scales to
        ImageNet range.
    """

    DEFAULT_LAYERS: dict[int, float] = {
        2: 1.0,   # relu1_1
        7: 1.0,   # relu2_1
        12: 1.0,  # relu3_1
        21: 1.0,  # relu4_1
        30: 1.0,  # relu5_1
    }

    def __init__(
        self,
        layer_weights: dict[int, float] | None = None,
        normalize_input: bool = True,
    ) -> None:
        super().__init__()
        self.normalize_input = normalize_input
        self.layer_weights = layer_weights or self.DEFAULT_LAYERS

        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        # Freeze VGG parameters
        for param in vgg.parameters():
            param.requires_grad = False

        max_layer = max(self.layer_weights.keys()) + 1
        self.slices = nn.ModuleList()
        prev = 0
        for idx in sorted(self.layer_weights.keys()):
            self.slices.append(nn.Sequential(*list(vgg.children())[prev : idx + 1]))
            prev = idx + 1

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize_input:
            x = (x + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        return (x - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = self._normalize(pred)
        target = self._normalize(target)

        loss = torch.tensor(0.0, device=pred.device)
        weights = sorted(self.layer_weights.values())

        x_pred = pred
        x_target = target
        for i, s in enumerate(self.slices):
            x_pred = s(x_pred)
            with torch.no_grad():
                x_target = s(x_target)
            loss = loss + weights[i] * torch.nn.functional.l1_loss(x_pred, x_target)

        return loss
