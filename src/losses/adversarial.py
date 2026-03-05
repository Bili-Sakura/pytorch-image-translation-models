"""GAN loss functions."""

from __future__ import annotations

import torch
import torch.nn as nn


class GANLoss(nn.Module):
    """Flexible GAN loss supporting multiple formulations.

    Parameters
    ----------
    mode:
        GAN loss variant.

        * ``"vanilla"`` – binary cross-entropy (original GAN).
        * ``"lsgan"``   – least-squares GAN.
        * ``"hinge"``   – hinge loss.
    real_label:
        Target label value for real images.
    fake_label:
        Target label value for fake images.
    """

    def __init__(
        self,
        mode: str = "vanilla",
        real_label: float = 1.0,
        fake_label: float = 0.0,
    ) -> None:
        super().__init__()
        self.register_buffer("real_label", torch.tensor(real_label))
        self.register_buffer("fake_label", torch.tensor(fake_label))
        self.mode = mode

        if mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif mode == "lsgan":
            self.loss = nn.MSELoss()
        elif mode == "hinge":
            self.loss = None
        else:
            raise ValueError(f"Unsupported GAN loss mode: {mode}")

    def _get_target_tensor(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        target_val = self.real_label if target_is_real else self.fake_label
        return target_val.expand_as(prediction)

    def forward(
        self,
        prediction: torch.Tensor,
        target_is_real: bool,
        for_discriminator: bool = True,
    ) -> torch.Tensor:
        """Compute the GAN loss.

        Parameters
        ----------
        prediction:
            Discriminator output tensor.
        target_is_real:
            Whether the ground truth label is real.
        for_discriminator:
            Whether this loss is used for discriminator training
            (only affects the hinge formulation).
        """
        if self.mode == "hinge":
            if for_discriminator:
                if target_is_real:
                    return torch.mean(torch.relu(1.0 - prediction))
                return torch.mean(torch.relu(1.0 + prediction))
            # Generator loss
            return -torch.mean(prediction)

        target_tensor = self._get_target_tensor(prediction, target_is_real)
        return self.loss(prediction, target_tensor)
