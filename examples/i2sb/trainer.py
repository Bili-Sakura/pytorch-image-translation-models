# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""I2SB Trainer – training loop for Image-to-Image Schrödinger Bridge."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from examples.i2sb.config import TaskConfig
from src.models.unet.i2sb_unet import I2SBUNet
from src.models.unet.unet_2d import create_model
from src.schedulers.i2sb import I2SBScheduler

logger = logging.getLogger(__name__)


class I2SBTrainer:
    """Trainer for I2SB image-translation models.

    Parameters
    ----------
    cfg : TaskConfig
        Task-level hyper-parameters.
    """

    def __init__(self, cfg: TaskConfig) -> None:
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    def build_model(self) -> I2SBUNet:
        """Instantiate the UNet from the current config."""
        return create_model(
            image_size=self.cfg.resolution,
            in_channels=self.cfg.in_channels,
            num_channels=self.cfg.num_channels,
            num_res_blocks=self.cfg.num_res_blocks,
            attention_resolutions=self.cfg.attention_resolutions,
            condition_mode=self.cfg.condition_mode,
        )

    def build_scheduler(self) -> I2SBScheduler:
        """Instantiate the noise scheduler from the current config."""
        return I2SBScheduler(
            interval=self.cfg.interval,
            beta_max=self.cfg.beta_max,
        )

    # ------------------------------------------------------------------
    # Training loss
    # ------------------------------------------------------------------

    @staticmethod
    def compute_training_loss(
        model: nn.Module,
        scheduler: I2SBScheduler,
        x0: torch.Tensor,
        x_T: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the I2SB denoising loss for a single batch.

        Parameters
        ----------
        model : nn.Module
            The UNet (or any noise-prediction network).
        scheduler : I2SBScheduler
            Noise scheduler.
        x0 : Tensor ``[B, C, H, W]``
            Source (clean) images.
        x_T : Tensor ``[B, C, H, W]``
            Target images (used as ``x_1`` in the bridge).

        Returns
        -------
        Tensor
            Scalar MSE loss with gradient attached.
        """
        batch_size = x0.shape[0]

        # Random timesteps
        t = torch.randint(0, scheduler.interval, (batch_size,))

        # Forward bridge sample
        xt = scheduler.q_sample(t, x0, x_T)

        # Training label (noise prediction target)
        label = scheduler.compute_label(t, x0, xt)

        # Model prediction
        t_float = t.float().to(x0.device)
        cond_mode = getattr(model, "condition_mode", None)
        if cond_mode == "concat":
            pred = model(xt, t_float, cond=x_T)
        else:
            pred = model(xt, t_float)

        return torch.nn.functional.mse_loss(pred, label)
