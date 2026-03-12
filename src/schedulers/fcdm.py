# Copyright (c) 2026 EarthBridge Team.
# Credits: FCDM (Kwon et al., CVPR 2026) - https://github.com/star-kwon/FCDM

"""FCDM scheduler for DDPM/DDIM sampling.

FCDM uses standard Gaussian diffusion with linear beta schedule and
noise-prediction objective, matching the original FCDM training setup.
"""

from __future__ import annotations

from typing import Optional

import torch

from src.schedulers.local_diffusion import (
    LocalDiffusionScheduler,
    LocalDiffusionSchedulerOutput,
)

__all__ = ["FCDMScheduler", "FCDMSchedulerOutput"]

# Re-export output type for API consistency
FCDMSchedulerOutput = LocalDiffusionSchedulerOutput


class FCDMScheduler(LocalDiffusionScheduler):
    """DDPM/DDIM scheduler for FCDM with linear schedule and pred_noise objective.

    Defaults match the original FCDM training: 1000 steps, linear beta schedule,
    epsilon (noise) prediction. For latent-space models (e.g. VAE), use
    clip_min/clip_max as needed for the latent range.

    Parameters
    ----------
    num_train_timesteps : int
        Total diffusion steps (default 1000, matching FCDM).
    beta_schedule : str
        ``"linear"`` (default) | ``"cosine"`` | ``"sigmoid"``
    objective : str
        ``"pred_noise"`` (default) | ``"pred_x0"`` | ``"pred_v"``
    ddim_sampling_eta : float
        DDIM stochasticity (0 = deterministic).
    use_ddpm : bool
        If True, use full DDPM steps; else DDIM with stride.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_schedule: str = "linear",
        objective: str = "pred_noise",
        ddim_sampling_eta: float = 0.0,
        min_snr_loss_weight: bool = False,
        min_snr_gamma: float = 5.0,
        use_ddpm: bool = False,
    ) -> None:
        super().__init__(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
            objective=objective,
            ddim_sampling_eta=ddim_sampling_eta,
            min_snr_loss_weight=min_snr_loss_weight,
            min_snr_gamma=min_snr_gamma,
        )
        self.use_ddpm = use_ddpm
        self.timesteps: Optional[torch.Tensor] = None

    def set_timesteps(self, num_inference_steps: Optional[int] = None) -> None:
        """Set the timestep sequence for sampling.

        For DDIM: use num_inference_steps evenly-spaced steps from T-1 down to 0.
        For DDPM: use all num_train_timesteps.
        """
        if self.use_ddpm:
            self.timesteps = torch.arange(
                self.num_train_timesteps - 1, -1, -1, dtype=torch.long
            )
        elif num_inference_steps is not None and num_inference_steps > 0:
            self.timesteps = torch.linspace(
                self.num_train_timesteps - 1,
                0,
                num_inference_steps,
                dtype=torch.float32,
            ).round().long()
        else:
            self.timesteps = torch.arange(
                self.num_train_timesteps - 1, -1, -1, dtype=torch.long
            )
