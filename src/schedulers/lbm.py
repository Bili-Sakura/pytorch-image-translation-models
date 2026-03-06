# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""LBM Scheduler – Latent Bridge Matching for fast image-to-image translation.

Implements the LBM flow-matching bridge process from the paper
`LBM: Latent Bridge Matching for Fast Image-to-Image Translation
<https://arxiv.org/abs/2503.07535>`_.

The forward process creates interpolants:

    ``x_t = sigma * x_source + (1 - sigma) * x_target + bridge_noise * sqrt(sigma * (1 - sigma)) * eps``

The model is trained to predict ``x_source - x_target`` (the flow direction),
and sampling uses Euler steps with optional bridge noise injection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import BaseOutput


@dataclass
class LBMSchedulerOutput(BaseOutput):
    """Output of the LBM scheduler's ``step`` function.

    Attributes
    ----------
    prev_sample : torch.Tensor
        Computed sample ``(x_{t-1})`` of previous timestep.
    pred_original_sample : torch.Tensor or None
        Predicted denoised sample ``(x_0)``.
    """

    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None


class LBMScheduler(SchedulerMixin, ConfigMixin):
    """Scheduler for Latent Bridge Matching (LBM).

    Parameters
    ----------
    num_train_timesteps : int
        Number of training timesteps.
    bridge_noise_sigma : float
        Bridge noise magnitude used during the interpolant construction and sampling.
    timestep_sampling : str
        How to sample timesteps during training (``"uniform"``, ``"log_normal"``,
        or ``"custom_timesteps"``).
    logit_mean : float
        Mean for ``log_normal`` timestep sampling.
    logit_std : float
        Standard deviation for ``log_normal`` timestep sampling.
    selected_timesteps : list of float, optional
        Timesteps used when ``timestep_sampling="custom_timesteps"``.
    prob : list of float, optional
        Probabilities for ``selected_timesteps``.
    """

    _compatibles: list[str] = []
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        bridge_noise_sigma: float = 0.001,
        timestep_sampling: str = "uniform",
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        selected_timesteps: Optional[List[float]] = None,
        prob: Optional[List[float]] = None,
    ) -> None:
        self.num_train_timesteps = num_train_timesteps
        self.bridge_noise_sigma = bridge_noise_sigma
        self.timestep_sampling = timestep_sampling
        self.logit_mean = logit_mean
        self.logit_std = logit_std
        self.selected_timesteps = selected_timesteps
        self.prob = prob

        sigmas = np.linspace(1.0, 0.0, num_train_timesteps + 1, dtype=np.float64)
        timesteps = np.arange(0, num_train_timesteps, dtype=np.int64)

        self.sigmas = torch.from_numpy(sigmas).float()
        self.timesteps = torch.from_numpy(timesteps).long()
        self.num_inference_steps: Optional[int] = None

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def sample_timesteps(
        self,
        n_samples: int,
        device: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        """Sample training timesteps according to the configured strategy."""
        if self.timestep_sampling == "uniform":
            idx = torch.randint(0, self.num_train_timesteps, (n_samples,), device="cpu")
            return self.timesteps[idx].to(device=device)

        if self.timestep_sampling == "log_normal":
            u = torch.normal(mean=self.logit_mean, std=self.logit_std, size=(n_samples,), device="cpu")
            u = torch.sigmoid(u)
            indices = (u * self.num_train_timesteps).long().clamp(0, self.num_train_timesteps - 1)
            return self.timesteps[indices].to(device=device)

        if self.timestep_sampling == "custom_timesteps":
            assert self.selected_timesteps is not None and self.prob is not None
            idx = np.random.choice(len(self.selected_timesteps), n_samples, p=self.prob)
            return torch.tensor([self.selected_timesteps[i] for i in idx], device=device, dtype=torch.long)

        raise ValueError(f"Unknown timestep_sampling: {self.timestep_sampling}")

    def get_sigmas(
        self,
        timesteps: torch.Tensor,
        n_dim: int = 4,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Look up sigma values for given timesteps."""
        sigmas = self.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = self.timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def add_noise(
        self,
        x_target: torch.Tensor,
        x_source: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Create an interpolant between source and target (forward process)."""
        sigmas = self.get_sigmas(timesteps, n_dim=x_target.ndim, device=x_target.device, dtype=x_target.dtype)
        return (
            sigmas * x_source
            + (1.0 - sigmas) * x_target
            + self.bridge_noise_sigma * (sigmas * (1.0 - sigmas)) ** 0.5 * torch.randn_like(x_target)
        )

    def compute_target(self, x_source: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
        """Compute the training target: ``x_source - x_target``."""
        return x_source - x_target

    def compute_pred_x0(
        self,
        noisy_sample: torch.Tensor,
        model_output: torch.Tensor,
        sigmas: torch.Tensor,
    ) -> torch.Tensor:
        """Recover predicted target: ``pred_x0 = noisy_sample - model_output * sigma``."""
        return noisy_sample - model_output * sigmas

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Union[str, torch.device] = None,
    ) -> None:
        """Prepare the scheduler for inference with ``num_inference_steps`` Euler steps."""
        self.num_inference_steps = num_inference_steps
        sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
        self.sigmas = torch.from_numpy(sigmas).float()
        if device is not None:
            self.sigmas = self.sigmas.to(device)
        self.timesteps = torch.linspace(0, self.num_train_timesteps - 1, num_inference_steps, device=device).long()

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[int, torch.Tensor],
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[LBMSchedulerOutput, Tuple]:
        """Perform a single Euler step: ``sample = sample - sigma * model_output``."""
        device = sample.device
        dtype = sample.dtype

        if isinstance(timestep, torch.Tensor):
            t = timestep.to(device)
        else:
            t = torch.tensor([timestep], device=device)
        if t.ndim == 0:
            t = t.unsqueeze(0)

        sigmas = self.get_sigmas(t, n_dim=sample.ndim, device=device, dtype=dtype)
        pred_x0 = self.compute_pred_x0(sample, model_output, sigmas)
        prev_sample = pred_x0

        if not return_dict:
            return (prev_sample, pred_x0)
        return LBMSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_x0)

    def scale_model_input(self, sample: torch.Tensor, timestep: Optional[int] = None) -> torch.Tensor:
        """No-op scaling for API compatibility."""
        return sample
