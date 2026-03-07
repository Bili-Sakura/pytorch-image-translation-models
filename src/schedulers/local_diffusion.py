# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Local Diffusion scheduler for Gaussian diffusion with DDPM / DDIM sampling.

Provides :class:`LocalDiffusionScheduler` which computes the forward process
noise schedule (linear, cosine, or sigmoid) and implements both DDPM and DDIM
reverse-process steps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from diffusers.utils import BaseOutput


@dataclass
class LocalDiffusionSchedulerOutput(BaseOutput):
    """Output of a single Local Diffusion scheduler step.

    Attributes
    ----------
    prev_sample : torch.Tensor
        Denoised sample for the previous timestep.
    pred_x_start : torch.Tensor
        Predicted clean image (x_0).
    """

    prev_sample: torch.Tensor
    pred_x_start: torch.Tensor


# ---------------------------------------------------------------------------
# Beta schedule factories
# ---------------------------------------------------------------------------

def _linear_beta_schedule(timesteps: int) -> torch.Tensor:
    """Linear beta schedule from the original DDPM paper."""
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def _cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine beta schedule (Nichol & Dhariwal, 2021)."""
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def _sigmoid_beta_schedule(timesteps: int, start: float = -3.0, end: float = 3.0, tau: float = 1.0) -> torch.Tensor:
    """Sigmoid beta schedule (Jabri et al., 2022). Better for images > 64×64."""
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class LocalDiffusionScheduler:
    """Gaussian diffusion scheduler for Local Diffusion.

    Supports DDPM (full-step) and DDIM (accelerated) reverse-process
    sampling, with three beta schedules (linear, cosine, sigmoid) and
    three prediction objectives (``pred_x0``, ``pred_noise``, ``pred_v``).

    Parameters
    ----------
    num_train_timesteps : int
        Total number of training diffusion steps.
    beta_schedule : str
        ``"linear"`` | ``"cosine"`` | ``"sigmoid"``
    objective : str
        ``"pred_x0"`` | ``"pred_noise"`` | ``"pred_v"``
    ddim_sampling_eta : float
        Stochasticity parameter for DDIM (0 = deterministic ODE).
    min_snr_loss_weight : bool
        Use min-SNR loss weighting (Hang et al., 2023).
    min_snr_gamma : float
        Clamp value for min-SNR weighting.
    """

    def __init__(
        self,
        num_train_timesteps: int = 250,
        beta_schedule: str = "sigmoid",
        objective: str = "pred_x0",
        ddim_sampling_eta: float = 0.0,
        min_snr_loss_weight: bool = False,
        min_snr_gamma: float = 5.0,
    ) -> None:
        self.num_train_timesteps = num_train_timesteps
        self.objective = objective
        self.ddim_sampling_eta = ddim_sampling_eta

        assert objective in {"pred_noise", "pred_x0", "pred_v"}, (
            f"objective must be pred_noise, pred_x0, or pred_v, got {objective}"
        )

        # Build beta schedule
        if beta_schedule == "linear":
            betas = _linear_beta_schedule(num_train_timesteps)
        elif beta_schedule == "cosine":
            betas = _cosine_beta_schedule(num_train_timesteps)
        elif beta_schedule == "sigmoid":
            betas = _sigmoid_beta_schedule(num_train_timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.betas = betas.float()
        self.alphas_cumprod = alphas_cumprod.float()
        self.alphas_cumprod_prev = alphas_cumprod_prev.float()

        # Precompute diffusion coefficients
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).float()
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).float()
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod).float()
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1).float()

        # Posterior coefficients q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_variance = posterior_variance.float()
        self.posterior_log_variance_clipped = torch.log(posterior_variance.clamp(min=1e-20)).float()
        self.posterior_mean_coef1 = (betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)).float()
        self.posterior_mean_coef2 = ((1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)).float()

        # Loss weight (SNR-based)
        snr = alphas_cumprod / (1 - alphas_cumprod)
        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max=min_snr_gamma)

        if objective == "pred_noise":
            self.loss_weight = (maybe_clipped_snr / snr).float()
        elif objective == "pred_x0":
            self.loss_weight = maybe_clipped_snr.float()
        elif objective == "pred_v":
            self.loss_weight = (maybe_clipped_snr / (snr + 1)).float()

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
        """Index into schedule tensor and reshape for broadcasting."""
        b = t.shape[0]
        out = a.to(t.device).gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

    # ------------------------------------------------------------------
    # Forward process
    # ------------------------------------------------------------------

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward diffusion: sample x_t given x_0 and timestep t."""
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------

    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Recover x_0 from x_t and predicted noise."""
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t: torch.Tensor, t: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        """Recover noise from x_t and predicted x_0."""
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0
        ) / self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def predict_start_from_v(self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Recover x_0 from x_t and predicted v (v-parameterisation)."""
        return (
            self._extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def model_output_to_x0_and_noise(
        self,
        model_output: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        clip_min: float = -1.0,
        clip_max: float = 1.0,
    ) -> tuple:
        """Convert model output to ``(x_0, pred_noise)`` based on objective."""
        if self.objective == "pred_noise":
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x_t, t, pred_noise)
            x_start = x_start.clamp(clip_min, clip_max)
        elif self.objective == "pred_x0":
            x_start = model_output.clamp(clip_min, clip_max)
            pred_noise = self.predict_noise_from_start(x_t, t, x_start)
        elif self.objective == "pred_v":
            x_start = self.predict_start_from_v(x_t, t, model_output)
            x_start = x_start.clamp(clip_min, clip_max)
            pred_noise = self.predict_noise_from_start(x_t, t, x_start)
        return x_start, pred_noise

    # ------------------------------------------------------------------
    # DDPM step
    # ------------------------------------------------------------------

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        clip_min: float = -1.0,
        clip_max: float = 1.0,
    ) -> LocalDiffusionSchedulerOutput:
        """Perform a single DDPM reverse-process step.

        Parameters
        ----------
        model_output : Tensor (B, C, H, W)
            Raw model prediction.
        timestep : Tensor (B,) or int
            Current timestep.
        sample : Tensor (B, C, H, W)
            Current noisy sample ``x_t``.
        clip_min, clip_max : float
            Clipping bounds for predicted x_0.

        Returns
        -------
        LocalDiffusionSchedulerOutput
        """
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor([timestep], device=sample.device).long()
        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0).expand(sample.shape[0])

        x_start, _ = self.model_output_to_x0_and_noise(model_output, sample, timestep, clip_min, clip_max)

        # Posterior q(x_{t-1} | x_t, x_0)
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, timestep, sample.shape) * x_start
            + self._extract(self.posterior_mean_coef2, timestep, sample.shape) * sample
        )
        posterior_variance = self._extract(self.posterior_variance, timestep, sample.shape)
        posterior_log_variance = self._extract(self.posterior_log_variance_clipped, timestep, sample.shape)

        noise = torch.randn_like(sample)
        # No noise at t=0
        nonzero_mask = (timestep != 0).float().reshape(sample.shape[0], *((1,) * (sample.ndim - 1)))
        prev_sample = posterior_mean + nonzero_mask * (0.5 * posterior_log_variance).exp() * noise

        return LocalDiffusionSchedulerOutput(prev_sample=prev_sample, pred_x_start=x_start)

    # ------------------------------------------------------------------
    # DDIM step
    # ------------------------------------------------------------------

    def ddim_step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        timestep_next: torch.Tensor,
        sample: torch.Tensor,
        clip_min: float = -1.0,
        clip_max: float = 1.0,
        eta: Optional[float] = None,
    ) -> LocalDiffusionSchedulerOutput:
        """Perform a single DDIM reverse-process step.

        Parameters
        ----------
        model_output : Tensor (B, C, H, W)
            Raw model prediction.
        timestep : Tensor (B,) or int
            Current timestep.
        timestep_next : Tensor (B,) or int
            Next (earlier) timestep.
        sample : Tensor (B, C, H, W)
            Current noisy sample.
        clip_min, clip_max : float
            Clipping bounds for predicted x_0.
        eta : float, optional
            DDIM stochasticity. Defaults to ``self.ddim_sampling_eta``.

        Returns
        -------
        LocalDiffusionSchedulerOutput
        """
        if eta is None:
            eta = self.ddim_sampling_eta

        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor([timestep], device=sample.device).long()
        if not isinstance(timestep_next, torch.Tensor):
            timestep_next = torch.tensor([timestep_next], device=sample.device).long()
        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0).expand(sample.shape[0])
        if timestep_next.dim() == 0:
            timestep_next = timestep_next.unsqueeze(0).expand(sample.shape[0])

        x_start, pred_noise = self.model_output_to_x0_and_noise(
            model_output, sample, timestep, clip_min, clip_max
        )

        alpha = self._extract(self.alphas_cumprod, timestep, sample.shape)
        # For timestep_next < 0 (final step), use alpha=1
        alpha_next = torch.where(
            timestep_next >= 0,
            self._extract(self.alphas_cumprod, timestep_next.clamp(min=0), sample.shape),
            torch.ones_like(alpha),
        )

        sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
        c = (1 - alpha_next - sigma ** 2).sqrt()

        noise = torch.randn_like(sample)
        prev_sample = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

        return LocalDiffusionSchedulerOutput(prev_sample=prev_sample, pred_x_start=x_start)

    # ------------------------------------------------------------------
    # Training loss
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        model_output: torch.Tensor,
        target: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Compute weighted MSE loss.

        Parameters
        ----------
        model_output : Tensor
            Raw model prediction.
        target : Tensor
            Target (x_0 for ``pred_x0``, noise for ``pred_noise``, v for ``pred_v``).
        t : Tensor (B,)
            Timestep indices.

        Returns
        -------
        Tensor
            Scalar training loss.
        """
        loss = F.mse_loss(model_output, target, reduction="none")
        loss = loss.mean(dim=list(range(1, loss.ndim)))  # reduce spatial dims
        loss = loss * self._extract(self.loss_weight, t, loss.shape).squeeze()
        return loss.mean()
