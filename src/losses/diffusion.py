# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.
#
# Loss designs borrowed from: SD/DDPM, EDM (Karras et al.), EDM2 (NVlabs), SiD2 (Hoogeboom et al.).

"""Unified diffusion loss for SD, EDM, EDM2, and SiD2 training.

Supports ε-prediction, x0-prediction, v-prediction, EDM preconditioning,
and SiD2 sigmoid weighting (discrete and continuous log-SNR schedules).
"""

from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DiffusionLoss", "cosine_interpolated_logsnr", "get_diffusion_loss"]


# ---------------------------------------------------------------------------
# Cosine log-SNR schedule (SiD2 Appendix B) – for continuous SiD2 loss weighting
# ---------------------------------------------------------------------------


def cosine_interpolated_logsnr(
    t: torch.Tensor,
    image_res: int = 512,
    logsnr_min: float = -10.0,
    logsnr_max: float = 10.0,
) -> torch.Tensor:
    """Exact SiD2 continuous log-SNR schedule (Appendix B)."""
    log_change_high = math.log(image_res) - math.log(512)
    log_change_low = math.log(image_res) - math.log(32)
    b = math.atan(math.exp(-0.5 * logsnr_max))
    a = math.atan(math.exp(-0.5 * logsnr_min)) - b
    logsnr_cosine = -2.0 * torch.log(torch.tan(a * t + b))
    logsnr_high = logsnr_cosine + log_change_high
    logsnr_low = logsnr_cosine + log_change_low
    return (1 - t) * logsnr_high + t * logsnr_low


def _cosine_dlogsnr_dt(
    t: torch.Tensor,
    image_res: int,
    logsnr_min: float,
    logsnr_max: float,
) -> torch.Tensor:
    """Analytical derivative of cosine_interpolated_logsnr w.r.t. t."""
    log_change_high = math.log(image_res) - math.log(512)
    log_change_low = math.log(image_res) - math.log(32)
    b = math.atan(math.exp(-0.5 * logsnr_max))
    a = math.atan(math.exp(-0.5 * logsnr_min)) - b
    x = a * t + b
    # d/dx log(tan(x)) = sec²(x)/tan(x) = 2/sin(2x); d(logsnr_cosine)/dt = -2 * a * 2/sin(2x)
    d_logsnr_cosine = -4 * a / torch.sin(2 * x)
    d_logsnr = d_logsnr_cosine + (log_change_low - log_change_high)
    return d_logsnr


# ---------------------------------------------------------------------------
# Diffusion Loss (unified)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Objective name mapping (DDPM/LocalDiffusion style -> DiffusionLoss)
# ---------------------------------------------------------------------------

_OBJECTIVE_MAP = {
    "pred_noise": "epsilon",
    "epsilon": "epsilon",
    "pred_x0": "sample",
    "sample": "sample",
    "pred_v": "v_prediction",
    "v_prediction": "v_prediction",
}


def get_diffusion_loss(
    loss_type: str = "mse",
    prediction_type: str = "sample",
    **kwargs,
) -> "DiffusionLoss":
    """One-line factory for diffusion loss used across local_diffusion, i2sb, etc.

    Parameters
    ----------
    loss_type : str
        ``"mse"`` (plain), ``"min_snr"`` (SNR weighting, LocalDiffusion default),
        ``"sid2"`` (SiD2 sigmoid), ``"edm"`` (EDM preconditioning).
    prediction_type : str
        ``"epsilon"``, ``"sample"`` (x0), ``"v_prediction"``, or scheduler-style
        ``"pred_noise"``, ``"pred_x0"``, ``"pred_v"``.
    **kwargs
        Passed to DiffusionLoss (sid2_bias, sigma_data, loss_norm, etc.).

    Returns
    -------
    DiffusionLoss
        Configured loss module.
    """
    pred = _OBJECTIVE_MAP.get(prediction_type, prediction_type)
    return DiffusionLoss(loss_type=loss_type, prediction_type=pred, **kwargs)


# ---------------------------------------------------------------------------
# Diffusion Loss (unified)
# ---------------------------------------------------------------------------


class DiffusionLoss(nn.Module):
    """Unified diffusion loss for SD, EDM, EDM2, SiD2, I2SB, and Local Diffusion.

    Parameters
    ----------
    prediction_type : str
        Model prediction target: ``"epsilon"``, ``"sample"`` (x0), or ``"v_prediction"``.
    loss_type : str
        Loss weighting: ``"mse"`` (plain), ``"min_snr"`` (SNR weighting),
        ``"edm"`` (Karras/EDM), ``"sid2"`` (SiD2 sigmoid).
    sid2_bias : float
        Sigmoid bias for SiD2 weighting (default -3.0).
    sigma_data : float
        EDM data scale (default 0.5).
    loss_norm : str
        Pixel loss: ``"mse"`` or ``"l1"``.
    """

    def __init__(
        self,
        prediction_type: str = "v_prediction",
        loss_type: str = "mse",
        sid2_bias: float = -3.0,
        sigma_data: float = 0.5,
        loss_norm: str = "mse",
    ) -> None:
        super().__init__()
        self.prediction_type = prediction_type
        self.loss_type = loss_type
        self.sid2_bias = sid2_bias
        self.sigma_data = sigma_data
        self.loss_norm = loss_norm
        self.register_buffer("_sid2_exp_bias", torch.tensor(sid2_bias).exp())

    def forward(
        self,
        model_output: torch.Tensor,
        clean_images: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        timesteps_or_sigma: Optional[torch.Tensor] = None,
        scheduler: Optional[Any] = None,
        *,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute diffusion training loss.

        Parameters
        ----------
        model_output : Tensor
            Raw model prediction.
        clean_images : Tensor, optional
            Ground-truth clean samples (x0). Not needed when target is provided.
        noise : Tensor, optional
            Noise used in forward diffusion (needed for epsilon/v_prediction).
        timesteps_or_sigma : Tensor, optional
            Discrete timestep indices (B,) or continuous sigma values (B,).
        scheduler : object, optional
            Scheduler with alphas_cumprod, get_velocity, or cosine_interpolated_logsnr.
        target : Tensor, optional
            Precomputed target (e.g. I2SB label). When set, clean_images/noise ignored.
        """
        # Allow precomputed target (I2SB, bridge methods)
        if target is not None:
            return self._compute_weighted(
                model_output, target, timesteps_or_sigma, scheduler
            )

        if clean_images is None or timesteps_or_sigma is None:
            raise ValueError("clean_images and timesteps_or_sigma required when target is None")

        if self.loss_type == "edm":
            return self._edm_loss(model_output, clean_images, timesteps_or_sigma)

        if self.loss_type == "sid2":
            return self._sid2_loss(
                model_output, clean_images, noise, timesteps_or_sigma, scheduler
            )

        # mse or min_snr
        target = self._get_target(clean_images, noise, timesteps_or_sigma, scheduler)
        return self._compute_weighted(
            model_output, target, timesteps_or_sigma, scheduler
        )

    def _compute_weighted(
        self,
        model_output: torch.Tensor,
        target: torch.Tensor,
        timesteps: Optional[torch.Tensor],
        scheduler: Optional[Any],
    ) -> torch.Tensor:
        """Compute loss with optional min_snr weighting."""
        if self.loss_norm == "l1":
            loss = F.l1_loss(model_output, target, reduction="none")
        else:
            loss = F.mse_loss(model_output, target, reduction="none")

        if self.loss_type == "min_snr" and scheduler is not None and hasattr(scheduler, "loss_weight"):
            t = timesteps if timesteps is not None else torch.zeros(model_output.shape[0], device=model_output.device, dtype=torch.long)
            w = scheduler._extract(scheduler.loss_weight, t, loss.shape)
            loss = loss.mean(dim=list(range(1, loss.ndim))) * w.squeeze()
            return loss.mean()
        return loss.mean()

    def _get_target(
        self,
        clean: torch.Tensor,
        noise: Optional[torch.Tensor],
        timesteps: torch.Tensor,
        scheduler: Optional[Any],
    ) -> torch.Tensor:
        if self.prediction_type == "epsilon":
            if noise is None:
                raise ValueError("noise required for epsilon prediction")
            return noise

        if self.prediction_type == "sample":
            return clean

        if self.prediction_type == "v_prediction":
            if noise is None:
                raise ValueError("noise required for v_prediction")
            return self._get_velocity(clean, noise, timesteps, scheduler)

        raise ValueError(
            f"prediction_type must be epsilon, sample, or v_prediction, got {self.prediction_type!r}"
        )

    def _get_velocity(
        self,
        clean: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        scheduler: Optional[Any],
    ) -> torch.Tensor:
        if scheduler is not None and hasattr(scheduler, "get_velocity"):
            return scheduler.get_velocity(clean, noise, timesteps)

        # Fallback: schedulers with alphas_cumprod (e.g. LocalDiffusionScheduler)
        if scheduler is None or not hasattr(scheduler, "alphas_cumprod"):
            raise ValueError(
                "scheduler with get_velocity or alphas_cumprod required for v_prediction"
            )
        alpha_bar = self._extract(scheduler.alphas_cumprod, timesteps, clean.shape)
        sqrt_alpha = self._extract(
            scheduler.sqrt_alphas_cumprod, timesteps, clean.shape
        )
        sqrt_one_minus_alpha = self._extract(
            scheduler.sqrt_one_minus_alphas_cumprod, timesteps, clean.shape
        )
        return sqrt_alpha * noise - sqrt_one_minus_alpha * clean

    def _edm_loss(
        self,
        model_output: torch.Tensor,
        clean: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """EDM loss: denoised = c_skip * clean + c_out * model_output, weight by σ."""
        sigma = sigma.view(-1, 1, 1, 1) if sigma.dim() == 1 else sigma
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        denoised = c_skip * clean + c_out * model_output
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        return (weight * F.mse_loss(denoised, clean, reduction="none")).mean()

    def _sid2_loss(
        self,
        model_output: torch.Tensor,
        clean: torch.Tensor,
        noise: Optional[torch.Tensor],
        timesteps_or_t: torch.Tensor,
        scheduler: Optional[Any],
    ) -> torch.Tensor:
        """SiD2 sigmoid-weighted loss (discrete or continuous)."""
        target = self._get_target(clean, noise, timesteps_or_t, scheduler)
        mse = F.mse_loss(model_output, target, reduction="none")

        # Continuous t ∈ [0,1]: use cosine log-SNR and dlogsnr_dt
        if scheduler is not None and hasattr(scheduler, "cosine_interpolated_logsnr"):
            t = timesteps_or_t.to(model_output.device).float()
            cfg = getattr(scheduler, "config", None)
            if t.max() > 1.0 and cfg is not None:
                t = t / getattr(cfg, "num_train_timesteps", 1000)
            img_res = getattr(scheduler, "image_size", 512)
            ln_min = getattr(cfg, "logsnr_min", -10.0) if cfg else -10.0
            ln_max = getattr(cfg, "logsnr_max", 10.0) if cfg else 10.0
            logsnr = cosine_interpolated_logsnr(t, image_res=img_res, logsnr_min=ln_min, logsnr_max=ln_max)
            dlogsnr = _cosine_dlogsnr_dt(t, img_res, ln_min, ln_max)
            w = (
                -0.5
                * dlogsnr
                * self._sid2_exp_bias.to(model_output.device)
                * torch.sigmoid(logsnr - self.sid2_bias)
            )
            w = w.view(-1, *([1] * (mse.ndim - 1)))
        else:
            # Discrete: weight from alphas_cumprod
            if scheduler is None or not hasattr(scheduler, "alphas_cumprod"):
                raise ValueError(
                    "scheduler with alphas_cumprod or cosine_interpolated_logsnr required for loss_type='sid2'"
                )
            alpha_bar = self._extract(
                scheduler.alphas_cumprod, timesteps_or_t.long(), mse.shape
            )
            logsnr = torch.log(alpha_bar / (1 - alpha_bar + 1e-8))
            w = (
                self._sid2_exp_bias.to(model_output.device)
                * torch.sigmoid(logsnr - self.sid2_bias)
            )

        return (w * mse).mean()

    @staticmethod
    def _extract(a: torch.Tensor, t: torch.Tensor, shape: tuple) -> torch.Tensor:
        """Index and broadcast for loss weighting."""
        b = t.shape[0]
        out = a.to(t.device).gather(-1, t.reshape(-1).long().clamp(0, a.shape[-1] - 1))
        return out.reshape(b, *((1,) * (len(shape) - 1)))
