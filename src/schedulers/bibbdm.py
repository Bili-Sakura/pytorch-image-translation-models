# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""BiBBDM Scheduler – Brownian Bridge noise schedule for bidirectional image translation.

Implements the Brownian Bridge noise schedule from the BiBBDM paper in a
``diffusers``-compatible style.

Note
----
This scheduler is bidirectional and intentionally separate from
``src.schedulers.bbdm.BBDMScheduler`` (one-way BBDM).

The schedule defines:
* ``m_t`` — interpolation weight between endpoints *a* (target) and *b* (source).
* ``variance_t`` — noise variance at timestep *t*.

The forward (q-sample) process is:
    ``x_t = (1 - m_t) * a + m_t * b + sqrt(variance_t) * noise``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.schedulers.scheduling_utils import SchedulerMixin


@dataclass
class BiBBDMSchedulerOutput(BaseOutput):
    """Output of a single BiBBDM scheduler step.

    Attributes
    ----------
    prev_sample : torch.Tensor
        Denoised sample ``x_{t-1}`` (or ``x_{t+1}`` for a2b direction).
    pred_original_sample : torch.Tensor or None
        One-step prediction of the clean endpoint (a or b).
    """

    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None


def _extract(a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
    """Gather values from *a* at indices *t* and reshape for broadcasting."""
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class BiBBDMScheduler(SchedulerMixin, ConfigMixin):
    """Brownian Bridge scheduler for BiBBDM.

    Supports bidirectional sampling: ``step_b2a`` (source → target) and
    ``step_a2b`` (target → source).

    Parameters
    ----------
    num_timesteps : int
        Total number of diffusion timesteps.
    mt_type : str
        Schedule type for ``m_t``: ``"linear"``, ``"sin"``, or ``"log"``.
    m0, mT : float
        Start / end values for the linear ``m_t`` schedule.
    eta : float
        Stochasticity parameter for the reverse/forward sampling step.
    var_scale : float
        Variance scaling factor.
    skip_sample : bool
        If ``True`` use a reduced set of sampling steps.
    sample_step : int
        Number of sampling steps when ``skip_sample`` is ``True``.
    sample_step_type : str
        ``"linear"`` or ``"cosine"`` spacing for skip-sampling steps.
    objective : str
        Prediction objective (determines how to interpret UNet output).
    """

    @register_to_config
    def __init__(
        self,
        num_timesteps: int = 1000,
        mt_type: str = "linear",
        m0: float = 0.001,
        mT: float = 0.999,
        eta: float = 1.0,
        var_scale: float = 2.0,
        skip_sample: bool = True,
        sample_step: int = 100,
        sample_step_type: str = "linear",
        objective: str = "dlns",
    ) -> None:
        super().__init__()
        self.num_timesteps = num_timesteps
        self.mt_type = mt_type
        self.m0 = m0
        self.mT = mT
        self.eta = eta
        self.var_scale = var_scale
        self.skip_sample = skip_sample
        self.sample_step = sample_step
        self.sample_step_type = sample_step_type
        self.objective = objective

        self.m_t: Optional[torch.Tensor] = None
        self.variance_t: Optional[torch.Tensor] = None
        self.steps: Optional[torch.Tensor] = None

        self._register_schedule()

    def _register_schedule(self) -> None:
        T = self.num_timesteps

        if self.mt_type == "linear":
            m_t = np.linspace(self.m0, self.mT, T)
        elif self.mt_type == "sin":
            m_t = np.arange(T) / T
            m_t[0] = 0.0005
            m_t = 0.5 * np.sin(np.pi * (m_t - 0.5)) + 0.5
        elif self.mt_type == "log":
            if T != 1000:
                raise ValueError(
                    f"mt_type='log' requires num_timesteps=1000 (got {T}); "
                    "use 'linear' or 'sin' for other values."
                )
            head = np.exp(np.linspace(np.log(self.m0), np.log(0.1), 270))
            mid = np.linspace(0.10165, 0.89835, 460)
            tail = np.flip(1.0 - head)
            m_t = np.concatenate((head, mid, tail))
        else:
            raise NotImplementedError(f"Unknown mt_type: {self.mt_type}")

        variance_t = (m_t - m_t ** 2) * self.var_scale

        self.m_t = torch.tensor(m_t, dtype=torch.float32)
        self.variance_t = torch.tensor(variance_t, dtype=torch.float32)
        self._build_steps()

    def _build_steps(self) -> None:
        if self.skip_sample:
            if self.sample_step_type == "linear":
                midsteps = torch.arange(
                    self.num_timesteps - 2, 1,
                    step=-((self.num_timesteps - 3) / (self.sample_step - 3)),
                ).long()
                self.steps = torch.cat(
                    (
                        torch.tensor([self.num_timesteps - 1], dtype=torch.long),
                        midsteps,
                        torch.tensor([1, 0], dtype=torch.long),
                    ),
                    dim=0,
                )
            elif self.sample_step_type == "cosine":
                steps = np.linspace(0, self.num_timesteps, self.sample_step + 1)
                steps = (np.cos(steps / self.num_timesteps * np.pi) + 1.0) / 2.0 * self.num_timesteps
                self.steps = torch.from_numpy(steps).long()
            else:
                raise NotImplementedError(f"Unknown sample_step_type: {self.sample_step_type}")
        else:
            self.steps = torch.arange(self.num_timesteps - 1, -1, -1)

    def add_noise(
        self,
        target: torch.Tensor,
        source: torch.Tensor,
        timesteps: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Brownian Bridge forward process: create ``x_t`` from endpoints.

        Parameters
        ----------
        target : Tensor (B, C, H, W)
            Clean target image (``a`` in BiBBDM notation).
        source : Tensor (B, C, H, W)
            Source image (``b`` in BiBBDM notation).
        timesteps : Tensor (B,) of long
            Timestep indices.
        noise : Tensor or None
            Optional pre-sampled Gaussian noise.

        Returns
        -------
        Tensor (B, C, H, W)
            Noisy sample ``x_t``.
        """
        if noise is None:
            noise = torch.randn_like(target)
        m_t = _extract(self.m_t.to(target.device), timesteps, target.shape)
        var_t = _extract(self.variance_t.to(target.device), timesteps, target.shape)
        sigma_t = torch.sqrt(var_t)
        return (1.0 - m_t) * target + m_t * source + sigma_t * noise

    def get_objective(
        self,
        target: torch.Tensor,
        source: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Return the training objective for the given samples."""
        m_t = _extract(self.m_t.to(target.device), timesteps, target.shape)
        var_t = _extract(self.variance_t.to(target.device), timesteps, target.shape)
        sigma_t = torch.sqrt(var_t)

        obj = self.objective
        if obj == "a":
            return target
        if obj == "grada":
            return m_t * (source - target) + sigma_t * noise
        if obj == "b":
            return source
        if obj == "gradb":
            return (m_t - 1.0) * (source - target) + sigma_t * noise
        if obj == "noise":
            return noise
        if obj == "bsuba":
            return source - target
        if obj == "dlns":
            return torch.cat((source - target, noise), dim=1)
        if obj == "dlab":
            return torch.cat((target, source), dim=1)
        if obj == "dlgab":
            obj_a = m_t * (source - target) + sigma_t * noise
            obj_b = (m_t - 1.0) * (source - target) + sigma_t * noise
            return torch.cat((obj_a, obj_b), dim=1)
        raise NotImplementedError(f"Unknown objective: {obj}")

    def predict_target_from_objective(
        self,
        x_t: torch.Tensor,
        source: torch.Tensor,
        timesteps: torch.Tensor,
        objective_recon: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the target (``a``) from the UNet's objective prediction."""
        _, c, _, _ = objective_recon.shape
        obj = self.objective
        if obj == "a":
            return objective_recon
        if obj == "grada":
            return x_t - objective_recon
        if obj == "noise":
            m_t = _extract(self.m_t.to(x_t.device), timesteps, x_t.shape)
            var_t = _extract(self.variance_t.to(x_t.device), timesteps, x_t.shape)
            sigma_t = torch.sqrt(var_t)
            return (x_t - m_t * source - sigma_t * objective_recon) / (1.0 - m_t + 1e-8)
        if obj == "bsuba":
            return source - objective_recon
        if obj == "dlns":
            m_t = _extract(self.m_t.to(x_t.device), timesteps, x_t.shape)
            var_t = _extract(self.variance_t.to(x_t.device), timesteps, x_t.shape)
            sigma_t = torch.sqrt(var_t)
            bsuba_recon = objective_recon[:, 0 : c // 2, :, :]
            noise_recon = objective_recon[:, c // 2 :, :, :]
            return x_t - m_t * bsuba_recon - sigma_t * noise_recon
        if obj == "dlab":
            return objective_recon[:, 0 : c // 2, :, :]
        if obj == "dlgab":
            return x_t - objective_recon[:, 0 : c // 2, :, :]
        raise NotImplementedError(f"Unknown objective: {obj}")

    def predict_source_from_objective(
        self,
        x_t: torch.Tensor,
        target: torch.Tensor,
        timesteps: torch.Tensor,
        objective_recon: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the source (``b``) from the UNet's objective prediction."""
        _, c, _, _ = objective_recon.shape
        obj = self.objective
        if obj == "b":
            return objective_recon
        if obj == "gradb":
            return x_t - objective_recon
        if obj == "noise":
            m_t = _extract(self.m_t.to(x_t.device), timesteps, x_t.shape)
            var_t = _extract(self.variance_t.to(x_t.device), timesteps, x_t.shape)
            sigma_t = torch.sqrt(var_t)
            return (x_t - (1.0 - m_t) * target - sigma_t * objective_recon) / (m_t + 1e-8)
        if obj == "bsuba":
            return target + objective_recon
        if obj == "dlns":
            m_t = _extract(self.m_t.to(x_t.device), timesteps, x_t.shape)
            var_t = _extract(self.variance_t.to(x_t.device), timesteps, x_t.shape)
            sigma_t = torch.sqrt(var_t)
            bsuba_recon = objective_recon[:, 0 : c // 2, :, :]
            noise_recon = objective_recon[:, c // 2 :, :, :]
            return x_t - (m_t - 1.0) * bsuba_recon - sigma_t * noise_recon
        if obj == "dlab":
            return objective_recon[:, c // 2 :, :, :]
        if obj == "dlgab":
            return x_t - objective_recon[:, c // 2 :, :, :]
        raise NotImplementedError(f"Unknown objective: {obj}")

    def step_b2a(
        self,
        model_output: torch.Tensor,
        step_index: int,
        x_t: torch.Tensor,
        source: torch.Tensor,
        clip_denoised: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> BiBBDMSchedulerOutput:
        """One reverse sampling step: ``x_t → x_{t-1}`` toward the target.

        Parameters
        ----------
        model_output : Tensor
            UNet output (objective reconstruction).
        step_index : int
            Index into ``self.steps``.
        x_t : Tensor
            Current noisy sample.
        source : Tensor
            Source image (``b``).
        clip_denoised : bool
            Clamp prediction to [-1, 1].
        generator : torch.Generator or None
            RNG for reproducibility.
        """
        device = x_t.device
        t = torch.full((x_t.shape[0],), self.steps[step_index].item(), device=device, dtype=torch.long)

        target_recon = self.predict_target_from_objective(x_t, source, t, model_output)
        if clip_denoised:
            target_recon = target_recon.clamp(-1.0, 1.0)

        if self.steps[step_index] == 0:
            return BiBBDMSchedulerOutput(prev_sample=target_recon, pred_original_sample=target_recon)

        n_t = torch.full((x_t.shape[0],), self.steps[step_index + 1].item(), device=device, dtype=torch.long)
        m_t = _extract(self.m_t.to(device), t, x_t.shape)
        m_nt = _extract(self.m_t.to(device), n_t, x_t.shape)
        var_t = _extract(self.variance_t.to(device), t, x_t.shape)
        var_nt = _extract(self.variance_t.to(device), n_t, x_t.shape)

        noise = torch.randn(x_t.shape, device=device, generator=generator, dtype=x_t.dtype)
        sigma2_t = self.config.var_scale * (m_t - m_nt) * m_nt / m_t
        sigma_t = torch.sqrt(sigma2_t) * self.eta
        coe_eps = torch.sqrt((var_nt - sigma_t ** 2) / var_t)

        x_prev = (
            (1.0 - m_nt) * target_recon
            + m_nt * source
            + coe_eps * (x_t - (1.0 - m_t) * target_recon - m_t * source)
            + sigma_t * noise
        )
        return BiBBDMSchedulerOutput(prev_sample=x_prev, pred_original_sample=target_recon)

    def step_a2b(
        self,
        model_output: torch.Tensor,
        step_index: int,
        x_t: torch.Tensor,
        target: torch.Tensor,
        clip_denoised: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> BiBBDMSchedulerOutput:
        """One forward sampling step: ``x_t → x_{t+1}`` toward the source.

        Parameters
        ----------
        model_output : Tensor
            UNet output (objective reconstruction).
        step_index : int
            Index into reversed ``self.steps``.
        x_t : Tensor
            Current sample.
        target : Tensor
            Target image (``a``).
        clip_denoised : bool
            Clamp prediction to [-1, 1].
        generator : torch.Generator or None
            RNG for reproducibility.
        """
        device = x_t.device
        t = torch.full((x_t.shape[0],), self.steps[step_index].item(), device=device, dtype=torch.long)

        source_recon = self.predict_source_from_objective(x_t, target, t, model_output)
        if clip_denoised:
            source_recon = source_recon.clamp(-1.0, 1.0)

        if self.steps[step_index] == self.num_timesteps - 1:
            return BiBBDMSchedulerOutput(prev_sample=source_recon, pred_original_sample=source_recon)

        n_t = torch.full((x_t.shape[0],), self.steps[step_index - 1].item(), device=device, dtype=torch.long)
        m_t = _extract(self.m_t.to(device), t, x_t.shape)
        m_nt = _extract(self.m_t.to(device), n_t, x_t.shape)
        var_t = _extract(self.variance_t.to(device), t, x_t.shape)
        var_nt = _extract(self.variance_t.to(device), n_t, x_t.shape)

        sigma2_t = 2 * (1 - m_nt) * (m_nt - m_t) / (1 - m_t + 1e-8)
        noise = torch.randn(x_t.shape, device=device, generator=generator, dtype=x_t.dtype)
        sigma_t = torch.sqrt(sigma2_t) * self.eta
        coe_eps = torch.sqrt((var_nt - sigma_t ** 2) / var_t)

        x_next = (
            (1.0 - m_nt) * target
            + m_nt * source_recon
            + coe_eps * (x_t - (1.0 - m_t) * target - m_t * source_recon)
            + sigma_t * noise
        )
        return BiBBDMSchedulerOutput(prev_sample=x_next, pred_original_sample=source_recon)

    def set_timesteps(self, num_inference_steps: Optional[int] = None) -> None:
        """Rebuild the sampling step schedule (optionally with a new step count)."""
        if num_inference_steps is not None:
            self.sample_step = num_inference_steps
            self.skip_sample = True
        self._build_steps()
