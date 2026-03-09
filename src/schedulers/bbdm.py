# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""BBDM scheduler for one-way Brownian Bridge image translation.

This scheduler follows the original BBDM formulation (CVPR 2023), which models
the bridge from target ``x0`` to source ``y`` and performs reverse sampling
from ``y`` back to ``x0``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import BaseOutput


@dataclass
class BBDMSchedulerOutput(BaseOutput):
    """Output of one BBDM reverse step."""

    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None


def _extract(values: torch.Tensor, timesteps: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
    """Gather timestep-indexed scalars and reshape for broadcasting."""
    bsz, *_ = timesteps.shape
    out = values.gather(-1, timesteps)
    return out.reshape(bsz, *((1,) * (len(x_shape) - 1)))


class BBDMScheduler(SchedulerMixin, ConfigMixin):
    """Brownian Bridge scheduler for BBDM (source -> target reverse path).

    Parameters
    ----------
    num_timesteps : int
        Total diffusion timesteps.
    mt_type : str
        ``"linear"`` or ``"sin"`` schedule for ``m_t``.
    eta : float
        Stochasticity parameter.
    max_var : float
        Variance scaling factor.
    skip_sample : bool
        If ``True`` use reduced sampling steps.
    sample_step : int
        Number of sampling steps when ``skip_sample`` is ``True``.
    sample_step_type : str
        ``"linear"`` or ``"cosine"`` spacing.
    objective : str
        Objective for reconstruction, one of ``"grad"``, ``"noise"``, ``"ysubx"``.
    """

    @register_to_config
    def __init__(
        self,
        num_timesteps: int = 1000,
        mt_type: str = "linear",
        eta: float = 1.0,
        max_var: float = 1.0,
        skip_sample: bool = True,
        sample_step: int = 200,
        sample_step_type: str = "linear",
        objective: str = "grad",
    ) -> None:
        super().__init__()
        self.num_timesteps = num_timesteps
        self.mt_type = mt_type
        self.eta = eta
        self.max_var = max_var
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
            m_t = np.linspace(0.001, 0.999, T)
        elif self.mt_type == "sin":
            m_t = 1.0075 ** np.linspace(0, T, T)
            m_t = m_t / m_t[-1]
            m_t[-1] = 0.999
        else:
            raise NotImplementedError(f"Unknown mt_type: {self.mt_type}")

        variance_t = 2.0 * (m_t - m_t**2) * self.max_var
        self.m_t = torch.tensor(m_t, dtype=torch.float32)
        self.variance_t = torch.tensor(variance_t, dtype=torch.float32)
        self._build_steps()

    def _build_steps(self) -> None:
        if self.skip_sample:
            if self.sample_step_type == "linear":
                if self.sample_step < 3:
                    raise ValueError("sample_step must be >= 3 for linear skip-sampling.")
                step = -((self.num_timesteps - 1) / (self.sample_step - 2))
                midsteps = torch.arange(self.num_timesteps - 1, 1, step=step).long()
                self.steps = torch.cat((midsteps, torch.tensor([1, 0], dtype=torch.long)), dim=0)
            elif self.sample_step_type == "cosine":
                steps = np.linspace(0, self.num_timesteps, self.sample_step + 1)
                steps = (np.cos(steps / self.num_timesteps * np.pi) + 1.0) / 2.0 * (self.num_timesteps - 1)
                self.steps = torch.from_numpy(np.round(steps).astype(np.int64))
            else:
                raise NotImplementedError(f"Unknown sample_step_type: {self.sample_step_type}")
        else:
            self.steps = torch.arange(self.num_timesteps - 1, -1, -1, dtype=torch.long)

    def add_noise(
        self,
        target: torch.Tensor,
        source: torch.Tensor,
        timesteps: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample bridge state ``x_t`` from endpoints and Gaussian noise."""
        if noise is None:
            noise = torch.randn_like(target)
        m_t = _extract(self.m_t.to(target.device), timesteps, target.shape)
        var_t = _extract(self.variance_t.to(target.device), timesteps, target.shape)
        sigma_t = torch.sqrt(torch.clamp(var_t, min=0.0))
        return (1.0 - m_t) * target + m_t * source + sigma_t * noise

    def get_objective(
        self,
        target: torch.Tensor,
        source: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Return BBDM training objective."""
        m_t = _extract(self.m_t.to(target.device), timesteps, target.shape)
        var_t = _extract(self.variance_t.to(target.device), timesteps, target.shape)
        sigma_t = torch.sqrt(torch.clamp(var_t, min=0.0))
        if self.objective == "grad":
            return m_t * (source - target) + sigma_t * noise
        if self.objective == "noise":
            return noise
        if self.objective == "ysubx":
            return source - target
        raise NotImplementedError(f"Unknown objective: {self.objective}")

    def predict_target_from_objective(
        self,
        x_t: torch.Tensor,
        source: torch.Tensor,
        timesteps: torch.Tensor,
        objective_recon: torch.Tensor,
    ) -> torch.Tensor:
        """Recover target endpoint ``x0`` from objective prediction."""
        if self.objective == "grad":
            return x_t - objective_recon
        if self.objective == "noise":
            m_t = _extract(self.m_t.to(x_t.device), timesteps, x_t.shape)
            var_t = _extract(self.variance_t.to(x_t.device), timesteps, x_t.shape)
            sigma_t = torch.sqrt(torch.clamp(var_t, min=0.0))
            return (x_t - m_t * source - sigma_t * objective_recon) / (1.0 - m_t + 1e-8)
        if self.objective == "ysubx":
            return source - objective_recon
        raise NotImplementedError(f"Unknown objective: {self.objective}")

    def step(
        self,
        model_output: torch.Tensor,
        step_index: int,
        x_t: torch.Tensor,
        source: torch.Tensor,
        clip_denoised: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> BBDMSchedulerOutput:
        """One reverse bridge step: ``x_t -> x_{t-1}``."""
        device = x_t.device
        t_value = int(self.steps[step_index].item())
        t = torch.full((x_t.shape[0],), t_value, device=device, dtype=torch.long)

        target_recon = self.predict_target_from_objective(x_t, source, t, model_output)
        if clip_denoised:
            target_recon = target_recon.clamp(-1.0, 1.0)

        if t_value == 0:
            return BBDMSchedulerOutput(prev_sample=target_recon, pred_original_sample=target_recon)

        n_t = torch.full(
            (x_t.shape[0],),
            int(self.steps[step_index + 1].item()),
            device=device,
            dtype=torch.long,
        )
        m_t = _extract(self.m_t.to(device), t, x_t.shape)
        m_nt = _extract(self.m_t.to(device), n_t, x_t.shape)
        var_t = _extract(self.variance_t.to(device), t, x_t.shape)
        var_nt = _extract(self.variance_t.to(device), n_t, x_t.shape)

        eps = 1e-8
        sigma2_t = (
            var_t
            - var_nt * ((1.0 - m_t) ** 2) / torch.clamp((1.0 - m_nt) ** 2, min=eps)
        ) * var_nt / torch.clamp(var_t, min=eps)
        sigma2_t = torch.clamp(sigma2_t, min=0.0)
        sigma_t = torch.sqrt(sigma2_t) * self.eta
        coeff = torch.sqrt(torch.clamp(var_nt - sigma2_t, min=0.0) / torch.clamp(var_t, min=eps))
        noise = torch.randn(x_t.shape, device=device, dtype=x_t.dtype, generator=generator)

        x_prev = (
            (1.0 - m_nt) * target_recon
            + m_nt * source
            + coeff * (x_t - (1.0 - m_t) * target_recon - m_t * source)
            + sigma_t * noise
        )
        return BBDMSchedulerOutput(prev_sample=x_prev, pred_original_sample=target_recon)

    def set_timesteps(self, num_inference_steps: Optional[int] = None) -> None:
        """Rebuild sampling schedule, optionally overriding step count."""
        if num_inference_steps is not None:
            self.sample_step = num_inference_steps
            self.skip_sample = True
        self._build_steps()
