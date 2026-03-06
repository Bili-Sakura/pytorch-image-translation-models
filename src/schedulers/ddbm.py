# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""DDBM Scheduler – noise schedule and sampling for Denoising Diffusion Bridge Models.

Implements the Heun sampler from the DDBM paper (https://arxiv.org/abs/2309.16948)
for image-to-image translation tasks using diffusion bridges.  Unlike standard
diffusion models that map noise to data, bridge models learn to transform between
two data distributions (source -> target).

This scheduler is compatible with the Hugging Face ``diffusers`` library.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_utils import SchedulerMixin


@dataclass
class DDBMSchedulerOutput(BaseOutput):
    """Output of a single DDBM scheduler step.

    Attributes
    ----------
    prev_sample : torch.Tensor
        Denoised sample ``x_{t-1}`` after one scheduler step.
    pred_original_sample : torch.Tensor or None
        Predicted clean sample ``x_0`` from the model, if available.
    """

    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None


class DDBMScheduler(SchedulerMixin, ConfigMixin):
    """Scheduler for Denoising Diffusion Bridge Models (DDBM).

    Implements Karras-style sigma scheduling and Euler / Heun sampling
    for diffusion bridge models that transform between two data
    distributions.

    Parameters
    ----------
    sigma_min : float
        Minimum noise level for the sigma schedule.
    sigma_max : float
        Maximum noise level (also the endpoint noise of the bridge).
    sigma_data : float
        Standard deviation of the data distribution, used for
        preconditioning.
    beta_d : float
        VP schedule parameter controlling the quadratic term of
        ``beta(t) = beta_min + beta_d * t``.
    beta_min : float
        VP schedule minimum beta value.
    rho : float
        Exponent for the Karras sigma schedule interpolation.
    pred_mode : str
        Noise schedule type: ``"vp"`` (variance preserving) or
        ``"ve"`` (variance exploding).
    num_train_timesteps : int
        Default number of training / inference timesteps.
    """

    @register_to_config
    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        sigma_data: float = 0.5,
        beta_d: float = 2.0,
        beta_min: float = 0.1,
        rho: float = 7.0,
        pred_mode: str = "vp",
        num_train_timesteps: int = 40,
    ) -> None:
        self.sigmas: torch.Tensor | None = None
        self.timesteps: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # VP helper functions
    # ------------------------------------------------------------------

    def _vp_logsnr(self, t: torch.Tensor) -> torch.Tensor:
        """Compute the log signal-to-noise ratio for the VP schedule.

        Parameters
        ----------
        t : torch.Tensor
            Continuous time values.

        Returns
        -------
        torch.Tensor
            Log-SNR: ``-beta_d * t**2 / 2 - beta_min * t``.
        """
        return -self.config.beta_d * t**2 / 2 - self.config.beta_min * t

    def _vp_logs(self, t: torch.Tensor) -> torch.Tensor:
        """Compute the log scaling factor ``log s(t)`` for the VP schedule.

        The scaling factor is ``s(t) = exp(logs(t))`` and satisfies
        ``ds/dt = -beta(t)/2 * s(t)``.

        Parameters
        ----------
        t : torch.Tensor
            Continuous time values.

        Returns
        -------
        torch.Tensor
            ``-beta_d * t**2 / 4 - beta_min * t / 2``.
        """
        return -self.config.beta_d * t**2 / 4 - self.config.beta_min * t / 2

    def _vp_sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Compute the noise level ``sigma(t)`` for the VP schedule.

        Parameters
        ----------
        t : torch.Tensor
            Continuous time values.

        Returns
        -------
        torch.Tensor
            ``sqrt(1 - exp(2 * logs(t)))``.
        """
        return (1 - torch.exp(2 * self._vp_logs(t))).clamp(min=1e-8).sqrt()

    def _sigma_to_t(self, sigma: torch.Tensor) -> torch.Tensor:
        """Convert a VP noise level ``sigma`` back to continuous time *t*.

        Inverts ``sigma(t) = sqrt(1 - exp(2 * logs(t)))`` by solving the
        quadratic ``beta_d * t**2 + 2 * beta_min * t + 2 * log(1 - sigma**2) = 0``.

        Parameters
        ----------
        sigma : torch.Tensor
            Noise levels.

        Returns
        -------
        torch.Tensor
            Continuous time values corresponding to each sigma.
        """
        log_term = torch.log((1 - sigma**2).clamp(min=1e-8))
        discriminant = (
            self.config.beta_min**2 - 2 * self.config.beta_d * log_term
        )
        t = (
            -self.config.beta_min + discriminant.clamp(min=0).sqrt()
        ) / self.config.beta_d
        return t.clamp(min=0)

    # ------------------------------------------------------------------
    # Timestep scheduling
    # ------------------------------------------------------------------

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        """Build the Karras sigma schedule for inference.

        Sigma values are interpolated in ``sigma**(1/rho)`` space between
        ``sigma_max`` and ``sigma_min``, then a final ``sigma = 0`` is
        appended.

        Parameters
        ----------
        num_inference_steps : int
            Number of denoising steps.
        device : str or torch.device
            Device on which to place the schedule tensors.
        """
        rho = self.config.rho
        sigma_min = self.config.sigma_min
        sigma_max = self.config.sigma_max

        ramp = torch.linspace(0, 1, num_inference_steps, device=device)
        min_inv_rho = sigma_min ** (1.0 / rho)
        max_inv_rho = sigma_max ** (1.0 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho

        self.sigmas = torch.cat([sigmas, sigmas.new_zeros(1)])
        self.timesteps = torch.arange(num_inference_steps, device=device)

    # ------------------------------------------------------------------
    # Step methods
    # ------------------------------------------------------------------

    def _get_sigma_pair(
        self, timestep: Union[int, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(sigma_cur, sigma_next)`` for the given timestep index.

        Parameters
        ----------
        timestep : int or torch.Tensor
            Discrete timestep index into ``self.sigmas``.

        Returns
        -------
        tuple of torch.Tensor
            Current and next sigma values.
        """
        index = timestep.item() if isinstance(timestep, torch.Tensor) else int(timestep)
        return self.sigmas[index], self.sigmas[index + 1]

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        x_T: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[DDBMSchedulerOutput, Tuple]:
        """Perform a single Euler step of the bridge probability-flow ODE.

        Parameters
        ----------
        model_output : torch.Tensor
            Denoised prediction from the model (``x_0`` estimate).
        timestep : int
            Current discrete timestep index.
        sample : torch.Tensor
            Current noisy sample ``x_t``.
        x_T : torch.Tensor
            Source (endpoint) sample that anchors the bridge.
        return_dict : bool
            If ``True``, return a :class:`DDBMSchedulerOutput`.

        Returns
        -------
        DDBMSchedulerOutput or tuple
            The denoised sample after one Euler step.
        """
        sigma_cur, sigma_next = self._get_sigma_pair(timestep)

        denoised = model_output

        if self.config.pred_mode == "vp":
            d = self._vp_ode_derivative(sample, denoised, sigma_cur, x_T)
        else:
            d = self._ve_ode_derivative(sample, denoised, sigma_cur, x_T)

        dt = sigma_next - sigma_cur
        prev_sample = sample + d * dt

        if not return_dict:
            return (prev_sample, denoised)
        return DDBMSchedulerOutput(
            prev_sample=prev_sample,
            pred_original_sample=denoised,
        )

    def step_heun(
        self,
        denoised_1: torch.Tensor,
        denoised_2: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        x_T: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[DDBMSchedulerOutput, Tuple]:
        """Perform a full Heun (improved Euler) step with two evaluations.

        The caller is responsible for obtaining *denoised_1* at the current
        sigma and *denoised_2* at the Euler-predicted intermediate point.

        Parameters
        ----------
        denoised_1 : torch.Tensor
            Model denoised output at the current sigma level.
        denoised_2 : torch.Tensor
            Model denoised output at the intermediate (Euler) sigma level.
        timestep : int
            Current discrete timestep index.
        sample : torch.Tensor
            Current noisy sample ``x_t``.
        x_T : torch.Tensor
            Source (endpoint) sample that anchors the bridge.
        return_dict : bool
            If ``True``, return a :class:`DDBMSchedulerOutput`.

        Returns
        -------
        DDBMSchedulerOutput or tuple
            The denoised sample after one Heun step.
        """
        sigma_cur, sigma_next = self._get_sigma_pair(timestep)

        if self.config.pred_mode == "vp":
            d1 = self._vp_ode_derivative(sample, denoised_1, sigma_cur, x_T)
        else:
            d1 = self._ve_ode_derivative(sample, denoised_1, sigma_cur, x_T)

        dt = sigma_next - sigma_cur
        x_euler = sample + d1 * dt

        if sigma_next.item() == 0:
            prev_sample = x_euler
        else:
            if self.config.pred_mode == "vp":
                d2 = self._vp_ode_derivative(x_euler, denoised_2, sigma_next, x_T)
            else:
                d2 = self._ve_ode_derivative(x_euler, denoised_2, sigma_next, x_T)

            d_avg = (d1 + d2) / 2.0
            prev_sample = sample + d_avg * dt

        if not return_dict:
            return (prev_sample, denoised_1)
        return DDBMSchedulerOutput(
            prev_sample=prev_sample,
            pred_original_sample=denoised_1,
        )

    # ------------------------------------------------------------------
    # ODE derivative computations
    # ------------------------------------------------------------------

    def _ve_ode_derivative(
        self,
        x: torch.Tensor,
        denoised: torch.Tensor,
        sigma: torch.Tensor,
        x_T: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ``dx/dsigma`` for the VE bridge probability-flow ODE.

        For VE the scaling factor is unity (``s(t) = 1``), so the
        derivative simplifies to a standard score term plus a bridge
        guidance term.

        Parameters
        ----------
        x : torch.Tensor
            Current sample.
        denoised : torch.Tensor
            Predicted clean sample ``x_0``.
        sigma : torch.Tensor
            Current noise level.
        x_T : torch.Tensor
            Bridge endpoint.

        Returns
        -------
        torch.Tensor
            Derivative ``dx/dsigma``.
        """
        sigma_max = self.config.sigma_max
        bridge_var = (sigma_max**2 - sigma**2).clamp(min=1e-8)

        d_score = (x - denoised) / sigma
        d_bridge = sigma * (x_T - x) / bridge_var

        return d_score + d_bridge

    def _vp_ode_derivative(
        self,
        x: torch.Tensor,
        denoised: torch.Tensor,
        sigma: torch.Tensor,
        x_T: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ``dx/dsigma`` for the VP bridge probability-flow ODE.

        Converts the time-domain ODE ``dx/dt`` to sigma space by dividing
        by ``dsigma/dt``.  The VP drift, unconditional score, and bridge
        guidance are combined into a single expression.

        Parameters
        ----------
        x : torch.Tensor
            Current sample.
        denoised : torch.Tensor
            Predicted clean sample ``x_0``.
        sigma : torch.Tensor
            Current noise level.
        x_T : torch.Tensor
            Bridge endpoint.

        Returns
        -------
        torch.Tensor
            Derivative ``dx/dsigma``.
        """
        sigma_max = torch.as_tensor(
            self.config.sigma_max, dtype=sigma.dtype, device=sigma.device
        )

        t = self._sigma_to_t(sigma)
        t_max = self._sigma_to_t(sigma_max)

        s_t = torch.exp(self._vp_logs(t))
        s_max = torch.exp(self._vp_logs(t_max))

        # Scale ratio  s(T) / s(t)
        r = s_max / s_t

        # Bridge transition variance:  sigma_T^2 - r^2 * sigma^2
        bridge_var = (sigma_max**2 - r**2 * sigma**2).clamp(min=1e-8)
        s_sq = s_t**2

        # Three additive terms of the VP bridge PF-ODE in sigma space:
        #   d_drift  = -x * sigma / s^2              (VP drift)
        #   d_score  = (x - s*D) / (sigma * s^2)     (unconditional score)
        #   d_bridge = r*(x_T - r*x)*sigma / (V*s^2) (bridge guidance)
        d_drift = -x * sigma / s_sq
        d_score = (x - s_t * denoised) / (sigma * s_sq)
        d_bridge = r * (x_T - r * x) * sigma / (bridge_var * s_sq)

        return d_drift + d_score + d_bridge

    # ------------------------------------------------------------------
    # Forward process (training)
    # ------------------------------------------------------------------

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        x_T: torch.Tensor,
    ) -> torch.Tensor:
        """Add bridge noise to clean samples (forward process).

        Samples from the bridge kernel ``q(x_t | x_0, x_T)`` using the
        provided Gaussian *noise*.  The *timesteps* tensor is interpreted
        as sigma values (noise levels), not discrete indices.

        Parameters
        ----------
        original_samples : torch.Tensor
            Clean target samples ``x_0``.
        noise : torch.Tensor
            Standard Gaussian noise with the same shape as
            *original_samples*.
        timesteps : torch.Tensor
            Sigma values (noise levels) for each sample in the batch.
        x_T : torch.Tensor
            Source (endpoint) samples.

        Returns
        -------
        torch.Tensor
            Noisy bridge samples ``x_t``.
        """
        sigma = timesteps.float()
        while sigma.dim() < original_samples.dim():
            sigma = sigma.unsqueeze(-1)

        sigma_max = self.config.sigma_max
        sigma_sq = sigma**2
        sigma_max_sq = sigma_max**2

        if self.config.pred_mode == "vp":
            sigma_max_t = torch.as_tensor(
                sigma_max, dtype=sigma.dtype, device=sigma.device
            )
            t = self._sigma_to_t(sigma)
            t_max = self._sigma_to_t(sigma_max_t)

            s_t = torch.exp(self._vp_logs(t))
            s_max = torch.exp(self._vp_logs(t_max))
            r = s_max / s_t

            bridge_var = (sigma_max_sq - r**2 * sigma_sq).clamp(min=1e-8)

            coeff_x0 = s_t * bridge_var / sigma_max_sq
            coeff_xT = r * sigma_sq / sigma_max_sq
        else:
            bridge_var = (sigma_max_sq - sigma_sq).clamp(min=1e-8)

            coeff_x0 = bridge_var / sigma_max_sq
            coeff_xT = sigma_sq / sigma_max_sq

        mu = coeff_x0 * original_samples + coeff_xT * x_T
        std = (sigma_sq * bridge_var / sigma_max_sq).clamp(min=0).sqrt()

        return mu + std * noise

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def scale_model_input(
        self,
        sample: torch.Tensor,
        timestep: Optional[int] = None,
    ) -> torch.Tensor:
        """Scale the model input (identity for DDBM).

        Included for API compatibility with the ``diffusers`` scheduler
        interface.

        Parameters
        ----------
        sample : torch.Tensor
            Input sample.
        timestep : int or None
            Current timestep (unused).

        Returns
        -------
        torch.Tensor
            Unmodified *sample*.
        """
        return sample
