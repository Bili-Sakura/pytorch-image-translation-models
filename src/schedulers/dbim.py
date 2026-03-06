# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""DBIM Scheduler – Diffusion Bridge Implicit Models sampler.

DBIM uses the same bridge model family as DDBM but introduces faster
samplers (``dbim``, ``dbim_high_order``) in addition to the original
Heun-based sampler.

Reference
---------
Zheng, K., et al. "Diffusion Bridge Implicit Models." 2024.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor


@dataclass
class DBIMSchedulerOutput(BaseOutput):
    """Output of a single DBIM scheduler step.

    Attributes
    ----------
    prev_sample : torch.Tensor
        Computed sample for the next timestep.
    pred_original_sample : torch.Tensor or None
        Predicted denoised sample used in the update.
    """

    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None


class DBIMScheduler(SchedulerMixin, ConfigMixin):
    """Scheduler for Diffusion Bridge Implicit Models (DBIM).

    Parameters
    ----------
    sigma_min : float
        Minimum time/noise level.
    sigma_max : float
        Maximum time/noise level.
    sigma_data : float
        Data standard deviation used by bridge preconditioning.
    beta_d : float
        VP schedule parameter.
    beta_min : float
        VP schedule parameter.
    rho : float
        Karras schedule exponent for ``heun``.
    pred_mode : str
        Bridge noise mode (``"vp"`` or ``"ve"``).
    num_train_timesteps : int
        Training-time nominal number of steps.
    sampler : str
        Default sampler: ``"dbim"``, ``"dbim_high_order"``, or ``"heun"``.
    eta : float
        Stochasticity level for DBIM sampler (0 = deterministic).
    order : int
        Solver order for ``dbim_high_order`` (2 or 3).
    lower_order_final : bool
        Use lower-order final update for stability.
    """

    _compatibles = []
    order = 2

    @register_to_config
    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 1.0,
        sigma_data: float = 0.5,
        beta_d: float = 2.0,
        beta_min: float = 0.1,
        rho: float = 7.0,
        pred_mode: str = "vp",
        num_train_timesteps: int = 40,
        sampler: str = "dbim",
        eta: float = 1.0,
        order: int = 2,
        lower_order_final: bool = True,
    ):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.rho = rho
        self.pred_mode = pred_mode
        self.num_train_timesteps = num_train_timesteps
        self.sampler = sampler
        self.eta = eta
        self.dbim_order = order
        self.lower_order_final = lower_order_final

        self.sigmas: Optional[torch.Tensor] = None
        self.timesteps: Optional[torch.Tensor] = None
        self.num_inference_steps: Optional[int] = None

        self.init_noise_sigma = sigma_max

    @staticmethod
    def _append_dims(x: torch.Tensor, target_dims: int) -> torch.Tensor:
        """Append singleton dimensions until ``x.ndim == target_dims``."""
        dims_to_append = target_dims - x.ndim
        if dims_to_append < 0:
            raise ValueError(
                f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
            )
        return x[(...,) + (None,) * dims_to_append]

    def _mode(self) -> str:
        if self.pred_mode.startswith("vp"):
            return "vp"
        if self.pred_mode.startswith("ve"):
            return "ve"
        raise ValueError(
            f"Unsupported pred_mode '{self.pred_mode}'. Expected vp/ve variants."
        )

    def _rho_terminal(
        self,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        mode = self._mode()
        if mode == "vp":
            t_T = torch.tensor(self.sigma_max, device=device, dtype=dtype)
            return torch.sqrt(
                torch.clamp(
                    torch.exp(self.beta_min * t_T + 0.5 * self.beta_d * t_T**2) - 1.0,
                    min=0.0,
                )
            )
        return torch.tensor(self.sigma_max, device=device, dtype=dtype)

    def get_alpha_rho(
        self, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(alpha_t, alpha_bar_t, rho_t, rho_bar_t)``."""
        t = torch.as_tensor(t, dtype=torch.float32)
        mode = self._mode()

        if mode == "vp":
            alpha_t = torch.exp(-0.5 * self.beta_min * t - 0.25 * self.beta_d * t**2)
            t_T = torch.tensor(self.sigma_max, device=t.device, dtype=t.dtype)
            alpha_T = torch.exp(
                -0.5 * self.beta_min * t_T - 0.25 * self.beta_d * t_T**2
            )
            alpha_bar_t = alpha_t / alpha_T
            rho_t = torch.sqrt(
                torch.clamp(
                    torch.exp(self.beta_min * t + 0.5 * self.beta_d * t**2) - 1.0,
                    min=0.0,
                )
            )
            rho_T = self._rho_terminal(t.device, t.dtype)
            rho_bar_t = torch.sqrt(torch.clamp(rho_T**2 - rho_t**2, min=0.0))
            return alpha_t, alpha_bar_t, rho_t, rho_bar_t

        # ve
        alpha_t = torch.ones_like(t)
        alpha_bar_t = torch.ones_like(t)
        rho_t = t
        rho_T = self._rho_terminal(t.device, t.dtype)
        rho_bar_t = torch.sqrt(torch.clamp(rho_T**2 - rho_t**2, min=0.0))
        return alpha_t, alpha_bar_t, rho_t, rho_bar_t

    def get_abc(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return bridge coefficients ``(a_t, b_t, c_t)``."""
        alpha_t, alpha_bar_t, rho_t, rho_bar_t = self.get_alpha_rho(t)
        rho_T = self._rho_terminal(
            device=torch.as_tensor(t).device, dtype=torch.as_tensor(t).dtype
        )
        a_t = (alpha_bar_t * rho_t**2) / (rho_T**2)
        b_t = (alpha_t * rho_bar_t**2) / (rho_T**2)
        c_t = (alpha_t * rho_bar_t * rho_t) / rho_T
        return a_t, b_t, c_t

    def bridge_sample(
        self,
        x0: torch.Tensor,
        x_T: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Sample ``x_t`` from bridge marginal ``q(x_t | x_0, x_T)``."""
        a_t, b_t, c_t = [
            self._append_dims(item, x0.ndim) for item in self.get_abc(t)
        ]
        return a_t * x_T + b_t * x0 + c_t * noise

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Union[str, torch.device] = None,
        sampler: Optional[str] = None,
    ) -> None:
        """Set sampling timesteps for the selected sampler."""
        self.num_inference_steps = num_inference_steps
        sampler_name = sampler or self.config.sampler

        if sampler_name == "heun":
            ramp = torch.linspace(0, 1, num_inference_steps)
            min_inv_rho = self.sigma_min ** (1 / self.rho)
            max_inv_rho = max(self.sigma_max - 1e-4, self.sigma_min) ** (1 / self.rho)
            sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** self.rho
            sigmas = torch.cat([sigmas, torch.zeros(1)])
        else:
            t_max = max(self.sigma_max - 1e-3, self.sigma_min)
            sigmas = torch.linspace(t_max, self.sigma_min, num_inference_steps + 1)

        self.sigmas = sigmas.to(device=device)
        self.timesteps = torch.arange(len(self.sigmas) - 1, device=device)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        x_T: torch.Tensor,
    ) -> torch.Tensor:
        """Add bridge noise for training."""
        t = timesteps.to(device=original_samples.device, dtype=original_samples.dtype)
        return self.bridge_sample(original_samples, x_T, t, noise)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        x_T: torch.Tensor,
        eta: Optional[float] = None,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[DBIMSchedulerOutput, Tuple]:
        """Single first-order DBIM step."""
        if self.sigmas is None:
            raise ValueError("Sigmas not initialized. Call `set_timesteps` first.")

        i = int(timestep)
        if i < 0 or i >= len(self.sigmas) - 1:
            raise ValueError(
                f"timestep index {i} out of range for schedule length {len(self.sigmas)}"
            )

        eta_val = self.config.eta if eta is None else eta
        sigma = self.sigmas[i]
        sigma_next = self.sigmas[i + 1]

        s_in = sample.new_ones([sample.shape[0]])
        a_s, b_s, c_s = [
            self._append_dims(item, sample.ndim) for item in self.get_abc(sigma * s_in)
        ]
        a_t, b_t, c_t = [
            self._append_dims(item, sample.ndim)
            for item in self.get_abc(sigma_next * s_in)
        ]
        alpha_t, _, rho_t, _ = [
            self._append_dims(item, sample.ndim)
            for item in self.get_alpha_rho(sigma_next * s_in)
        ]
        _, _, rho_s, _ = [
            self._append_dims(item, sample.ndim) for item in self.get_alpha_rho(sigma * s_in)
        ]

        ratio = torch.clamp(1.0 - (rho_t**2 / torch.clamp(rho_s**2, min=1e-20)), min=0.0)
        omega_st = eta_val * (alpha_t * rho_t) * torch.sqrt(ratio)
        tmp_var = torch.sqrt(torch.clamp(c_t**2 - omega_st**2, min=0.0)) / torch.clamp(
            c_s, min=1e-20
        )

        coeff_xs = tmp_var
        coeff_x0_hat = b_t - tmp_var * b_s
        coeff_xT = a_t - tmp_var * a_s

        noise = randn_tensor(
            sample.shape,
            generator=generator,
            device=sample.device,
            dtype=sample.dtype,
        )
        is_last = i == len(self.sigmas) - 2
        prev_sample = (
            coeff_x0_hat * model_output
            + coeff_xT * x_T
            + coeff_xs * sample
            + (0.0 if is_last else 1.0) * omega_st * noise
        )

        if not return_dict:
            return (prev_sample, model_output)
        return DBIMSchedulerOutput(
            prev_sample=prev_sample,
            pred_original_sample=model_output,
        )

    def scale_model_input(
        self,
        sample: torch.Tensor,
        timestep: Optional[int] = None,
    ) -> torch.Tensor:
        """Compatibility hook with schedulers that scale model inputs."""
        return sample
