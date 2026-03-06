# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""CDTSDE Scheduler – Adaptive Domain Shift Diffusion Bridge.

Implements the conditional diffusion process with dynamic domain-shift
scheduling for image-to-image translation.

Reference
---------
CDTSDE (ICLR 2026).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor


def _make_beta_schedule_linear_sqrt(
    n_timestep: int,
    linear_start: float,
    linear_end: float,
) -> np.ndarray:
    """Linear-in-sqrt beta schedule used by latent diffusion variants."""
    betas = torch.linspace(
        linear_start ** 0.5,
        linear_end ** 0.5,
        n_timestep,
        dtype=torch.float64,
    ) ** 2
    return betas.numpy()


def make_eta_schedule(
    schedule: str,
    n_timestep: int,
    linear_start: float = 0.001,
    linear_end: float = 0.999,
    return_t1: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, int]]:
    """Build CDTSDE eta schedule."""
    if schedule == "trunc_23":
        trunc_t = n_timestep * 2 // 3
    elif schedule == "trunc_13":
        trunc_t = n_timestep // 3
    elif schedule == "trunc_12":
        trunc_t = n_timestep // 2
    elif schedule == "full":
        trunc_t = n_timestep
    else:
        raise ValueError(f"Unknown eta schedule: {schedule}")

    etas = torch.ones(n_timestep, dtype=torch.float64)
    times = torch.linspace(linear_start, linear_end, trunc_t, dtype=torch.float64)
    etas[:trunc_t] = (1.0 - torch.cos(times * np.pi)) / 2.0
    etas_np = etas.numpy()
    if return_t1:
        return etas_np, trunc_t
    return etas_np


def space_timesteps(num_timesteps: int, section_counts: Union[str, int]) -> set[int]:
    """Create spaced timesteps."""
    if isinstance(section_counts, int):
        section_counts = str(section_counts)

    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim"):])
            for stride in range(1, num_timesteps):
                if len(range(0, num_timesteps, stride)) == desired_count:
                    return set(range(0, num_timesteps, stride))
            raise ValueError(f"Cannot create exactly {desired_count} steps with integer stride.")
        section_counts_list = [int(x) for x in section_counts.split(",")]
    else:
        section_counts_list = [int(section_counts)]

    size_per = num_timesteps // len(section_counts_list)
    extra = num_timesteps % len(section_counts_list)
    start_idx = 0
    all_steps: list[int] = []

    for i, section_count in enumerate(section_counts_list):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(f"Cannot divide section of {size} steps into {section_count}")

        frac_stride = 1.0 if section_count <= 1 else (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps: list[int] = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride

        all_steps.extend(taken_steps)
        start_idx += size

    return set(all_steps)


def _extract_into_tensor(arr: torch.Tensor, timesteps: torch.Tensor, broadcast_shape: Tuple[int, ...]) -> torch.Tensor:
    """Gather 1-D tensor values at ``timesteps`` and expand to ``broadcast_shape``."""
    res = arr.to(device=timesteps.device, dtype=torch.float32)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


@dataclass
class CDTSDESchedulerOutput(BaseOutput):
    """Output of a single CDTSDE scheduler step.

    Attributes
    ----------
    prev_sample : torch.Tensor
        Sample after one reverse step.
    pred_original_sample : torch.Tensor or None
        Predicted clean sample ``x_0``.
    """

    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None


class CDTSDEScheduler(SchedulerMixin, ConfigMixin):
    """CDTSDE scheduler with dynamic shift-oriented reverse updates.

    Parameters
    ----------
    num_train_timesteps : int
        Number of training diffusion steps.
    beta_schedule : str
        Only ``"linear"`` is supported.
    beta_start, beta_end : float
        Beta schedule endpoints.
    eta_schedule : str
        One of ``"trunc_12"``, ``"trunc_13"``, ``"trunc_23"``, ``"full"``.
    eta_start, eta_end : float
        Eta schedule endpoints.
    lambda_eps : float
        Epsilon for numerical stability.
    """

    _compatibles = []
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_schedule: str = "linear",
        beta_start: float = 0.00085,
        beta_end: float = 0.0120,
        eta_schedule: str = "trunc_12",
        eta_start: float = 0.001,
        eta_end: float = 0.999,
        lambda_eps: float = 1e-6,
    ):
        if beta_schedule != "linear":
            raise ValueError("Only beta_schedule='linear' is supported for CDTSDE.")

        self.lambda_eps = float(lambda_eps)
        self.init_noise_sigma = 1.0

        train_betas = _make_beta_schedule_linear_sqrt(
            n_timestep=num_train_timesteps,
            linear_start=beta_start,
            linear_end=beta_end,
        )
        train_alphas = 1.0 - train_betas
        train_alphas_cumprod = np.cumprod(train_alphas, axis=0)

        self.train_alphas_cumprod = torch.tensor(train_alphas_cumprod, dtype=torch.float32)
        self.train_sqrt_alphas_cumprod = torch.sqrt(self.train_alphas_cumprod)
        self.train_sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.train_alphas_cumprod)
        self.train_sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.train_alphas_cumprod)
        self.train_sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.train_alphas_cumprod - 1.0)

        # Inference-time schedule (set by set_timesteps).
        self.num_inference_steps: Optional[int] = None
        self.timesteps: Optional[torch.Tensor] = None
        self.alphas_cumprod: Optional[torch.Tensor] = None
        self.sqrt_alphas_cumprod: Optional[torch.Tensor] = None
        self.sqrt_one_minus_alphas_cumprod: Optional[torch.Tensor] = None
        self.sqrt_recip_alphas_cumprod: Optional[torch.Tensor] = None
        self.sqrt_recipm1_alphas_cumprod: Optional[torch.Tensor] = None
        self.etas: Optional[torch.Tensor] = None
        self.one_minus_etas: Optional[torch.Tensor] = None
        self.alphas_multi_one_minus_etas: Optional[torch.Tensor] = None
        self.sigmas: Optional[torch.Tensor] = None
        self.lambdas: Optional[torch.Tensor] = None

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Union[str, torch.device, None] = None,
    ) -> None:
        """Set reduced CDTSDE timesteps and derived coefficients."""
        self.num_inference_steps = int(num_inference_steps)

        full_etas, trunc_t = make_eta_schedule(
            schedule=self.config.eta_schedule,
            n_timestep=self.config.num_train_timesteps,
            linear_start=self.config.eta_start,
            linear_end=self.config.eta_end,
            return_t1=True,
        )
        used_timesteps = space_timesteps(trunc_t, str(num_inference_steps + 1))
        used_sorted = sorted(list(used_timesteps))

        train_alpha_cumprod_np = self.train_alphas_cumprod.cpu().numpy().astype(np.float64)
        reduced_betas: list[float] = []
        reduced_etas: list[float] = []
        last_alpha_cumprod = 1.0
        for i, alpha_cumprod in enumerate(train_alpha_cumprod_np):
            if i in used_timesteps:
                reduced_betas.append(1.0 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                reduced_etas.append(float(full_etas[i]))

        if len(reduced_betas) != num_inference_steps + 1:
            raise ValueError(
                f"Expected {num_inference_steps + 1} reduced steps, got {len(reduced_betas)}."
            )

        betas = torch.tensor(reduced_betas, dtype=torch.float32, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        etas = torch.tensor(reduced_etas, dtype=torch.float32, device=device).clamp(0.0, 1.0)
        one_minus_etas = (1.0 - etas).clamp(min=self.lambda_eps)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sigmas = torch.sqrt(1.0 - alphas_cumprod)
        alphas_multi_one_minus_etas = sqrt_alphas_cumprod * one_minus_etas
        lambdas = sigmas / alphas_multi_one_minus_etas.clamp(min=self.lambda_eps)

        self.timesteps = torch.tensor(used_sorted, dtype=torch.long, device=device)
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = sqrt_alphas_cumprod
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1.0)
        self.etas = etas
        self.one_minus_etas = one_minus_etas
        self.alphas_multi_one_minus_etas = alphas_multi_one_minus_etas
        self.sigmas = sigmas
        self.lambdas = lambdas
        self.init_noise_sigma = float(sigmas[-1].item())

    def scale_model_input(
        self,
        sample: torch.Tensor,
        timestep: Optional[int] = None,
    ) -> torch.Tensor:
        """Identity scaling for model input compatibility."""
        return sample

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        reference_samples: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward diffusion using full training schedule."""
        sqrt_alpha_t = _extract_into_tensor(
            self.train_sqrt_alphas_cumprod,
            timesteps,
            original_samples.shape,
        )
        sqrt_one_minus_alpha_t = _extract_into_tensor(
            self.train_sqrt_one_minus_alphas_cumprod,
            timesteps,
            original_samples.shape,
        )
        return sqrt_alpha_t * original_samples + sqrt_one_minus_alpha_t * noise

    def predict_start_from_noise(
        self,
        sample: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
        use_inference_schedule: bool = False,
    ) -> torch.Tensor:
        """Recover ``x_0`` from noisy sample and predicted epsilon."""
        if use_inference_schedule:
            if self.sqrt_recip_alphas_cumprod is None or self.sqrt_recipm1_alphas_cumprod is None:
                raise ValueError("Inference schedule is not initialized. Call set_timesteps first.")
            sqrt_recip = _extract_into_tensor(
                self.sqrt_recip_alphas_cumprod, timesteps, sample.shape
            )
            sqrt_recipm1 = _extract_into_tensor(
                self.sqrt_recipm1_alphas_cumprod, timesteps, sample.shape
            )
        else:
            sqrt_recip = _extract_into_tensor(
                self.train_sqrt_recip_alphas_cumprod, timesteps, sample.shape
            )
            sqrt_recipm1 = _extract_into_tensor(
                self.train_sqrt_recipm1_alphas_cumprod, timesteps, sample.shape
            )
        return sqrt_recip * sample - sqrt_recipm1 * noise

    def step(
        self,
        pred_original_sample: torch.Tensor,
        step_index: int,
        sample: torch.Tensor,
        reference_sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        stochastic: bool = True,
        return_dict: bool = True,
    ) -> Union[CDTSDESchedulerOutput, Tuple[torch.Tensor, torch.Tensor]]:
        """One CDTSDE reverse step from ``j`` to ``j-1``."""
        if self.lambdas is None or self.alphas_multi_one_minus_etas is None or self.etas is None:
            raise ValueError("Inference schedule is not initialized. Call set_timesteps first.")
        if step_index <= 0:
            if return_dict:
                return CDTSDESchedulerOutput(
                    prev_sample=pred_original_sample,
                    pred_original_sample=pred_original_sample,
                )
            return pred_original_sample, pred_original_sample

        j = int(step_index)
        dtype = sample.dtype
        device = sample.device
        lambdas = self.lambdas.to(device=device, dtype=dtype)
        amoe = self.alphas_multi_one_minus_etas.to(device=device, dtype=dtype)
        etas = self.etas.to(device=device, dtype=dtype)
        eps = torch.as_tensor(self.lambda_eps, device=device, dtype=dtype)

        lambda_prev = lambdas[j - 1]
        lambda_cur = lambdas[j].clamp(min=eps)
        r_lambdas = lambda_prev / lambda_cur

        amoe_prev = amoe[j - 1]
        amoe_cur = amoe[j].clamp(min=eps)
        r1 = amoe_prev / amoe_cur

        eta_prev = etas[j - 1]
        eta_cur = etas[j]
        one_prev = (1.0 - eta_prev).clamp(min=eps)
        one_cur = (1.0 - eta_cur).clamp(min=eps)

        rec_ref = amoe_prev * (
            eta_prev / one_prev - eta_cur / one_cur * (r_lambdas ** 2)
        ) * reference_sample
        rec_pred = amoe_prev * (1.0 - r_lambdas ** 2) * pred_original_sample

        noise_term = torch.zeros(1, device=device, dtype=dtype)
        if stochastic:
            sigma_coeff_sq = (lambda_prev ** 2) - (lambda_prev ** 2) * (r_lambdas ** 2)
            sigma_coeff = torch.sqrt(torch.clamp(sigma_coeff_sq, min=0.0))
            noise = randn_tensor(
                sample.shape,
                generator=generator,
                device=device,
                dtype=dtype,
            )
            noise_term = noise * sigma_coeff * amoe_prev

        prev_sample = r1 * (r_lambdas ** 2) * sample + rec_ref + rec_pred + noise_term

        if not return_dict:
            return prev_sample, pred_original_sample
        return CDTSDESchedulerOutput(
            prev_sample=prev_sample,
            pred_original_sample=pred_original_sample,
        )
