# Copyright (c) 2026 EarthBridge Team.
# Credits: Adapted from E3Diff (Qin et al., IEEE GRSL 2024).

"""Diffusion pipeline for E3Diff.

Includes:

* ``GaussianDiffusion`` – DDPM / DDIM diffusion wrapper for :class:`E3DiffUNet`.
* ``_make_beta_schedule`` – Beta schedule factory (linear, cosine, etc.).
"""

from __future__ import annotations

import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn

from examples.community.e3diff.model import E3DiffUNet, _ResnetBlocWithAttn, _default


# ---------------------------------------------------------------------------
# Beta schedule
# ---------------------------------------------------------------------------


def _make_beta_schedule(
    schedule: str,
    n_timestep: int,
    linear_start: float = 1e-4,
    linear_end: float = 2e-2,
    cosine_s: float = 8e-3,
) -> np.ndarray:
    """Create a beta schedule for the diffusion process."""
    if schedule == "quad":
        betas = np.linspace(linear_start**0.5, linear_end**0.5, n_timestep, dtype=np.float64) ** 2
    elif schedule == "linear":
        betas = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64)
    elif schedule == "warmup10":
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
        warmup = int(n_timestep * 0.1)
        betas[:warmup] = np.linspace(linear_start, linear_end, warmup, dtype=np.float64)
    elif schedule == "warmup50":
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
        warmup = int(n_timestep * 0.5)
        betas[:warmup] = np.linspace(linear_start, linear_end, warmup, dtype=np.float64)
    elif schedule == "const":
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == "jsd":
        betas = 1.0 / np.linspace(n_timestep, 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas_t = 1 - alphas[1:] / alphas[:-1]
        betas = betas_t.clamp(max=0.999).numpy()
    else:
        raise NotImplementedError(f"Beta schedule '{schedule}' is not implemented.")
    return betas


# ---------------------------------------------------------------------------
# GaussianDiffusion
# ---------------------------------------------------------------------------


class GaussianDiffusion(nn.Module):
    """DDPM / DDIM diffusion wrapper for E3DiffUNet.

    Supports two training stages:

    * **Stage 1** (``stage=1``): standard DDPM – randomly sample a timestep,
      add noise, predict the noise with the U-Net, compute L1 loss.
    * **Stage 2** (``stage=2``): one-step DDIM refinement – use a single DDIM
      step with the pre-trained Stage-1 model; return the predicted ``x_0``
      for GAN-based fine-tuning.

    Parameters
    ----------
    denoise_fn : E3DiffUNet
        The denoising network.
    image_size : int
        Spatial resolution.
    channels : int
        Number of channels in the output (optical) image.
    loss_type : ``"l1"`` | ``"l2"``
        Pixel-level loss type.
    conditional : bool
        If ``True``, the model takes a conditioning image concatenated with the
        noisy target.
    xT_noise_r : float
        Mixing ratio for DDPM-inversion initialisation in Stage 1 sampling
        (0 = pure random noise, 1 = fully inverted).
    """

    def __init__(
        self,
        denoise_fn: E3DiffUNet,
        image_size: int = 256,
        channels: int = 3,
        loss_type: str = "l1",
        conditional: bool = True,
        xT_noise_r: float = 0.1,
    ) -> None:
        super().__init__()
        self.denoise_fn = denoise_fn
        self.channels = channels
        self.image_size = image_size
        self.loss_type = loss_type
        self.conditional = conditional
        self.xT_noise_r = xT_noise_r

        if loss_type == "l1":
            self.loss_func = nn.L1Loss(reduction="sum")
        elif loss_type == "l2":
            self.loss_func = nn.MSELoss(reduction="sum")
        else:
            raise NotImplementedError(f"Loss type '{loss_type}' is not supported.")

        # Noise schedule buffers – populated by set_noise_schedule()
        self.num_timesteps: int = 0
        self.num_train_timesteps: int = 0
        self._schedule_ready: bool = False

    # ------------------------------------------------------------------
    # Noise schedule
    # ------------------------------------------------------------------

    def set_noise_schedule(
        self,
        n_timestep: int = 1000,
        schedule: str = "linear",
        linear_start: float = 1e-6,
        linear_end: float = 1e-2,
        device: torch.device | str = "cpu",
    ) -> None:
        """Initialise / update the noise schedule buffers.

        This must be called before training or sampling.
        """
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        betas = _make_beta_schedule(
            schedule=schedule,
            n_timestep=n_timestep,
            linear_start=linear_start,
            linear_end=linear_end,
        )
        betas = np.asarray(betas, dtype=np.float64)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(np.append(1.0, alphas_cumprod))

        self.num_timesteps = int(betas.shape[0])
        self.num_train_timesteps = self.num_timesteps

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod)))
        self.register_buffer("sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod)))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1)))
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        self.register_buffer(
            "posterior_log_variance_clipped", to_torch(np.log(np.maximum(posterior_variance, 1e-20)))
        )
        self.register_buffer(
            "posterior_mean_coef1",
            to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            to_torch((1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)),
        )
        self._schedule_ready = True

    # ------------------------------------------------------------------
    # Forward / reverse diffusion helpers
    # ------------------------------------------------------------------

    def _q_sample(
        self,
        x_start: torch.Tensor,
        continuous_sqrt_alpha_cumprod: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        noise = _default(noise, lambda: torch.randn_like(x_start))
        return continuous_sqrt_alpha_cumprod * x_start + (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise

    def _predict_x0(self, x_t: torch.Tensor, t: int, noise: torch.Tensor) -> torch.Tensor:
        return self.sqrt_recip_alphas_cumprod[t] * x_t - self.sqrt_recipm1_alphas_cumprod[t] * noise

    def _p_mean_variance(
        self, x: torch.Tensor, t: int, condition_x: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = x.shape[0]
        noise_level = torch.FloatTensor([self.sqrt_alphas_cumprod_prev[t + 1]]).repeat(b, 1).to(x.device)
        inp = torch.cat([condition_x, x], dim=1) if condition_x is not None else x
        noise_pred = self.denoise_fn(inp, noise_level)
        x_recon = self._predict_x0(x, t, noise_pred).clamp(-1.0, 1.0)
        mean = self.posterior_mean_coef1[t] * x_recon + self.posterior_mean_coef2[t] * x
        log_var = self.posterior_log_variance_clipped[t]
        return mean, log_var, x_recon

    @torch.no_grad()
    def _p_sample(self, x: torch.Tensor, t: int, condition_x: torch.Tensor | None = None):
        mean, log_var, x_recon = self._p_mean_variance(x, t, condition_x)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        return mean + noise * (0.5 * log_var).exp(), x_recon

    @torch.no_grad()
    def ddim_sample_onestep(
        self,
        condition_x: torch.Tensor,
        shape: tuple[int, ...],
        eta: float = 0.8,
    ) -> torch.Tensor:
        """Run a single DDIM step starting from t=T-1 → 0.

        Returns the predicted ``x_0`` (de-noised optical image).
        """
        device = self.betas.device
        total_t = self.num_train_timesteps

        # Initialise noisy image
        x = torch.randn(shape, device=device)
        if self.xT_noise_r > 0 and condition_x is not None:
            x_cond = condition_x[:, : self.channels, ...]
            sqrt_a = torch.FloatTensor([self.sqrt_alphas_cumprod_prev[total_t - 1]]).to(device)
            x_noisy = sqrt_a * x_cond + (1 - sqrt_a**2).sqrt() * torch.randn_like(x_cond)
            x = self.xT_noise_r * x_noisy + (1 - self.xT_noise_r) * x

        cur_t = total_t - 1
        prev_t = 0

        noise_level = (
            torch.FloatTensor([self.sqrt_alphas_cumprod_prev[cur_t]]).repeat(shape[0], 1).to(device)
        )
        alpha_t = self.alphas_cumprod[cur_t]
        alpha_prev = self.alphas_cumprod[prev_t]
        beta_t = 1 - alpha_t
        sigma2 = eta * (1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)

        inp = torch.cat([condition_x, x], dim=1)
        noise_pred = self.denoise_fn(inp, noise_level)

        x0_pred = ((x - beta_t**0.5 * noise_pred) / alpha_t**0.5).clamp(-1.0, 1.0)
        pred_dir = (1 - alpha_prev - sigma2) ** 0.5 * noise_pred
        x_prev = alpha_prev**0.5 * x0_pred + pred_dir + sigma2**0.5 * torch.randn_like(x)
        return x_prev, x0_pred

    # ------------------------------------------------------------------
    # Training objective
    # ------------------------------------------------------------------

    def p_losses(
        self,
        data: dict[str, torch.Tensor],
        stage: int = 1,
        ddim_steps: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the diffusion training loss.

        Parameters
        ----------
        data : dict with keys ``"HR"`` (optical target) and ``"SR"`` (SAR condition).
        stage : int
            ``1`` for noise-prediction, ``2`` for one-step DDIM prediction.
        ddim_steps : int
            Number of DDIM steps used in Stage 2 (typically 1).

        Returns
        -------
        l_pix : Tensor, scalar
            Pixel-level loss.
        x_start : Tensor ``[B, C, H, W]``
            Ground-truth optical image.
        x_pred : Tensor ``[B, C, H, W]``
            Predicted image (predicted noise in Stage 1, predicted x0 in Stage 2).
        """
        x_start = data["HR"]
        x_cond = data["SR"]
        b, c, h, w = x_start.shape
        device = x_start.device

        if stage == 2:
            shape = (b, self.channels, h, w)
            _, x_pred = self.ddim_sample_onestep(x_cond, shape)
            l_pix = self.loss_func(x_start, x_pred)
        else:
            t = int(np.random.randint(1, self.num_timesteps + 1))
            sqrt_a_prev = self.sqrt_alphas_cumprod_prev[t - 1]
            sqrt_a = self.sqrt_alphas_cumprod_prev[t]
            cont_sqrt = torch.FloatTensor(
                np.random.uniform(sqrt_a_prev, sqrt_a, size=b)
            ).to(device)
            cont_sqrt = cont_sqrt.view(b, -1)

            noise = torch.randn_like(x_start)
            x_noisy = self._q_sample(x_start, cont_sqrt.view(-1, 1, 1, 1), noise)

            inp = torch.cat([x_cond, x_noisy], dim=1)
            x_pred = self.denoise_fn(inp, cont_sqrt)
            l_pix = self.loss_func(noise, x_pred)

        return l_pix, x_start, x_pred

    def forward(self, data: dict[str, torch.Tensor], stage: int = 1) -> tuple:
        return self.p_losses(data, stage=stage)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        n_ddim_steps: int = 50,
        eta: float = 0.8,
    ) -> torch.Tensor:
        """Sample an optical image conditioned on a SAR image.

        Parameters
        ----------
        condition : Tensor ``[B, condition_ch, H, W]``
            SAR / conditioning image.
        n_ddim_steps : int
            Number of DDIM denoising steps.
        eta : float
            DDIM stochasticity parameter (0 = deterministic, 1 = DDPM).

        Returns
        -------
        Tensor ``[B, channels, H, W]``
            Predicted optical image in [-1, 1].
        """
        device = next(self.parameters()).device
        b, _, h, w = condition.shape
        shape = (b, self.channels, h, w)
        total_t = self.num_train_timesteps

        x = torch.randn(shape, device=device)
        ts = torch.linspace(total_t, 0, n_ddim_steps + 1, dtype=torch.long, device=device)

        for i in range(1, n_ddim_steps + 1):
            cur_t = int(ts[i - 1].item()) - 1
            prev_t = int(ts[i].item()) - 1
            noise_level = (
                torch.FloatTensor([self.sqrt_alphas_cumprod_prev[cur_t]]).repeat(b, 1).to(device)
            )
            alpha_t = self.alphas_cumprod[cur_t]
            alpha_prev = self.alphas_cumprod[max(prev_t, 0)]
            beta_t = 1 - alpha_t
            sigma2 = eta * (1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)

            inp = torch.cat([condition, x], dim=1)
            noise_pred = self.denoise_fn(inp, noise_level)
            x0_pred = ((x - beta_t**0.5 * noise_pred) / alpha_t**0.5).clamp(-1.0, 1.0)
            pred_dir = (1 - alpha_prev - sigma2) ** 0.5 * noise_pred
            x = alpha_prev**0.5 * x0_pred + pred_dir + sigma2**0.5 * torch.randn_like(x)

        return x
