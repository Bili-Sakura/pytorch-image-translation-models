# Copyright (c) 2026 EarthBridge Team.
# Credits: Adapted from E3Diff (Qin et al., IEEE GRSL 2024).

"""Diffusion pipeline for E3Diff.

Includes:

* ``E3DiffPipeline`` – Inference pipeline inheriting from ``DiffusionPipeline``.
* ``E3DiffPipelineOutput`` – Dataclass for pipeline output.
* ``GaussianDiffusion`` – DDPM / DDIM diffusion wrapper for :class:`E3DiffUNet`.
* ``_make_beta_schedule`` – Beta schedule factory (linear, cosine, etc.).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from typing import Any, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from diffusers import DiffusionPipeline
from diffusers.utils import BaseOutput

from examples.community.e3diff.model import E3DiffUNet, _default


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


# ---------------------------------------------------------------------------
# E3DiffPipeline – DiffusionPipeline-based inference wrapper
# ---------------------------------------------------------------------------


@dataclass
class E3DiffPipelineOutput(BaseOutput):
    """Output of the E3Diff pipeline.

    Attributes
    ----------
    images : list[PIL.Image.Image] | np.ndarray | torch.Tensor
        Translated images in the requested format.
    nfe : int
        Number of function evaluations (denoising steps) used.
    """

    images: Any
    nfe: int = 0


class E3DiffPipeline(DiffusionPipeline):
    """Image-to-image translation pipeline using E3Diff conditional diffusion.

    Wraps a :class:`GaussianDiffusion` model (which contains the
    :class:`E3DiffUNet` and noise schedule) with a standard
    ``DiffusionPipeline`` interface for inference.

    Inherits from :class:`~diffusers.DiffusionPipeline` so that checkpoints
    can be loaded via ``from_pretrained`` following the HuggingFace
    *diffusers* convention.

    Parameters
    ----------
    diffusion : GaussianDiffusion
        A ``GaussianDiffusion`` instance with the noise schedule already set
        via :meth:`GaussianDiffusion.set_noise_schedule`.

    Example
    -------
    ::

        from examples.community.e3diff import (
            E3DiffUNet, GaussianDiffusion, E3DiffPipeline,
        )

        unet = E3DiffUNet(...)
        diff = GaussianDiffusion(denoise_fn=unet, image_size=256, channels=3)
        diff.set_noise_schedule(n_timestep=1000, schedule="linear", device="cpu")
        pipeline = E3DiffPipeline(diffusion=diff)
        output = pipeline(source_image=sar_tensor, num_inference_steps=50)
    """

    model_cpu_offload_seq = "diffusion"

    def __init__(self, diffusion: GaussianDiffusion) -> None:
        super().__init__()
        self.register_modules(diffusion=diffusion)

    @property
    def device(self) -> torch.device:
        """Get the device of the pipeline."""
        return next(self.diffusion.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Get the dtype of the pipeline."""
        return next(self.diffusion.parameters()).dtype

    def prepare_inputs(
        self,
        image: Union[torch.Tensor, Image.Image, List[Image.Image]],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Prepare input images for the pipeline.

        Converts PIL images or numpy arrays to normalised tensors in
        ``[-1, 1]`` range.
        """
        if isinstance(image, Image.Image):
            image = [image]

        if isinstance(image, list) and isinstance(image[0], Image.Image):
            images = []
            for img in image:
                img_array = np.array(img, dtype=np.float32)
                if img_array.max() > 1.0:
                    img_array = img_array / 255.0
                if img_array.ndim == 2:
                    img_array = img_array[:, :, np.newaxis]
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
                images.append(img_tensor)
            image = torch.stack(images)

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)

        # Ensure image is in [-1, 1] range
        if image.min() >= 0 and image.max() <= 1.0:
            image = image * 2 - 1  # [0, 1] → [-1, 1]
        elif image.max() > 1.0:
            image = image / 255.0 * 2 - 1  # [0, 255] → [-1, 1]

        return image.to(device=device, dtype=dtype)

    @torch.no_grad()
    def __call__(
        self,
        source_image: Union[torch.Tensor, Image.Image, List[Image.Image]],
        num_inference_steps: int = 50,
        eta: float = 0.8,
        output_type: str = "pil",
        return_dict: bool = True,
    ) -> Union[E3DiffPipelineOutput, tuple]:
        """Run E3Diff DDIM sampling to translate a conditioning (SAR) image.

        Parameters
        ----------
        source_image : Tensor | PIL.Image | list[PIL.Image]
            Source (SAR / conditioning) image(s).  Tensors should be
            ``(B, C, H, W)`` in ``[0, 1]`` or ``[-1, 1]`` range.
        num_inference_steps : int
            Number of DDIM denoising steps.
        eta : float
            DDIM stochasticity parameter (0 = deterministic, 1 = DDPM).
        output_type : str
            ``"pil"``, ``"np"``, or ``"pt"`` (default ``"pil"``).
        return_dict : bool
            If ``True``, return an :class:`E3DiffPipelineOutput`.

        Returns
        -------
        E3DiffPipelineOutput or tuple
            Translated images and number of function evaluations.
        """
        device = self.device
        dtype = self.dtype

        condition = self.prepare_inputs(source_image, device, dtype)

        self.diffusion.eval()
        images = self.diffusion.sample(
            condition, n_ddim_steps=num_inference_steps, eta=eta,
        )
        images = images.clamp(-1, 1)

        nfe = num_inference_steps

        if output_type == "pil":
            images = self._convert_to_pil(images)
        elif output_type == "np":
            images = self._convert_to_numpy(images)

        if not return_dict:
            return (images, nfe)
        return E3DiffPipelineOutput(images=images, nfe=nfe)

    @staticmethod
    def _convert_to_pil(images: torch.Tensor) -> List[Image.Image]:
        """Convert tensor in [-1, 1] to PIL images."""
        images = (images + 1) / 2  # [-1, 1] → [0, 1]
        images = images.clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).round().astype(np.uint8)
        pil_images = []
        for img in images:
            if img.shape[2] == 1:
                pil_images.append(Image.fromarray(img.squeeze(2), mode="L"))
            else:
                pil_images.append(Image.fromarray(img))
        return pil_images

    @staticmethod
    def _convert_to_numpy(images: torch.Tensor) -> np.ndarray:
        """Convert tensor in [-1, 1] to numpy array in [0, 1]."""
        images = (images + 1) / 2
        images = images.clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        return images
