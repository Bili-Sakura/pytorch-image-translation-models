# Copyright (c) 2026 EarthBridge Team.
# Credits: See upstream/paper attribution below and README.md citations.

# Copyright 2024 The DDBM Authors and The Hugging Face Team.
# Licensed under the Apache License, Version 2.0 (the "License");
#
# Self-contained DDBM pipeline for production inference - no external project code.
# Use as custom_pipeline: DiffusionPipeline.from_pretrained(ckpt_path, custom_pipeline="examples/pipelines/ddbm")
#
# Components: DDBMUNet, DDBMScheduler, DDBMPipeline

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from diffusers import DiffusionPipeline, ModelMixin, UNet2DModel
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_utils import SchedulerMixin


# ---------------------------------------------------------------------------
# DDBMUNet – self-contained, no external imports
# ---------------------------------------------------------------------------


def _build_block_types(
    channel_mult: Tuple[int, ...],
    attention_resolutions: Tuple[int, ...],
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Build down_block_types and up_block_types from channel_mult and attention indices."""
    down_block_types = []
    for i in range(len(channel_mult)):
        if i in attention_resolutions:
            down_block_types.append("AttnDownBlock2D")
        else:
            down_block_types.append("DownBlock2D")
    up_block_types = []
    for i in range(len(channel_mult)):
        if (len(channel_mult) - 1 - i) in attention_resolutions:
            up_block_types.append("AttnUpBlock2D")
        else:
            up_block_types.append("UpBlock2D")
    return tuple(down_block_types), tuple(up_block_types)


def _channel_mult_for_resolution(resolution: int) -> Tuple[int, ...]:
    """Return a sensible default channel multiplier tuple.

    ADM-style: 256px=4 stages (256→16), 512px=5 stages (512→16), 1024px=6 stages.
    """
    return {
        1024: (1, 1, 2, 2, 4, 4),
        512: (1, 2, 4, 4, 8),
        256: (1, 2, 2, 4),
        128: (1, 1, 2, 3, 4),
        64: (1, 2, 3, 4),
        32: (1, 2, 3, 4),
    }.get(resolution, (1, 2, 3, 4))


class DDBMUNet(ModelMixin, ConfigMixin):
    """Wrapper around UNet2DModel that accepts the DDBM calling convention (x, timestep, xT=…)."""

    @register_to_config
    def __init__(
        self,
        image_size: int = 256,
        in_channels: int = 3,
        model_channels: int = 128,
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (1,),
        dropout: float = 0.0,
        condition_mode: Optional[str] = "concat",
        channel_mult: Optional[Tuple[int, ...]] = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.condition_mode = condition_mode

        if channel_mult is None:
            channel_mult = _channel_mult_for_resolution(image_size)

        unet_in_channels = in_channels * 2 if condition_mode == "concat" else in_channels
        block_out_channels = tuple(model_channels * m for m in channel_mult)
        down_block_types, up_block_types = _build_block_types(channel_mult, attention_resolutions)

        self.unet = UNet2DModel(
            sample_size=image_size,
            in_channels=unet_in_channels,
            out_channels=in_channels,
            block_out_channels=block_out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            layers_per_block=num_res_blocks,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        xT: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.condition_mode == "concat" and xT is not None:
            x = torch.cat([x, xT], dim=1)
        return self.unet(x, timestep).sample

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Load UNet, preferring ema_unet weights when subfolder is ema_unet.
        If ema_unet/ has no config.json, load config from unet/ and weights from ema_unet/.
        """
        from pathlib import Path

        path = Path(pretrained_model_name_or_path)
        subfolder = kwargs.get("subfolder", "unet")
        if subfolder == "ema_unet" and not (path / "ema_unet" / "config.json").exists():
            # ema_unet has no config; load config from unet, weights from ema_unet
            unet = super().from_pretrained(path, subfolder="unet", **{k: v for k, v in kwargs.items() if k != "subfolder"})
            from safetensors.torch import load_file
            ema_path = path / "ema_unet" / "diffusion_pytorch_model.safetensors"
            if ema_path.exists():
                state = load_file(str(ema_path))
                unet.load_state_dict(state, strict=True)
            return unet
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)


# ---------------------------------------------------------------------------
# DDBMScheduler
# ---------------------------------------------------------------------------


@dataclass
class DDBMSchedulerOutput(BaseOutput):
    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None


class DDBMScheduler(SchedulerMixin, ConfigMixin):
    """Scheduler for Denoising Diffusion Bridge Models (DDBM)."""

    _compatibles = []
    order = 2

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
    ):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.rho = rho
        self.pred_mode = pred_mode
        self.num_train_timesteps = num_train_timesteps
        self.sigmas: Optional[torch.Tensor] = None
        self.timesteps: Optional[torch.Tensor] = None
        self.num_inference_steps: Optional[int] = None
        self.init_noise_sigma = sigma_max
        self._init_vp_functions()

    def _init_vp_functions(self):
        beta_d, beta_min = self.beta_d, self.beta_min

        def vp_snr_sqrt_reciprocal(t):
            t_tensor = torch.as_tensor(t)
            return (torch.exp(0.5 * beta_d * (t_tensor ** 2) + beta_min * t_tensor) - 1) ** 0.5

        def vp_snr_sqrt_reciprocal_deriv(t):
            snr_sqrt_recip = vp_snr_sqrt_reciprocal(t)
            return 0.5 * (beta_min + beta_d * t) * (snr_sqrt_recip + 1 / snr_sqrt_recip)

        def s(t):
            return (1 + vp_snr_sqrt_reciprocal(t) ** 2).rsqrt()

        def s_deriv(t):
            return -vp_snr_sqrt_reciprocal(t) * vp_snr_sqrt_reciprocal_deriv(t) * (s(t) ** 3)

        def logs(t):
            return -0.25 * torch.as_tensor(t) ** 2 * beta_d - 0.5 * torch.as_tensor(t) * beta_min

        def std(t):
            return vp_snr_sqrt_reciprocal(t) * s(t)

        def logsnr(t):
            return -2 * torch.log(vp_snr_sqrt_reciprocal(t))

        self._vp_snr_sqrt_reciprocal = vp_snr_sqrt_reciprocal
        self._vp_snr_sqrt_reciprocal_deriv = vp_snr_sqrt_reciprocal_deriv
        self._s_deriv = s_deriv
        self._logs = logs
        self._std = std
        self._logsnr = logsnr

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Union[str, torch.device] = None,
    ):
        self.num_inference_steps = num_inference_steps
        ramp = torch.linspace(0, 1, num_inference_steps)
        min_inv_rho = self.sigma_min ** (1 / self.rho)
        max_inv_rho = (self.sigma_max - 1e-4) ** (1 / self.rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** self.rho
        sigmas = torch.cat([sigmas, torch.zeros(1)])
        self.sigmas = sigmas.to(device)
        self.timesteps = torch.arange(num_inference_steps, device=device)

    def _append_dims(self, x, target_dims):
        dims_to_append = target_dims - x.ndim
        if dims_to_append < 0:
            raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}")
        return x[(...,) + (None,) * dims_to_append]

    def scale_model_input(self, sample: torch.Tensor, timestep: Optional[int] = None) -> torch.Tensor:
        return sample


# ---------------------------------------------------------------------------
# DDBMPipeline
# ---------------------------------------------------------------------------


@dataclass
class DDBMPipelineOutput(BaseOutput):
    images: Union[List[Image.Image], np.ndarray, torch.Tensor]
    nfe: int = 0


class DDBMPipeline(DiffusionPipeline):
    """Pipeline for image-to-image generation using Denoising Diffusion Bridge Models."""

    model_cpu_offload_seq = "unet"

    def __init__(self, unet: DDBMUNet, scheduler: DDBMScheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)
        self.sigma_data = scheduler.config.sigma_data
        self.sigma_max = scheduler.config.sigma_max
        self.sigma_min = scheduler.config.sigma_min
        self.beta_d = scheduler.config.beta_d
        self.beta_min = scheduler.config.beta_min
        self.pred_mode = scheduler.config.pred_mode
        self.cov_xy = 0.0

    @property
    def device(self) -> torch.device:
        return next(self.unet.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.unet.parameters()).dtype

    def _vp_logsnr(self, t):
        t = torch.as_tensor(t)
        return -torch.log((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1)

    def _vp_logs(self, t):
        t = torch.as_tensor(t)
        return -0.25 * t ** 2 * self.beta_d - 0.5 * t * self.beta_min

    def _append_dims(self, x, target_dims):
        dims_to_append = target_dims - x.ndim
        if dims_to_append < 0:
            raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}")
        return x[(...,) + (None,) * dims_to_append]

    def _get_bridge_scalings(self, sigma):
        sigma_data = self.sigma_data
        sigma_data_end = sigma_data
        cov_xy = self.cov_xy
        c = 1

        if self.pred_mode == "ve":
            A = (
                sigma ** 4 / self.sigma_max ** 4 * sigma_data_end ** 2
                + (1 - sigma ** 2 / self.sigma_max ** 2) ** 2 * sigma_data ** 2
                + 2 * sigma ** 2 / self.sigma_max ** 2 * (1 - sigma ** 2 / self.sigma_max ** 2) * cov_xy
                + c ** 2 * sigma ** 2 * (1 - sigma ** 2 / self.sigma_max ** 2)
            )
            c_in = 1 / A ** 0.5
            c_skip = (
                (1 - sigma ** 2 / self.sigma_max ** 2) * sigma_data ** 2
                + sigma ** 2 / self.sigma_max ** 2 * cov_xy
            ) / A
            c_out = (
                (sigma / self.sigma_max) ** 4 * (sigma_data_end ** 2 * sigma_data ** 2 - cov_xy ** 2)
                + sigma_data ** 2 * c ** 2 * sigma ** 2 * (1 - sigma ** 2 / self.sigma_max ** 2)
            ) ** 0.5 * c_in
            return c_skip, c_out, c_in

        elif self.pred_mode == "vp":
            logsnr_t = self._vp_logsnr(sigma)
            logsnr_T = self._vp_logsnr(1)
            logs_t = self._vp_logs(sigma)
            logs_T = self._vp_logs(1)
            a_t = (logsnr_T - logsnr_t + logs_t - logs_T).exp()
            b_t = -torch.expm1(logsnr_T - logsnr_t) * logs_t.exp()
            c_t = -torch.expm1(logsnr_T - logsnr_t) * (2 * logs_t - logsnr_t).exp()
            A = a_t ** 2 * sigma_data_end ** 2 + b_t ** 2 * sigma_data ** 2 + 2 * a_t * b_t * cov_xy + c ** 2 * c_t
            c_in = 1 / A ** 0.5
            c_skip = (b_t * sigma_data ** 2 + a_t * cov_xy) / A
            c_out = (
                a_t ** 2 * (sigma_data_end ** 2 * sigma_data ** 2 - cov_xy ** 2)
                + sigma_data ** 2 * c ** 2 * c_t
            ) ** 0.5 * c_in
            return c_skip, c_out, c_in

        elif self.pred_mode in ["ve_simple", "vp_simple"]:
            c_in = torch.ones_like(sigma)
            c_out = torch.ones_like(sigma)
            c_skip = torch.zeros_like(sigma)
            return c_skip, c_out, c_in

        raise ValueError(f"Unknown pred_mode: {self.pred_mode}")

    def denoise(self, x_t, sigmas, x_T, clip_denoised=True):
        c_skip, c_out, c_in = [
            self._append_dims(x, x_t.ndim) for x in self._get_bridge_scalings(sigmas)
        ]
        rescaled_t = 1000 * 0.25 * torch.log(sigmas + 1e-44)
        model_output = self.unet(c_in * x_t, rescaled_t, xT=x_T)
        denoised = c_out * model_output + c_skip * x_t
        if clip_denoised:
            denoised = denoised.clamp(-1, 1)
        return denoised

    def prepare_inputs(
        self,
        image: Union[torch.Tensor, Image.Image, List[Image.Image]],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if isinstance(image, Image.Image):
            image = [image]
        if isinstance(image, list) and isinstance(image[0], Image.Image):
            images = []
            for img in image:
                # Keep L (grayscale) for 1-channel models; RGB for 3-channel
                if img.mode not in ("L", "RGB"):
                    img = img.convert("RGB")
                img_array = np.array(img).astype(np.float32) / 255.0
                if img_array.ndim == 2:
                    img_tensor = torch.from_numpy(img_array).unsqueeze(0)  # (1, H, W)
                else:
                    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
                images.append(img_tensor)
            image = torch.stack(images)
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        if image.max() > 1.0:
            image = image / 255.0
        if image.min() >= 0:
            image = image * 2 - 1
        return image.to(device=device, dtype=dtype)

    def _get_d_stochastic(self, x, sigma, denoised, x_T, guidance):
        if self.pred_mode == "ve":
            return self._get_d_ve(x, sigma, denoised, x_T, guidance, stochastic=True)
        elif self.pred_mode.startswith("vp"):
            return self._get_d_vp(x, sigma, denoised, x_T, guidance, stochastic=True)
        raise ValueError(f"Unknown pred_mode: {self.pred_mode}")

    def _get_d(self, x, sigma, denoised, x_T, guidance):
        if self.pred_mode == "ve":
            return self._get_d_ve(x, sigma, denoised, x_T, guidance, stochastic=False)
        elif self.pred_mode.startswith("vp"):
            return self._get_d_vp(x, sigma, denoised, x_T, guidance, stochastic=False)
        raise ValueError(f"Unknown pred_mode: {self.pred_mode}")

    def _get_d_ve(self, x, sigma, denoised, x_T, w, stochastic=False):
        grad_pxtlx0 = (denoised - x) / self._append_dims(sigma ** 2, x.ndim)
        grad_pxTlxt = (x_T - x) / (
            self._append_dims(torch.ones_like(sigma) * self.sigma_max ** 2, x.ndim)
            - self._append_dims(sigma ** 2, x.ndim)
        )
        gt2 = 2 * sigma
        d = -(0.5 if not stochastic else 1) * gt2 * (
            grad_pxtlx0 - w * grad_pxTlxt * (0 if stochastic else 1)
        )
        if stochastic:
            return d, self._append_dims(gt2, x.ndim)
        return d

    def _get_d_vp(self, x, sigma, denoised, x_T, w, stochastic=False):
        vp_snr_sqrt_reciprocal = self.scheduler._vp_snr_sqrt_reciprocal
        vp_snr_sqrt_reciprocal_deriv = self.scheduler._vp_snr_sqrt_reciprocal_deriv
        s_deriv = self.scheduler._s_deriv
        logs = self.scheduler._logs
        std = self.scheduler._std
        logsnr = self.scheduler._logsnr
        logsnr_T = logsnr(torch.as_tensor(self.sigma_max))
        logs_T = logs(torch.as_tensor(self.sigma_max))
        std_t = std(sigma)
        logsnr_t = logsnr(sigma)
        logs_t = logs(sigma)
        s_t_deriv = s_deriv(sigma)
        sigma_t = vp_snr_sqrt_reciprocal(sigma)
        sigma_t_deriv = vp_snr_sqrt_reciprocal_deriv(sigma)
        a_t = (logsnr_T - logsnr_t + logs_t - logs_T).exp()
        b_t = -torch.expm1(logsnr_T - logsnr_t) * logs_t.exp()
        mu_t = a_t * x_T + b_t * denoised
        grad_logq = -(x - mu_t) / std_t ** 2 / (-torch.expm1(logsnr_T - logsnr_t))
        grad_logpxTlxt = -(x - torch.exp(logs_t - logs_T) * x_T) / std_t ** 2 / torch.expm1(
            logsnr_t - logsnr_T
        )
        f = s_t_deriv * (-logs_t).exp() * x
        gt2 = 2 * logs_t.exp() ** 2 * sigma_t * sigma_t_deriv
        d = f - gt2 * ((0.5 if not stochastic else 1) * grad_logq - w * grad_logpxTlxt)
        if stochastic:
            return d, self._append_dims(gt2, x.ndim)
        return d

    def _convert_to_pil(self, images: torch.Tensor) -> List[Image.Image]:
        images = (images + 1) / 2
        images = images.clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).round().astype(np.uint8)
        return [Image.fromarray(img) for img in images]

    def _convert_to_numpy(self, images: torch.Tensor) -> np.ndarray:
        images = (images + 1) / 2
        images = images.clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        return images

    @torch.no_grad()
    def __call__(
        self,
        source_image: Union[torch.Tensor, Image.Image, List[Image.Image]],
        num_inference_steps: int = 40,
        guidance: float = 1.0,
        churn_step_ratio: float = 0.33,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
    ):
        device = self.device
        dtype = self.dtype
        x_T = self.prepare_inputs(source_image, device, dtype)
        batch_size = x_T.shape[0]
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        sigmas = self.scheduler.sigmas
        x = x_T.clone()
        s_in = x.new_ones([batch_size])
        nfe = 0
        progress_bar = tqdm(range(len(sigmas) - 1), desc="DDBM Sampling")

        for i in progress_bar:
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            if churn_step_ratio > 0 and sigma_next != 0:
                sigma_hat = (sigma_next - sigma) * churn_step_ratio + sigma
                denoised = self.denoise(x, sigma * s_in, x_T)
                nfe += 1
                d_1, gt2 = self._get_d_stochastic(x, sigma, denoised, x_T, guidance)
                dt = sigma_hat - sigma
                noise = randn_tensor(x.shape, generator=generator, device=device, dtype=dtype)
                x = x + d_1 * dt + noise * (dt.abs() ** 0.5) * gt2.sqrt()
            else:
                sigma_hat = sigma

            denoised = self.denoise(x, sigma_hat * s_in, x_T)
            nfe += 1
            d = self._get_d(x, sigma_hat, denoised, x_T, guidance)
            dt = sigma_next - sigma_hat

            if sigma_next == 0:
                x = x + d * dt
            else:
                x_2 = x + d * dt
                denoised_2 = self.denoise(x_2, sigma_next * s_in, x_T)
                nfe += 1
                d_2 = self._get_d(x_2, sigma_next, denoised_2, x_T, guidance)
                d_prime = (d + d_2) / 2
                x = x + d_prime * dt

            if callback is not None and i % callback_steps == 0:
                callback(i, num_inference_steps, x)

        images = x.clamp(-1, 1)
        if output_type == "pil":
            images = self._convert_to_pil(images)
        elif output_type == "np":
            images = self._convert_to_numpy(images)
        if not return_dict:
            return (images, nfe)
        return DDBMPipelineOutput(images=images, nfe=nfe)
