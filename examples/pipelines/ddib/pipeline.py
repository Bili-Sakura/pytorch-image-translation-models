# Copyright (c) 2026 EarthBridge Team.
# Credits: See upstream/paper attribution below and README.md citations.

# Copyright 2024 The DDIB Authors and The Hugging Face Team.
# Licensed under the MIT License.
#
# Self-contained DDIB pipeline for production inference - no external project code.
# DDIB translates by: (1) DDIM reverse encode source→latent, (2) DDIM forward decode latent→target.

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image

from diffusers import DiffusionPipeline, ModelMixin, UNet2DModel
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.schedulers.scheduling_utils import SchedulerMixin


# ---------------------------------------------------------------------------
# DDIBUNet
# ---------------------------------------------------------------------------


def _channel_mult_for_resolution(resolution: int) -> tuple:
    return {
        1024: (1, 1, 2, 2, 4, 4),
        512: (1, 2, 4, 4, 8),
        256: (1, 2, 2, 4),
        128: (1, 1, 2, 3, 4),
        64: (1, 2, 3, 4),
        32: (1, 2, 3, 4),
    }.get(resolution, (1, 2, 3, 4))


class DDIBUNet(ModelMixin, ConfigMixin):
    """Unconditional UNet for DDIB diffusion models."""

    @register_to_config
    def __init__(
        self,
        image_size: int = 256,
        in_channels: int = 3,
        model_channels: int = 128,
        num_res_blocks: int = 2,
        attention_resolutions: tuple = (1,),
        dropout: float = 0.0,
        learn_sigma: bool = False,
        channel_mult: Optional[tuple] = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.learn_sigma = learn_sigma
        if channel_mult is None:
            channel_mult = _channel_mult_for_resolution(image_size)
        out_channels = in_channels * 2 if learn_sigma else in_channels
        block_out_channels = tuple(model_channels * m for m in channel_mult)
        down_block_types = [
            "AttnDownBlock2D" if i in attention_resolutions else "DownBlock2D"
            for i in range(len(channel_mult))
        ]
        up_block_types = [
            "AttnUpBlock2D" if (len(channel_mult) - 1 - i) in attention_resolutions else "UpBlock2D"
            for i in range(len(channel_mult))
        ]
        self.unet = UNet2DModel(
            sample_size=image_size,
            in_channels=in_channels,
            out_channels=out_channels,
            block_out_channels=block_out_channels,
            down_block_types=tuple(down_block_types),
            up_block_types=tuple(up_block_types),
            layers_per_block=num_res_blocks,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return self.unet(x, timestep).sample


# ---------------------------------------------------------------------------
# DDIBScheduler
# ---------------------------------------------------------------------------


@dataclass
class DDIBSchedulerOutput(BaseOutput):
    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None


def _get_named_beta_schedule(schedule_name: str, num_diffusion_steps: int) -> np.ndarray:
    if schedule_name == "linear":
        scale = 1000 / num_diffusion_steps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(beta_start, beta_end, num_diffusion_steps, dtype=np.float64)
    elif schedule_name == "cosine":
        max_beta = 0.999
        s = 0.008

        def alpha_bar(t):
            return np.cos((t + s) / (1 + s) * np.pi / 2) ** 2

        betas = []
        for i in range(num_diffusion_steps):
            t1 = i / num_diffusion_steps
            t2 = (i + 1) / num_diffusion_steps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return np.array(betas, dtype=np.float64)
    raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


class DDIBScheduler(SchedulerMixin, ConfigMixin):
    """Gaussian diffusion scheduler with DDIM sampling for DDIB."""

    _compatibles = []
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        noise_schedule: str = "linear",
        learn_sigma: bool = False,
        predict_xstart: bool = False,
        rescale_timesteps: bool = False,
    ):
        betas = _get_named_beta_schedule(noise_schedule, num_train_timesteps)
        betas = np.array(betas, dtype=np.float64)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        self.num_train_timesteps = num_train_timesteps
        self.learn_sigma = learn_sigma
        self.predict_xstart = predict_xstart
        self.rescale_timesteps = rescale_timesteps
        self.betas = torch.from_numpy(betas).float()
        self.alphas_cumprod = torch.from_numpy(alphas_cumprod).float()
        self.alphas_cumprod_prev = torch.from_numpy(alphas_cumprod_prev).float()
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        self.timesteps: Optional[torch.Tensor] = None
        self.num_inference_steps: Optional[int] = None
        self.init_noise_sigma = 1.0

    def set_timesteps(
        self,
        num_inference_steps: int,
        device=None,
    ):
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps).to(device)

    def _scale_timesteps(self, t: torch.Tensor) -> torch.Tensor:
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_train_timesteps)
        return t

    def _predict_xstart_from_eps(self, x_t, t, eps):
        device = x_t.device
        coeff1 = self.sqrt_recip_alphas_cumprod.to(device)[t]
        coeff2 = self.sqrt_recipm1_alphas_cumprod.to(device)[t]
        while coeff1.ndim < x_t.ndim:
            coeff1 = coeff1.unsqueeze(-1)
            coeff2 = coeff2.unsqueeze(-1)
        return coeff1 * x_t - coeff2 * eps

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        device = x_t.device
        sqrt_alpha = self.sqrt_alphas_cumprod.to(device)[t]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod.to(device)[t]
        while sqrt_alpha.ndim < x_t.ndim:
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)
        return (x_t - sqrt_alpha * pred_xstart) / sqrt_one_minus_alpha

    def _get_xstart_and_eps(self, model_output, x_t, t):
        if self.learn_sigma:
            C = model_output.shape[1] // 2
            model_output = model_output[:, :C]
        if self.predict_xstart:
            pred_xstart = model_output
            pred_eps = self._predict_eps_from_xstart(x_t, t, pred_xstart)
        else:
            pred_eps = model_output
            pred_xstart = self._predict_xstart_from_eps(x_t, t, pred_eps)
        return pred_xstart, pred_eps

    def ddim_step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        timestep_prev: torch.Tensor,
        sample: torch.Tensor,
        eta: float = 0.0,
        clip_denoised: bool = True,
    ) -> DDIBSchedulerOutput:
        pred_xstart, pred_eps = self._get_xstart_and_eps(model_output, sample, timestep)
        if clip_denoised:
            pred_xstart = pred_xstart.clamp(-1, 1)
        device = sample.device
        alpha_bar = self.alphas_cumprod.to(device)[timestep]
        alpha_bar_prev = self.alphas_cumprod.to(device)[timestep_prev]
        while alpha_bar.ndim < sample.ndim:
            alpha_bar = alpha_bar.unsqueeze(-1)
            alpha_bar_prev = alpha_bar_prev.unsqueeze(-1)
        prev_sample = alpha_bar_prev.sqrt() * pred_xstart + (1.0 - alpha_bar_prev).sqrt() * pred_eps
        return DDIBSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_xstart)

    def ddim_reverse_step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        timestep_next: torch.Tensor,
        sample: torch.Tensor,
        clip_denoised: bool = True,
    ) -> DDIBSchedulerOutput:
        pred_xstart, pred_eps = self._get_xstart_and_eps(model_output, sample, timestep)
        if clip_denoised:
            pred_xstart = pred_xstart.clamp(-1, 1)
        device = sample.device
        alpha_bar_next = self.alphas_cumprod.to(device)[timestep_next]
        while alpha_bar_next.ndim < sample.ndim:
            alpha_bar_next = alpha_bar_next.unsqueeze(-1)
        next_sample = alpha_bar_next.sqrt() * pred_xstart + (1.0 - alpha_bar_next).sqrt() * pred_eps
        return DDIBSchedulerOutput(prev_sample=next_sample, pred_original_sample=pred_xstart)

    def scale_model_input(self, sample: torch.Tensor, timestep: Optional[int] = None) -> torch.Tensor:
        return sample


# ---------------------------------------------------------------------------
# DDIBPipeline
# ---------------------------------------------------------------------------


@dataclass
class DDIBPipelineOutput(BaseOutput):
    images: Union[List[Image.Image], np.ndarray, torch.Tensor]
    latent: Optional[torch.Tensor] = None


class DDIBPipeline(DiffusionPipeline):
    """Pipeline for image-to-image translation using Dual Diffusion Implicit Bridges."""

    model_cpu_offload_seq = "source_unet->target_unet"

    def __init__(self, source_unet: DDIBUNet, target_unet: DDIBUNet, scheduler: DDIBScheduler):
        super().__init__()
        self.register_modules(
            source_unet=source_unet,
            target_unet=target_unet,
            scheduler=scheduler,
        )

    @property
    def device(self) -> torch.device:
        return next(self.target_unet.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.target_unet.parameters()).dtype

    def _ddim_reverse_sample_loop(
        self,
        model: torch.nn.Module,
        x_0: torch.Tensor,
        timesteps: torch.Tensor,
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        x = x_0
        for i in range(len(timesteps) - 1):
            t_cur = timesteps[i]
            t_next = timesteps[i + 1]
            t_batch = t_cur.expand(x.shape[0])
            scaled_t = self.scheduler._scale_timesteps(t_batch)
            model_output = model(x, scaled_t)
            out = self.scheduler.ddim_reverse_step(
                model_output=model_output,
                timestep=t_batch,
                timestep_next=t_next.expand(x.shape[0]),
                sample=x,
                clip_denoised=clip_denoised,
            )
            x = out.prev_sample
        return x

    def _ddim_sample_loop(
        self,
        model: torch.nn.Module,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        clip_denoised: bool = True,
        eta: float = 0.0,
    ) -> torch.Tensor:
        x = noise
        reversed_timesteps = timesteps.flip(0)
        for i in range(len(reversed_timesteps) - 1):
            t_cur = reversed_timesteps[i]
            t_prev = reversed_timesteps[i + 1]
            t_batch = t_cur.expand(x.shape[0])
            scaled_t = self.scheduler._scale_timesteps(t_batch)
            model_output = model(x, scaled_t)
            out = self.scheduler.ddim_step(
                model_output=model_output,
                timestep=t_batch,
                timestep_prev=t_prev.expand(x.shape[0]),
                sample=x,
                eta=eta,
                clip_denoised=clip_denoised,
            )
            x = out.prev_sample
        return x

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
                img_array = np.array(img).astype(np.float32) / 255.0
                if img_array.ndim == 2:
                    img_tensor = torch.from_numpy(img_array).unsqueeze(0)
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

    @torch.no_grad()
    def __call__(
        self,
        source_image: Union[torch.Tensor, Image.Image, List[Image.Image]],
        num_inference_steps: int = 250,
        clip_denoised: bool = True,
        eta: float = 0.0,
        output_type: str = "pil",
        return_dict: bool = True,
        return_latent: bool = False,
    ):
        device = self.device
        dtype = self.dtype
        x_source = self.prepare_inputs(source_image, device, dtype)
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        latent = self._ddim_reverse_sample_loop(
            self.source_unet, x_source, timesteps, clip_denoised=clip_denoised,
        )
        images = self._ddim_sample_loop(
            self.target_unet, latent, timesteps, clip_denoised=clip_denoised, eta=eta,
        )
        images = images.clamp(-1, 1)
        if output_type == "pil":
            images = self._convert_to_pil(images)
        elif output_type == "np":
            images = self._convert_to_numpy(images)
        if not return_dict:
            return (images,)
        return DDIBPipelineOutput(
            images=images,
            latent=latent if return_latent else None,
        )

    @staticmethod
    def _convert_to_pil(images: torch.Tensor) -> List[Image.Image]:
        images = (images + 1) / 2
        images = images.clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).round().astype(np.uint8)
        return [Image.fromarray(img) for img in images]

    @staticmethod
    def _convert_to_numpy(images: torch.Tensor) -> np.ndarray:
        images = (images + 1) / 2
        images = images.clamp(0, 1)
        return images.cpu().permute(0, 2, 3, 1).numpy()
