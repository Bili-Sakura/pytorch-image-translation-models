# Copyright (c) 2026 EarthBridge Team.
# Credits: See upstream/paper attribution below and README.md citations.

# Self-contained CDTSDE pipeline for production inference - no external project code.

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union

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
# CDTSDEUNet
# ---------------------------------------------------------------------------


def _build_block_types(channel_mult: tuple, attention_resolutions: tuple) -> tuple:
    down_block_types = [
        "AttnDownBlock2D" if i in attention_resolutions else "DownBlock2D"
        for i in range(len(channel_mult))
    ]
    up_block_types = [
        "AttnUpBlock2D" if (len(channel_mult) - 1 - i) in attention_resolutions else "UpBlock2D"
        for i in range(len(channel_mult))
    ]
    return tuple(down_block_types), tuple(up_block_types)


def _channel_mult_for_resolution(resolution: int) -> tuple:
    return {
        1024: (1, 1, 2, 2, 4, 4),
        512: (1, 2, 4, 4, 8),
        256: (1, 2, 2, 4),
        128: (1, 1, 2, 3, 4),
        64: (1, 2, 3, 4),
        32: (1, 2, 3, 4),
    }.get(resolution, (1, 2, 3, 4))


class CDTSDEUNet(ModelMixin, ConfigMixin):
    """Wrapper around UNet2DModel for CDTSDE (x, timestep, xT=source)."""

    @register_to_config
    def __init__(
        self,
        image_size: int = 256,
        in_channels: int = 3,
        out_channels: Optional[int] = None,
        model_channels: int = 128,
        num_res_blocks: int = 2,
        attention_resolutions: tuple = (1,),
        dropout: float = 0.0,
        condition_mode: Optional[str] = "concat",
        channel_mult: Optional[tuple] = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.condition_mode = condition_mode
        if out_channels is None:
            out_channels = in_channels
        self.out_channels = out_channels
        if channel_mult is None:
            channel_mult = _channel_mult_for_resolution(image_size)
        unet_in_channels = in_channels * 2 if condition_mode == "concat" else in_channels
        block_out_channels = tuple(model_channels * m for m in channel_mult)
        down_block_types, up_block_types = _build_block_types(channel_mult, attention_resolutions)
        self.unet = UNet2DModel(
            sample_size=image_size,
            in_channels=unet_in_channels,
            out_channels=out_channels,
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


# ---------------------------------------------------------------------------
# CDTSDEScheduler
# ---------------------------------------------------------------------------


@dataclass
class CDTSDESchedulerOutput(BaseOutput):
    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None


class CDTSDEScheduler(SchedulerMixin, ConfigMixin):
    """CDTSDE scheduler with dynamic domain-shift scheduling."""

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        linear_start: float = 0.0015,
        linear_end: float = 0.0195,
        eta_max: float = 1.0,
        eta_min: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_train_timesteps = num_train_timesteps
        betas = torch.linspace(linear_start ** 0.5, linear_end ** 0.5, num_train_timesteps, dtype=torch.float64) ** 2
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.train_alphas_cumprod = alphas_cumprod.float()
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).float()
        self.sigmas = torch.sqrt(1.0 - alphas_cumprod).float()
        etas = torch.linspace(eta_max, eta_min, num_train_timesteps + 1).float()
        self.etas = etas
        self.timesteps: Optional[torch.Tensor] = None
        self.lambdas: Optional[torch.Tensor] = None

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None) -> None:
        step_ratio = self.num_train_timesteps // num_inference_steps
        self.timesteps = (torch.arange(0, num_inference_steps) * step_ratio).long()
        if device is not None:
            self.timesteps = self.timesteps.to(device)
        self.lambdas = self.etas[:num_inference_steps + 1]
        if device is not None:
            self.lambdas = self.lambdas.to(device)

    def add_noise(self, original, noise, timesteps):
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1).to(original)
        sigma = self.sigmas[timesteps].view(-1, 1, 1, 1).to(original)
        return sqrt_alpha * original + sigma * noise

    def predict_start_from_noise(self, sample, timesteps, noise, use_inference_schedule=False):
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1).to(sample)
        sigma = self.sigmas[timesteps].view(-1, 1, 1, 1).to(sample)
        return (sample - sigma * noise) / (sqrt_alpha + 1e-8)

    def step(
        self,
        pred_original_sample: torch.Tensor,
        step_index: int,
        sample: torch.Tensor,
        reference_sample: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        stochastic: bool = True,
        return_dict: bool = True,
    ) -> Union[CDTSDESchedulerOutput, tuple]:
        if step_index <= 0:
            if not return_dict:
                return (pred_original_sample, pred_original_sample)
            return CDTSDESchedulerOutput(prev_sample=pred_original_sample, pred_original_sample=pred_original_sample)

        prev_index = step_index - 1
        sqrt_alpha_t = self.sqrt_alphas_cumprod[step_index].to(sample)
        sqrt_alpha_s = self.sqrt_alphas_cumprod[prev_index].to(sample)
        sigma_t = self.sigmas[step_index].to(sample)
        sigma_s = self.sigmas[prev_index].to(sample)

        coeff_x0 = sqrt_alpha_s - (sigma_s / (sigma_t + 1e-8)) * sqrt_alpha_t
        coeff_xt = sigma_s / (sigma_t + 1e-8)
        x_prev = coeff_x0 * pred_original_sample + coeff_xt * sample

        if stochastic and step_index > 1:
            noise = randn_tensor(sample.shape, generator=generator, device=sample.device, dtype=sample.dtype)
            noise_scale = (sigma_s ** 2 - (sigma_s / (sigma_t + 1e-8) * sigma_t) ** 2).clamp(min=0).sqrt()
            x_prev = x_prev + noise_scale * noise

        if not return_dict:
            return (x_prev, pred_original_sample)
        return CDTSDESchedulerOutput(prev_sample=x_prev, pred_original_sample=pred_original_sample)


# ---------------------------------------------------------------------------
# CDTSDEPipeline
# ---------------------------------------------------------------------------


@dataclass
class CDTSDEPipelineOutput(BaseOutput):
    images: Union[List[Image.Image], np.ndarray, torch.Tensor]
    nfe: int = 0


class CDTSDEPipeline(DiffusionPipeline):
    """Image-to-image translation pipeline using CDTSDE."""

    def __init__(self, unet: CDTSDEUNet, scheduler: CDTSDEScheduler) -> None:
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        source_image: Union[torch.Tensor, Image.Image, List[Image.Image]],
        num_inference_steps: int = 50,
        stochastic: bool = True,
        output_type: str = "pil",
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ):
        device = next(self.unet.parameters()).device
        dtype = next(self.unet.parameters()).dtype
        x_T = self._prepare_inputs(source_image, device, dtype)
        batch_size = x_T.shape[0]

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        sqrt_alpha_last = self.scheduler.sqrt_alphas_cumprod[-1].to(device=device, dtype=dtype)
        sigma_last = self.scheduler.sigmas[-1].to(device=device, dtype=dtype)
        noise = randn_tensor(x_T.shape, generator=generator, device=device, dtype=dtype)
        x = sqrt_alpha_last * x_T + sigma_last * noise

        nfe = 0
        total_steps = len(self.scheduler.timesteps) - 1
        for i in tqdm(range(total_steps), desc="CDTSDE Sampling"):
            j = total_steps - i
            t_model = self.scheduler.timesteps[j]
            t_batch = torch.full((batch_size,), t_model, device=device, dtype=torch.long)
            pred_noise = self.unet(x, t_batch, xT=x_T)
            nfe += 1
            idx_batch = torch.full((batch_size,), j, device=device, dtype=torch.long)
            pred_x0 = self.scheduler.predict_start_from_noise(sample=x, timesteps=idx_batch, noise=pred_noise)
            step_out = self.scheduler.step(
                pred_original_sample=pred_x0, step_index=j, sample=x,
                reference_sample=x_T, generator=generator, stochastic=stochastic, return_dict=True,
            )
            x = step_out.prev_sample

        images = x.clamp(-1.0, 1.0)
        if output_type == "pil":
            images = self._convert_to_pil(images)
        elif output_type == "np":
            images = self._convert_to_numpy(images)
        if not return_dict:
            return (images, nfe)
        return CDTSDEPipelineOutput(images=images, nfe=nfe)

    @staticmethod
    def _prepare_inputs(image, device, dtype):
        if isinstance(image, Image.Image):
            image = [image]
        if isinstance(image, list) and isinstance(image[0], Image.Image):
            imgs = []
            for img in image:
                arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
                imgs.append(torch.from_numpy(arr).permute(2, 0, 1))
            image = torch.stack(imgs)
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        if image.max() > 1.0:
            image = image / 255.0
        if image.min() >= 0:
            image = image * 2 - 1
        return image.to(device=device, dtype=dtype)

    @staticmethod
    def _convert_to_pil(images):
        images = (images + 1.0) / 2.0
        images = images.clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).round().astype(np.uint8)
        return [Image.fromarray(img.squeeze(-1) if img.shape[-1] == 1 else img) for img in images]

    @staticmethod
    def _convert_to_numpy(images):
        images = (images + 1.0) / 2.0
        return images.clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
