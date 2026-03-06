# Copyright (c) 2026 EarthBridge Team.
# Credits: See upstream/paper attribution below and README.md citations.

# Self-contained DBIM pipeline for production inference - no external project code.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

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
# DBIMUNet
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


class DBIMUNet(ModelMixin, ConfigMixin):
    """Wrapper around UNet2DModel for DBIM (x, timestep, xT=source)."""

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
# DBIMScheduler
# ---------------------------------------------------------------------------


@dataclass
class DBIMSchedulerOutput(BaseOutput):
    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None


class DBIMScheduler(SchedulerMixin, ConfigMixin):
    """Diffusion Bridge Implicit Models (DBIM) scheduler."""

    @register_to_config
    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        sigma_data: float = 0.5,
        rho: float = 7.0,
        num_train_timesteps: int = 40,
        pred_mode: str = "ve",
        eta: float = 1.0,
        beta_d: float = 2.0,
        beta_min: float = 0.1,
    ) -> None:
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.num_train_timesteps = num_train_timesteps
        self.pred_mode = pred_mode
        self.eta = eta
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.sigmas: Optional[torch.Tensor] = None

    def _vp_logsnr(self, t: torch.Tensor) -> torch.Tensor:
        return -self.beta_d * t ** 2 / 2 - self.beta_min * t

    def _vp_logs(self, t: torch.Tensor) -> torch.Tensor:
        return -self.beta_d * t ** 2 / 4 - self.beta_min * t / 2

    def get_abc(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.pred_mode == "ve":
            a_t = torch.ones_like(t)
            b_t = torch.zeros_like(t)
            c_t = t
        elif self.pred_mode == "vp":
            log_mean_coeff = self._vp_logs(t)
            a_t = torch.exp(log_mean_coeff)
            b_t = 1.0 - a_t
            c_t = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))
        else:
            raise ValueError(f"Unknown pred_mode: {self.pred_mode}")
        return a_t, b_t, c_t

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None) -> None:
        rho = self.rho
        inv_rho = 1.0 / rho
        step_indices = torch.linspace(0, 1, num_inference_steps + 1)
        sigmas = (self.sigma_max ** inv_rho + step_indices * (self.sigma_min ** inv_rho - self.sigma_max ** inv_rho)) ** rho
        self.sigmas = sigmas
        if device is not None:
            self.sigmas = self.sigmas.to(device)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        x_T: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[DBIMSchedulerOutput, Tuple]:
        sigma = self.sigmas[timestep]
        sigma_next = self.sigmas[timestep + 1]
        a_t, b_t, c_t = self.get_abc(sigma.unsqueeze(0))
        a_s, b_s, c_s = self.get_abc(sigma_next.unsqueeze(0))
        a_t = self._append_dims(a_t, sample.ndim).to(sample)
        b_t = self._append_dims(b_t, sample.ndim).to(sample)
        c_t = self._append_dims(c_t, sample.ndim).to(sample)
        a_s = self._append_dims(a_s, sample.ndim).to(sample)
        b_s = self._append_dims(b_s, sample.ndim).to(sample)
        c_s = self._append_dims(c_s, sample.ndim).to(sample)

        coeff_xt = a_s / (a_t + 1e-20)
        coeff_xT = b_s - b_t * a_s / (a_t + 1e-20)
        c_s2 = c_s ** 2
        c_t2 = c_t ** 2
        c_ratio = c_s2 / (c_t2 + 1e-20)
        coeff_d = (a_s / (a_t + 1e-20)) * (1.0 - c_ratio) if self.eta > 0 else torch.zeros_like(a_s)
        noise_var = c_s2 - c_ratio * (a_s / (a_t + 1e-20)) ** 2 * c_t2
        noise_var = noise_var.clamp(min=0)
        noise_std = torch.sqrt(noise_var) * self.eta

        x_0 = model_output
        noise = randn_tensor(sample.shape, generator=generator, device=sample.device, dtype=sample.dtype)
        x_prev = coeff_xt * sample
        if x_T is not None:
            x_prev = x_prev + coeff_xT * x_T
        x_prev = x_prev + coeff_d * (x_0 - sample) + noise_std * noise

        if not return_dict:
            return (x_prev, model_output)
        return DBIMSchedulerOutput(prev_sample=x_prev, pred_original_sample=model_output)

    @staticmethod
    def _append_dims(x: torch.Tensor, target_dims: int) -> torch.Tensor:
        dims = target_dims - x.ndim
        return x[(...,) + (None,) * dims]


# ---------------------------------------------------------------------------
# DBIMPipeline
# ---------------------------------------------------------------------------


@dataclass
class DBIMPipelineOutput(BaseOutput):
    images: Union[List[Image.Image], np.ndarray, torch.Tensor]
    nfe: int = 0


class DBIMPipeline(DiffusionPipeline):
    """Image-to-image translation pipeline using DBIM."""

    def __init__(self, unet: DBIMUNet, scheduler: DBIMScheduler) -> None:
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @staticmethod
    def _append_dims(x: torch.Tensor, target_dims: int) -> torch.Tensor:
        dims = target_dims - x.ndim
        return x[(...,) + (None,) * dims]

    def _bridge_scalings(self, t: torch.Tensor):
        a_t, b_t, c_t = self.scheduler.get_abc(t)
        sigma_data = self.scheduler.config.sigma_data
        A = a_t ** 2 * sigma_data ** 2 + b_t ** 2 * sigma_data ** 2 + c_t ** 2
        c_in = torch.rsqrt(torch.clamp(A, min=1e-20))
        c_skip = (b_t * sigma_data ** 2) / torch.clamp(A, min=1e-20)
        c_out = torch.sqrt(torch.clamp(a_t ** 2 * sigma_data ** 4 + sigma_data ** 2 * c_t ** 2, min=1e-20)) * c_in
        c_noise = 1000.0 * 0.25 * torch.log(torch.clamp(t, min=1e-44))
        return c_skip, c_out, c_in, c_noise

    def denoise(self, x_t, t, x_T, clip_denoised=True):
        c_skip, c_out, c_in, c_noise = self._bridge_scalings(t)
        c_skip = self._append_dims(c_skip, x_t.ndim).to(x_t)
        c_out = self._append_dims(c_out, x_t.ndim).to(x_t)
        c_in = self._append_dims(c_in, x_t.ndim).to(x_t)
        model_output = self.unet(c_in * x_t, c_noise, xT=x_T)
        denoised = c_out * model_output + c_skip * x_t
        if clip_denoised:
            denoised = denoised.clamp(-1, 1)
        return denoised

    @torch.no_grad()
    def __call__(
        self,
        source_image: Union[torch.Tensor, Image.Image, List[Image.Image]],
        num_inference_steps: int = 20,
        output_type: str = "pil",
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ):
        device = next(self.unet.parameters()).device
        dtype = next(self.unet.parameters()).dtype
        x_T = self._prepare_inputs(source_image, device, dtype)
        batch_size = x_T.shape[0]
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        sigmas = self.scheduler.sigmas
        s_in = x_T.new_ones([batch_size])
        nfe = 0

        a_0, b_0, c_0 = [self._append_dims(v, x_T.ndim) for v in self.scheduler.get_abc(sigmas[0] * s_in)]
        noise = randn_tensor(x_T.shape, generator=generator, device=device, dtype=dtype)
        x = a_0.to(x_T) * x_T + c_0.to(x_T) * noise

        for i in tqdm(range(len(sigmas) - 1), desc="DBIM Sampling"):
            denoised = self.denoise(x, sigmas[i] * s_in, x_T)
            nfe += 1
            result = self.scheduler.step(denoised, timestep=i, sample=x, x_T=x_T, generator=generator)
            x = result.prev_sample

        images = x.clamp(-1, 1)
        if output_type == "pil":
            images = self._convert_to_pil(images)
        elif output_type == "np":
            images = self._convert_to_numpy(images)
        if not return_dict:
            return (images, nfe)
        return DBIMPipelineOutput(images=images, nfe=nfe)

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
        images = (images + 1) / 2
        images = images.clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).round().astype(np.uint8)
        return [Image.fromarray(img) for img in images]

    @staticmethod
    def _convert_to_numpy(images):
        images = (images + 1) / 2
        return images.clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
