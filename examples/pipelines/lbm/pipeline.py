# Copyright (c) 2026 EarthBridge Team.
# Credits: See upstream/paper attribution below and README.md citations.

# Self-contained LBM pipeline for production inference - no external project code.

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from diffusers import DiffusionPipeline, ModelMixin, UNet2DModel
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.schedulers.scheduling_utils import SchedulerMixin


# ---------------------------------------------------------------------------
# LBMUNet
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


class LBMUNet(ModelMixin, ConfigMixin):
    """Wrapper around UNet2DModel for LBM (sample, timestep, cond=…)."""

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
        sample: torch.Tensor,
        timestep: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.condition_mode == "concat" and cond is not None:
            sample = torch.cat([sample, cond], dim=1)
        return self.unet(sample, timestep).sample


# ---------------------------------------------------------------------------
# LBMScheduler
# ---------------------------------------------------------------------------


@dataclass
class LBMSchedulerOutput(BaseOutput):
    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None


class LBMScheduler(SchedulerMixin, ConfigMixin):
    """Latent Bridge Matching (LBM) flow-matching scheduler."""

    _compatibles: list[str] = []
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        bridge_noise_sigma: float = 0.001,
    ) -> None:
        super().__init__()
        self.num_train_timesteps = num_train_timesteps
        self.bridge_noise_sigma = bridge_noise_sigma
        sigmas = np.linspace(1.0, 0.0, num_train_timesteps + 1, dtype=np.float64)
        timesteps = np.arange(0, num_train_timesteps, dtype=np.int64)
        self.sigmas = torch.from_numpy(sigmas).float()
        self.timesteps = torch.from_numpy(timesteps).long()
        self.num_inference_steps: Optional[int] = None

    def get_sigmas(self, timesteps, n_dim=4, device="cpu", dtype=torch.float32):
        sigmas = self.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = self.timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def set_timesteps(self, num_inference_steps, device=None):
        self.num_inference_steps = num_inference_steps
        sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
        self.sigmas = torch.from_numpy(sigmas).float()
        if device is not None:
            self.sigmas = self.sigmas.to(device)
        self.timesteps = torch.linspace(0, self.num_train_timesteps - 1, num_inference_steps, device=device).long()

    def step(self, model_output, timestep, sample, generator=None, return_dict=True):
        device = sample.device
        dtype = sample.dtype
        t = timestep.to(device) if isinstance(timestep, torch.Tensor) else torch.tensor([timestep], device=device)
        if t.ndim == 0:
            t = t.unsqueeze(0)
        sigmas = self.get_sigmas(t, n_dim=sample.ndim, device=device, dtype=dtype)
        pred_x0 = sample - model_output * sigmas
        if not return_dict:
            return (pred_x0, pred_x0)
        return LBMSchedulerOutput(prev_sample=pred_x0, pred_original_sample=pred_x0)


# ---------------------------------------------------------------------------
# LBMPipeline
# ---------------------------------------------------------------------------


@dataclass
class LBMPipelineOutput(BaseOutput):
    images: Union[List[Image.Image], np.ndarray, torch.Tensor]
    nfe: int = 0


class LBMPipeline(DiffusionPipeline):
    """Image-to-image translation pipeline using LBM flow-matching."""

    def __init__(self, unet: LBMUNet, scheduler: LBMScheduler) -> None:
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        source_image: Union[torch.Tensor, Image.Image, List[Image.Image]],
        num_inference_steps: int = 1,
        cfg_scale: float = 1.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
    ):
        device = next(self.unet.parameters()).device
        dtype = next(self.unet.parameters()).dtype
        x_source = self._prepare_inputs(source_image, device, dtype)
        batch_size = x_source.shape[0]

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        sample = x_source.clone()

        has_condition = hasattr(self.unet, "condition_mode") and self.unet.condition_mode == "concat"
        cond = x_source if has_condition else None
        use_cfg = has_condition and abs(float(cfg_scale) - 1.0) > 1e-6
        null_condition = torch.zeros_like(cond) if use_cfg else None
        nfe_per_step = 2 if use_cfg else 1
        nfe_count = 0

        for i, t in tqdm(enumerate(self.scheduler.timesteps), total=len(self.scheduler.timesteps), desc="LBM Sampling"):
            t_batch = t.to(device).repeat(batch_size) if isinstance(t, torch.Tensor) else torch.full(
                (batch_size,), t, device=device, dtype=torch.long,
            )
            if use_cfg:
                model_input = torch.cat([sample, sample], dim=0)
                timestep_input = torch.cat([t_batch, t_batch], dim=0)
                cond_input = torch.cat([cond, null_condition], dim=0)
                pred_batched = self.unet(model_input, timestep_input, cond=cond_input)
                pred_cond, pred_uncond = pred_batched.chunk(2, dim=0)
                pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
            else:
                pred = self.unet(sample, t_batch, cond=cond)
            nfe_count += nfe_per_step

            result = self.scheduler.step(pred, t, sample, return_dict=True)
            sample = result.prev_sample

            if i < len(self.scheduler.timesteps) - 1:
                next_t = self.scheduler.timesteps[i + 1]
                next_t_batch = next_t.to(device).repeat(batch_size)
                next_sigmas = self.scheduler.get_sigmas(next_t_batch, n_dim=sample.ndim, device=device, dtype=dtype)
                bridge_noise = self.scheduler.bridge_noise_sigma
                sample = sample + bridge_noise * (next_sigmas * (1.0 - next_sigmas)) ** 0.5 * torch.randn_like(sample)

            if callback is not None and i % callback_steps == 0:
                callback(i, len(self.scheduler.timesteps), sample)

        images = sample.clamp(-1, 1)
        if output_type == "pil":
            images = self._convert_to_pil(images)
        elif output_type == "np":
            images = self._convert_to_numpy(images)
        if not return_dict:
            return (images, nfe_count)
        return LBMPipelineOutput(images=images, nfe=nfe_count)

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
        pil_images = []
        for img in images:
            if img.shape[2] == 1:
                img = img.squeeze(2)
            pil_images.append(Image.fromarray(img))
        return pil_images

    @staticmethod
    def _convert_to_numpy(images):
        images = (images + 1) / 2
        return images.clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
