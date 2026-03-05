# Copyright (c) 2026 EarthBridge Team.
# Credits: See upstream/paper attribution below and README.md citations.

# Copyright 2024 The I2SB Authors and The Hugging Face Team.
# Licensed under the Apache License, Version 2.0.
#
# Self-contained I2SB pipeline for production inference - no external project code.

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
# I2SBUNet
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


class I2SBUNet(ModelMixin, ConfigMixin):
    """Wrapper around UNet2DModel that accepts the I2SB calling convention (x, timestep, cond=…)."""

    @register_to_config
    def __init__(
        self,
        image_size: int = 256,
        in_channels: int = 3,
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
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.condition_mode == "concat" and cond is not None:
            x = torch.cat([x, cond], dim=1)
        return self.unet(x, timestep).sample

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Load UNet; if ema_unet has no config.json, load config from unet and weights from ema_unet."""
        from pathlib import Path
        path = Path(pretrained_model_name_or_path)
        subfolder = kwargs.get("subfolder", "unet")
        if subfolder == "ema_unet" and not (path / "ema_unet" / "config.json").exists():
            unet = super().from_pretrained(path, subfolder="unet", **{k: v for k, v in kwargs.items() if k != "subfolder"})
            from safetensors.torch import load_file
            ema_path = path / "ema_unet" / "diffusion_pytorch_model.safetensors"
            if ema_path.exists():
                unet.load_state_dict(load_file(str(ema_path)), strict=True)
            return unet
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)


# ---------------------------------------------------------------------------
# I2SBScheduler
# ---------------------------------------------------------------------------


@dataclass
class I2SBSchedulerOutput(BaseOutput):
    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None


def _make_beta_schedule(n_timestep: int, linear_start: float = 1e-4, linear_end: float = 2e-2) -> np.ndarray:
    betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=np.float64) ** 2
    return betas


class I2SBScheduler(SchedulerMixin, ConfigMixin):
    """Scheduler for Image-to-Image Schrödinger Bridge (I2SB)."""

    _compatibles = []
    order = 1

    @register_to_config
    def __init__(
        self,
        interval: int = 1000,
        beta_max: float = 0.3,
        t0: float = 1e-4,
        T: float = 1.0,
    ):
        self.interval = interval
        linear_end = beta_max / interval
        betas = _make_beta_schedule(interval, linear_start=1e-4, linear_end=linear_end)
        half = interval // 2
        betas = np.concatenate([betas[:half], betas[:half][::-1]])
        std_fwd = np.sqrt(np.cumsum(betas))
        std_bwd = np.sqrt(np.flip(np.cumsum(np.flip(betas))))
        denom = std_fwd ** 2 + std_bwd ** 2
        mu_x0 = std_bwd ** 2 / denom
        mu_x1 = std_fwd ** 2 / denom
        var = (std_fwd ** 2 * std_bwd ** 2) / denom
        std_sb = np.sqrt(var)
        self.betas = torch.from_numpy(betas).float()
        self.std_fwd = torch.from_numpy(std_fwd).float()
        self.std_bwd = torch.from_numpy(std_bwd).float()
        self.mu_x0 = torch.from_numpy(mu_x0).float()
        self.mu_x1 = torch.from_numpy(mu_x1).float()
        self.std_sb = torch.from_numpy(std_sb).float()
        self.timesteps: Optional[torch.Tensor] = None
        self.num_inference_steps: Optional[int] = None

    def compute_pred_x0(
        self,
        step,
        xt: torch.Tensor,
        net_out: torch.Tensor,
        clip_denoise: bool = False,
    ) -> torch.Tensor:
        device = xt.device
        batch_size = xt.shape[0]
        if not isinstance(step, torch.Tensor):
            step = torch.full((batch_size,), step, device=device, dtype=torch.long)
        std_fwd_t = self.std_fwd.to(device)[step].view(batch_size, 1, 1, 1)
        pred_x0 = xt - std_fwd_t * net_out
        if clip_denoise:
            pred_x0 = pred_x0.clamp(-1, 1)
        return pred_x0

    def p_posterior(
        self,
        nprev,
        n,
        x_n: torch.Tensor,
        x0: torch.Tensor,
        ot_ode: bool = False,
    ) -> torch.Tensor:
        device = x_n.device
        batch_size = x_n.shape[0]
        if not isinstance(n, torch.Tensor):
            n = torch.full((batch_size,), n, device=device, dtype=torch.long)
        if not isinstance(nprev, torch.Tensor):
            nprev = torch.full((batch_size,), nprev, device=device, dtype=torch.long)
        n = n.reshape(-1)
        nprev = nprev.reshape(-1)
        if n.numel() == 1:
            n = n.expand(batch_size)
        if nprev.numel() == 1:
            nprev = nprev.expand(batch_size)
        std_fwd_n = self.std_fwd.to(device)[n].view(batch_size, 1, 1, 1)
        std_fwd_nprev = self.std_fwd.to(device)[nprev].view(batch_size, 1, 1, 1)
        std_delta = (std_fwd_n ** 2 - std_fwd_nprev ** 2).sqrt()
        mu = (std_fwd_nprev ** 2) / (std_fwd_n ** 2) * x_n + (std_delta ** 2) / (std_fwd_n ** 2) * x0
        if ot_ode:
            return mu
        var = (std_fwd_nprev ** 2 * std_delta ** 2) / (std_fwd_n ** 2)
        noise = torch.randn_like(x_n)
        return mu + var.sqrt() * noise

    def set_timesteps(self, nfe: int, device=None):
        self.num_inference_steps = nfe
        steps = torch.linspace(0, self.interval - 1, nfe + 1, device=device).long()
        self.timesteps = steps

    def scale_model_input(self, sample: torch.Tensor, timestep: Optional[int] = None) -> torch.Tensor:
        return sample


# ---------------------------------------------------------------------------
# I2SBPipeline
# ---------------------------------------------------------------------------


@dataclass
class I2SBPipelineOutput(BaseOutput):
    images: Union[List[Image.Image], np.ndarray, torch.Tensor]
    nfe: int = 0


class I2SBPipeline(DiffusionPipeline):
    """Pipeline for image-to-image generation using Image-to-Image Schrödinger Bridge."""

    model_cpu_offload_seq = "unet"

    def __init__(self, unet: I2SBUNet, scheduler: I2SBScheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @property
    def device(self) -> torch.device:
        return next(self.unet.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.unet.parameters()).dtype

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
                img = img.convert("RGB") if img.mode != "L" else img
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
        nfe: int = 100,
        ot_ode: bool = False,
        clip_denoise: bool = False,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
    ):
        device = self.device
        dtype = self.dtype
        x1 = self.prepare_inputs(source_image, device, dtype)
        batch_size = x1.shape[0]
        self.scheduler.set_timesteps(nfe, device=device)
        steps = self.scheduler.timesteps
        xt = x1.clone()
        interval = self.scheduler.config.interval
        t0_val = self.scheduler.config.t0
        T_val = self.scheduler.config.T
        noise_levels = torch.linspace(t0_val, T_val, interval, device=device, dtype=dtype)
        has_condition = hasattr(self.unet, "condition_mode") and self.unet.condition_mode == "concat"
        cond = x1 if has_condition else None
        num_steps = len(steps) - 1
        progress_bar = tqdm(range(num_steps), desc="I2SB Sampling")
        nfe_count = 0

        for i in progress_bar:
            step = steps[num_steps - i]
            prev_step = steps[num_steps - i - 1]
            step_int = step.item() if isinstance(step, torch.Tensor) else step
            t_emb = noise_levels[step_int] * interval
            t_batch = torch.full((batch_size,), t_emb, device=device, dtype=dtype)
            pred = self.unet(xt, t_batch, cond=cond)
            nfe_count += 1
            pred_x0 = self.scheduler.compute_pred_x0(step_int, xt, pred, clip_denoise=clip_denoise)
            xt = self.scheduler.p_posterior(prev_step, step, xt, pred_x0, ot_ode=ot_ode)
            if callback is not None and i % callback_steps == 0:
                callback(i, num_steps, xt)

        images = xt.clamp(-1, 1)
        if output_type == "pil":
            images = self._convert_to_pil(images)
        elif output_type == "np":
            images = self._convert_to_numpy(images)
        if not return_dict:
            return (images, nfe_count)
        return I2SBPipelineOutput(images=images, nfe=nfe_count)

    def _convert_to_pil(self, images: torch.Tensor) -> List[Image.Image]:
        images = (images + 1) / 2
        images = images.clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).round().astype(np.uint8)
        pil_images = []
        for img in images:
            if img.shape[2] == 1:
                img = img.squeeze(2)
            pil_images.append(Image.fromarray(img))
        return pil_images

    def _convert_to_numpy(self, images: torch.Tensor) -> np.ndarray:
        images = (images + 1) / 2
        images = images.clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        return images
