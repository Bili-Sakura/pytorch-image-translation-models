# Copyright (c) 2026 EarthBridge Team.
# Credits: See upstream/paper attribution below and README.md citations.

# Copyright 2024 The I2SB Authors and The Hugging Face Team.
# Licensed under the Apache License, Version 2.0.
#
# Diffusers-compatible I2SB pipeline for production inference.
# Core scheduler logic is imported from src.schedulers.i2sb to avoid duplication.

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

# Re-use the core I2SBScheduler to avoid code duplication.
from src.schedulers.i2sb import I2SBScheduler  # noqa: F401


# ---------------------------------------------------------------------------
# I2SBUNet  (diffusers-compatible wrapper – unique to this module)
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
# I2SBPipeline
# ---------------------------------------------------------------------------


@dataclass
class I2SBPipelineOutput(BaseOutput):
    images: Union[List[Image.Image], np.ndarray, torch.Tensor]
    nfe: int = 0


class I2SBPipeline(DiffusionPipeline):
    """Pipeline for image-to-image generation using Image-to-Image Schrödinger Bridge.

    Uses the core ``I2SBScheduler`` from ``src.schedulers.i2sb`` to avoid
    duplicating scheduler logic.
    """

    model_cpu_offload_seq = "unet"

    def __init__(self, unet: I2SBUNet, scheduler: I2SBScheduler):
        super().__init__()
        # Only register the diffusers-compatible UNet module.  The core
        # I2SBScheduler is stored as a plain attribute because it does not
        # inherit from diffusers SchedulerMixin.
        self.register_modules(unet=unet)
        self.scheduler = scheduler

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

        self.scheduler.set_timesteps(nfe)
        # Core scheduler returns descending timesteps: [interval-1 … 0]
        steps = self.scheduler.timesteps.to(device)

        xt = x1.clone()
        interval = self.scheduler.interval
        noise_levels = torch.linspace(1e-4, 1.0, interval, device=device, dtype=dtype)
        has_condition = hasattr(self.unet, "condition_mode") and self.unet.condition_mode == "concat"
        cond = x1 if has_condition else None
        num_steps = len(steps) - 1
        progress_bar = tqdm(range(num_steps), desc="I2SB Sampling")
        nfe_count = 0

        for i in progress_bar:
            step = steps[i]
            prev_step = steps[i + 1]
            step_int = step.item()
            t_emb = noise_levels[step_int] * interval
            t_batch = torch.full((batch_size,), t_emb, device=device, dtype=dtype)
            pred = self.unet(xt, t_batch, cond=cond)
            nfe_count += 1
            pred_x0 = self.scheduler.compute_pred_x0(step_int, xt, pred, clip_denoise=clip_denoise)
            xt = self.scheduler.p_posterior(prev_step.item(), step_int, xt, pred_x0, ot_ode=ot_ode)
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
