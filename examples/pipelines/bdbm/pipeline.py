# Copyright (c) 2026 EarthBridge Team.
# Credits: See upstream/paper attribution below and README.md citations.

# Self-contained BDBM pipeline for production inference - no external project code.

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
from diffusers.schedulers.scheduling_utils import SchedulerMixin


# ---------------------------------------------------------------------------
# BDBMUNet
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


def _extract(a: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class BDBMUNet(ModelMixin, ConfigMixin):
    """Wrapper around UNet2DModel for BDBM (x_t, timesteps, context=…)."""

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
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.condition_mode == "concat" and context is not None:
            x_t = torch.cat([x_t, context], dim=1)
        return self.unet(x_t, timesteps).sample


# ---------------------------------------------------------------------------
# BDBMScheduler
# ---------------------------------------------------------------------------


@dataclass
class BDBMSchedulerOutput(BaseOutput):
    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None


class BDBMScheduler(SchedulerMixin, ConfigMixin):
    """Bidirectional Brownian Bridge scheduler for BDBM."""

    @register_to_config
    def __init__(
        self,
        num_timesteps: int = 1000,
        mt_type: str = "linear",
        m0: float = 0.001,
        mT: float = 0.999,
        eta: float = 1.0,
        var_scale: float = 2.0,
        skip_sample: bool = True,
        sample_step: int = 100,
        sample_step_type: str = "linear",
        objective: str = "noise",
    ) -> None:
        super().__init__()
        self.num_timesteps = num_timesteps
        self.mt_type = mt_type
        self.m0 = m0
        self.mT = mT
        self.eta = eta
        self.var_scale = var_scale
        self.skip_sample = skip_sample
        self.sample_step = sample_step
        self.sample_step_type = sample_step_type
        self.objective = objective
        self.m_t: Optional[torch.Tensor] = None
        self.variance_t: Optional[torch.Tensor] = None
        self.steps: Optional[torch.Tensor] = None
        self.asc_steps: Optional[torch.Tensor] = None
        self._register_schedule()

    def _register_schedule(self) -> None:
        T = self.num_timesteps
        if self.mt_type == "linear":
            m_t = np.linspace(self.m0, self.mT, T)
        else:
            raise NotImplementedError(f"Unknown mt_type: {self.mt_type}")
        variance_t = (m_t - m_t ** 2) * self.var_scale
        self.m_t = torch.tensor(m_t, dtype=torch.float32)
        self.variance_t = torch.tensor(variance_t, dtype=torch.float32)
        self._build_steps()

    def _build_steps(self) -> None:
        if self.skip_sample:
            midsteps = torch.arange(
                self.num_timesteps - 2, 1,
                step=-((self.num_timesteps - 3) / (self.sample_step - 3)),
            ).long()
            self.steps = torch.cat((
                torch.tensor([self.num_timesteps - 1], dtype=torch.long),
                midsteps,
                torch.tensor([1, 0], dtype=torch.long),
            ), dim=0)
        else:
            self.steps = torch.arange(self.num_timesteps - 1, -1, -1)
        self.asc_steps = self.steps.flip(0)

    def step_b2a(
        self,
        model_output: torch.Tensor,
        step_index: int,
        x_t: torch.Tensor,
        source: torch.Tensor,
        clip_denoised: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> BDBMSchedulerOutput:
        device = x_t.device
        t = torch.full((x_t.shape[0],), self.steps[step_index].item(), device=device, dtype=torch.long)
        m_t = _extract(self.m_t.to(device), t, x_t.shape)
        var_t = _extract(self.variance_t.to(device), t, x_t.shape)
        sigma_t = torch.sqrt(var_t)
        target_recon = (x_t - m_t * source - sigma_t * model_output) / (1.0 - m_t + 1e-8)
        if clip_denoised:
            target_recon = target_recon.clamp(-1.0, 1.0)
        if self.steps[step_index] == 0:
            return BDBMSchedulerOutput(prev_sample=target_recon, pred_original_sample=target_recon)
        n_t = torch.full((x_t.shape[0],), self.steps[step_index + 1].item(), device=device, dtype=torch.long)
        m_nt = _extract(self.m_t.to(device), n_t, x_t.shape)
        var_nt = _extract(self.variance_t.to(device), n_t, x_t.shape)
        noise = torch.randn(x_t.shape, device=device, generator=generator, dtype=x_t.dtype)
        sigma2_t = self.var_scale * (m_t - m_nt) * m_nt / (m_t + 1e-8)
        sigma_new = torch.sqrt(sigma2_t) * self.eta
        coe_eps = torch.sqrt((var_nt - sigma_new ** 2).clamp(min=0) / (var_t + 1e-8))
        x_prev = (1.0 - m_nt) * target_recon + m_nt * source + coe_eps * (x_t - (1.0 - m_t) * target_recon - m_t * source) + sigma_new * noise
        return BDBMSchedulerOutput(prev_sample=x_prev, pred_original_sample=target_recon)

    def step_a2b(
        self,
        model_output: torch.Tensor,
        step_index: int,
        x_t: torch.Tensor,
        target: torch.Tensor,
        clip_denoised: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> BDBMSchedulerOutput:
        device = x_t.device
        t = torch.full((x_t.shape[0],), self.asc_steps[step_index].item(), device=device, dtype=torch.long)
        m_t = _extract(self.m_t.to(device), t, x_t.shape)
        var_t = _extract(self.variance_t.to(device), t, x_t.shape)
        sigma_t = torch.sqrt(var_t)
        source_recon = (x_t - (1.0 - m_t) * target - sigma_t * model_output) / (m_t + 1e-8)
        if clip_denoised:
            source_recon = source_recon.clamp(-1.0, 1.0)
        if step_index >= len(self.asc_steps) - 1:
            return BDBMSchedulerOutput(prev_sample=source_recon, pred_original_sample=source_recon)
        n_t = torch.full((x_t.shape[0],), self.asc_steps[step_index + 1].item(), device=device, dtype=torch.long)
        m_nt = _extract(self.m_t.to(device), n_t, x_t.shape)
        var_nt = _extract(self.variance_t.to(device), n_t, x_t.shape)
        noise = torch.randn(x_t.shape, device=device, generator=generator, dtype=x_t.dtype)
        sigma2_t = 2 * (1 - m_nt) * (m_nt - m_t) / (1 - m_t + 1e-8)
        sigma_new = torch.sqrt(sigma2_t.clamp(min=0)) * self.eta
        coe_eps = torch.sqrt((var_nt - sigma_new ** 2).clamp(min=0) / (var_t + 1e-8))
        x_next = (1.0 - m_nt) * target + m_nt * source_recon + coe_eps * (x_t - (1.0 - m_t) * target - m_t * source_recon) + sigma_new * noise
        return BDBMSchedulerOutput(prev_sample=x_next, pred_original_sample=source_recon)

    def set_timesteps(self, num_inference_steps: Optional[int] = None) -> None:
        if num_inference_steps is not None:
            self.sample_step = num_inference_steps
            self.skip_sample = True
        self._build_steps()

    def add_noise(
        self,
        target: torch.Tensor,
        source: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        m_t = _extract(self.m_t.to(target.device), timesteps, target.shape)
        var_t = _extract(self.variance_t.to(target.device), timesteps, target.shape)
        sigma_t = torch.sqrt(var_t)
        noise = torch.randn_like(target)
        return (1.0 - m_t) * target + m_t * source + sigma_t * noise


# ---------------------------------------------------------------------------
# BDBMPipeline
# ---------------------------------------------------------------------------


@dataclass
class BDBMPipelineOutput(BaseOutput):
    images: Union[List[Image.Image], np.ndarray, torch.Tensor]


class BDBMPipeline(DiffusionPipeline):
    """Bidirectional image-to-image pipeline for BDBM."""

    def __init__(self, unet: BDBMUNet, scheduler: BDBMScheduler) -> None:
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    def _make_context(self, endpoint: torch.Tensor, direction: str) -> Optional[torch.Tensor]:
        mode = getattr(self.unet, "condition_mode", "concat")
        if mode in (None, "nocond"):
            return None
        if mode == "concat":
            return endpoint
        if mode == "dual":
            zeros = torch.zeros_like(endpoint)
            if direction == "b2a":
                return torch.cat((zeros, endpoint), dim=1)
            return torch.cat((endpoint, zeros), dim=1)
        raise ValueError(f"Unknown condition_mode: {mode}")

    @torch.no_grad()
    def __call__(
        self,
        source_image: torch.Tensor,
        direction: str = "b2a",
        num_inference_steps: Optional[int] = None,
        clip_denoised: bool = False,
        output_type: str = "pt",
        generator: Optional[torch.Generator] = None,
    ) -> BDBMPipelineOutput:
        if num_inference_steps is not None:
            self.scheduler.set_timesteps(num_inference_steps)
        if self.scheduler.steps is None:
            raise RuntimeError("Scheduler steps not initialised; call set_timesteps().")
        if direction == "b2a":
            return self._sample_b2a(source_image, clip_denoised, output_type, generator)
        if direction == "a2b":
            return self._sample_a2b(source_image, clip_denoised, output_type, generator)
        raise ValueError(f"Unknown direction: {direction!r}; expected 'b2a' or 'a2b'.")

    def _sample_b2a(self, source, clip_denoised, output_type, generator):
        img = source.clone()
        context = self._make_context(source, direction="b2a")
        steps = self.scheduler.steps
        for i in tqdm(range(len(steps)), desc="BDBM b2a sampling"):
            t = torch.full((img.shape[0],), int(steps[i].item()), device=img.device, dtype=torch.long)
            model_output = self.unet(img, t, context=context)
            result = self.scheduler.step_b2a(
                model_output=model_output, step_index=i, x_t=img,
                source=source, clip_denoised=clip_denoised, generator=generator,
            )
            img = result.prev_sample
        return self._format_output(img, output_type)

    def _sample_a2b(self, target, clip_denoised, output_type, generator):
        img = target.clone()
        context = self._make_context(target, direction="a2b")
        asc_steps = self.scheduler.asc_steps
        for i in tqdm(range(len(asc_steps)), desc="BDBM a2b sampling"):
            t = torch.full((img.shape[0],), int(asc_steps[i].item()), device=img.device, dtype=torch.long)
            model_output = self.unet(img, t, context=context)
            result = self.scheduler.step_a2b(
                model_output=model_output, step_index=i, x_t=img,
                target=target, clip_denoised=clip_denoised, generator=generator,
            )
            img = result.prev_sample
        return self._format_output(img, output_type)

    @staticmethod
    def _format_output(images, output_type):
        if output_type == "pt":
            return BDBMPipelineOutput(images=images)
        images_np = ((images + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        images_np = images_np.permute(0, 2, 3, 1).cpu().numpy()
        if output_type == "np":
            return BDBMPipelineOutput(images=images_np)
        pil_images = []
        for arr in images_np:
            if arr.shape[2] == 1:
                arr = arr.squeeze(2)
            pil_images.append(Image.fromarray(arr))
        return BDBMPipelineOutput(images=pil_images)
