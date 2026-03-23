# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""BDBM inference pipeline for bidirectional image-to-image translation.

Implements the Bidirectional Diffusion Bridge Models (BDBM) sampling loop
with both forward (a2b) and reverse (b2a) directions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import torch
from tqdm.auto import tqdm

from diffusers import DiffusionPipeline
from diffusers.utils import BaseOutput, pt_to_pil

from src.models.unet import BDBMUNet
from src.schedulers.bdbm import BDBMScheduler


@dataclass
class BDBMPipelineOutput(BaseOutput):
    """Output of the BDBM pipeline.

    Attributes
    ----------
    images : list[PIL.Image.Image] | np.ndarray | torch.Tensor
        Generated images.
    """

    images: Any


class BDBMPipeline(DiffusionPipeline):
    """Bidirectional image-to-image pipeline for BDBM.

    Parameters
    ----------
    unet : torch.nn.Module
        The denoising model.
    scheduler : BDBMScheduler
        The BDBM noise scheduler.
    """

    def __init__(self, unet: torch.nn.Module, scheduler: BDBMScheduler) -> None:
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        subfolder: str = "unet",
        scheduler_subfolder: str = "scheduler",
        device: str | torch.device = "cpu",
        torch_dtype: torch.dtype | None = None,
        **kwargs,
    ) -> "BDBMPipeline":
        """Load BDBM pipeline from local checkpoint directories."""
        unet = BDBMUNet.from_pretrained(pretrained_model_name_or_path, subfolder=subfolder)
        try:
            scheduler = BDBMScheduler.from_pretrained(
                pretrained_model_name_or_path, subfolder=scheduler_subfolder
            )
        except Exception:
            scheduler = BDBMScheduler()
        unet = unet.eval().to(device=device)
        if torch_dtype is not None:
            unet = unet.to(dtype=torch_dtype)
        return cls(unet=unet, scheduler=scheduler)

    def _make_context(
        self, endpoint: torch.Tensor, direction: str,
    ) -> Optional[torch.Tensor]:
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
        """Run bidirectional BDBM sampling.

        Parameters
        ----------
        source_image : Tensor (B, C, H, W) in [-1, 1]
            Input image.
        direction : str
            ``"b2a"`` for source→target or ``"a2b"`` for reverse.
        num_inference_steps : int or None
            Override the scheduler's step count.
        output_type : str
            ``"pt"`` | ``"pil"`` | ``"np"``.
        """
        if num_inference_steps is not None:
            self.scheduler.set_timesteps(num_inference_steps)

        if self.scheduler.steps is None or self.scheduler.asc_steps is None:
            raise RuntimeError("Scheduler steps not initialised; call set_timesteps().")

        if direction == "b2a":
            return self._sample_b2a(
                source_image, self.scheduler.steps, clip_denoised, output_type, generator,
            )
        if direction == "a2b":
            return self._sample_a2b(
                source_image, self.scheduler.asc_steps, clip_denoised, output_type, generator,
            )
        raise ValueError(f"Unknown direction: {direction!r}; expected 'b2a' or 'a2b'.")

    def _sample_b2a(self, source, steps, clip_denoised, output_type, generator):
        """Source → Target direction."""
        img = source.clone()
        context = self._make_context(source, direction="b2a")
        model_device = next(self.unet.parameters()).device
        model_dtype = next(self.unet.parameters()).dtype
        for i in tqdm(range(len(steps)), desc="BDBM b2a sampling"):
            t = torch.full(
                (img.shape[0],), int(steps[i].item()),
                device=img.device, dtype=torch.long,
            )
            model_output = self.unet(
                img.to(device=model_device, dtype=model_dtype),
                t,
                context=context.to(device=model_device, dtype=model_dtype)
                if context is not None else None,
            )
            model_output = model_output.to(device=img.device, dtype=img.dtype)
            result = self.scheduler.step_b2a(
                model_output=model_output, step_index=i, x_t=img,
                source=source, clip_denoised=clip_denoised, generator=generator,
            )
            img = result.prev_sample
        return self._format_output(img, output_type)

    def _sample_a2b(self, target, asc_steps, clip_denoised, output_type, generator):
        """Target → Source direction."""
        img = target.clone()
        context = self._make_context(target, direction="a2b")
        model_device = next(self.unet.parameters()).device
        model_dtype = next(self.unet.parameters()).dtype
        for i in tqdm(range(len(asc_steps)), desc="BDBM a2b sampling"):
            t = torch.full(
                (img.shape[0],), int(asc_steps[i].item()),
                device=img.device, dtype=torch.long,
            )
            model_output = self.unet(
                img.to(device=model_device, dtype=model_dtype),
                t,
                context=context.to(device=model_device, dtype=model_dtype)
                if context is not None else None,
            )
            model_output = model_output.to(device=img.device, dtype=img.dtype)
            result = self.scheduler.step_a2b(
                model_output=model_output, step_index=i, x_t=img,
                target=target, clip_denoised=clip_denoised, generator=generator,
            )
            img = result.prev_sample
        return self._format_output(img, output_type)

    @staticmethod
    def _format_output(images: torch.Tensor, output_type: str) -> BDBMPipelineOutput:
        if output_type == "pt":
            return BDBMPipelineOutput(images=images)
        if output_type == "pil":
            return BDBMPipelineOutput(images=pt_to_pil(images))
        images_np = ((images + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        images_np = images_np.permute(0, 2, 3, 1).cpu().numpy()
        return BDBMPipelineOutput(images=images_np)
