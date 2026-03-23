# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""BiBBDM inference pipeline for bidirectional image-to-image translation.

Supports bidirectional image-to-image translation using the Brownian Bridge
diffusion process. The default direction is **b2a** (source → target).

Note
----
This pipeline is bidirectional and intentionally separate from
``src.pipelines.bbdm.BBDMPipeline`` (one-way BBDM).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
import torch
from tqdm.auto import tqdm

from diffusers import DiffusionPipeline
from diffusers.utils import BaseOutput, pt_to_pil

from src.models.unet import BiBBDMUNet
from src.schedulers.bibbdm import BiBBDMScheduler


@dataclass
class BiBBDMPipelineOutput(BaseOutput):
    """Output of the BiBBDM pipeline.

    Attributes
    ----------
    images : list[PIL.Image.Image] | np.ndarray | torch.Tensor
        Generated images.
    """

    images: Any


class BiBBDMPipeline(DiffusionPipeline):
    """Bidirectional image-to-image pipeline using BiBBDM.

    Parameters
    ----------
    unet : torch.nn.Module
        The denoising model.
    scheduler : BiBBDMScheduler
        The Brownian Bridge noise scheduler.
    """

    def __init__(self, unet: torch.nn.Module, scheduler: BiBBDMScheduler) -> None:
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
    ) -> "BiBBDMPipeline":
        """Load BiBBDM pipeline from local checkpoint directories."""
        unet = BiBBDMUNet.from_pretrained(pretrained_model_name_or_path, subfolder=subfolder)
        try:
            scheduler = BiBBDMScheduler.from_pretrained(
                pretrained_model_name_or_path, subfolder=scheduler_subfolder
            )
        except Exception:
            scheduler = BiBBDMScheduler()
        unet = unet.eval().to(device=device)
        if torch_dtype is not None:
            unet = unet.to(dtype=torch_dtype)
        return cls(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        source_image: torch.Tensor,
        direction: str = "b2a",
        num_inference_steps: Optional[int] = None,
        clip_denoised: bool = False,
        output_type: str = "pt",
        generator: Optional[torch.Generator] = None,
    ) -> BiBBDMPipelineOutput:
        """Run the BiBBDM sampling loop.

        Parameters
        ----------
        source_image : Tensor (B, C, H, W) in [-1, 1]
            The source/conditioning image.
        direction : str
            ``"b2a"`` for source→target or ``"a2b"`` for reverse.
        num_inference_steps : int or None
            Override the scheduler's default step count.
        clip_denoised : bool
            Clamp intermediate predictions to [-1, 1].
        output_type : str
            ``"pt"`` | ``"pil"`` | ``"np"``.
        generator : torch.Generator or None
            RNG for reproducibility.
        """
        if num_inference_steps is not None:
            self.scheduler.set_timesteps(num_inference_steps)

        steps = self.scheduler.steps
        if steps is None:
            raise RuntimeError("Scheduler steps not initialised; call set_timesteps().")

        if direction == "b2a":
            return self._sample_b2a(
                source_image, steps, clip_denoised, output_type, generator,
            )
        if direction == "a2b":
            return self._sample_a2b(
                source_image, steps, clip_denoised, output_type, generator,
            )
        raise ValueError(f"Unknown direction: {direction!r}; expected 'b2a' or 'a2b'.")

    # ------------------------------------------------------------------

    def _sample_b2a(self, source, steps, clip_denoised, output_type, generator):
        """Source → Target (reverse Brownian Bridge)."""
        img = source.clone()
        p = next(self.unet.parameters())
        model_device, model_dtype = p.device, p.dtype
        for i in tqdm(range(len(steps)), desc="B2A sampling", total=len(steps)):
            t = torch.full(
                (img.shape[0],), steps[i].item(), device=img.device, dtype=torch.long,
            )
            model_output = self.unet(
                img.to(device=model_device, dtype=model_dtype),
                t,
                context=source.to(device=model_device, dtype=model_dtype),
            )
            model_output = model_output.to(device=img.device, dtype=img.dtype)
            result = self.scheduler.step_b2a(
                model_output, step_index=i, x_t=img, source=source,
                clip_denoised=clip_denoised, generator=generator,
            )
            img = result.prev_sample
        return self._format_output(img, output_type)

    def _sample_a2b(self, target, steps, clip_denoised, output_type, generator):
        """Target → Source (forward Brownian Bridge)."""
        img = target.clone()
        p = next(self.unet.parameters())
        model_device, model_dtype = p.device, p.dtype
        for i in tqdm(reversed(range(len(steps))), desc="A2B sampling", total=len(steps)):
            t = torch.full(
                (img.shape[0],), steps[i].item(), device=img.device, dtype=torch.long,
            )
            model_output = self.unet(
                img.to(device=model_device, dtype=model_dtype),
                t,
                context=target.to(device=model_device, dtype=model_dtype),
            )
            model_output = model_output.to(device=img.device, dtype=img.dtype)
            result = self.scheduler.step_a2b(
                model_output, step_index=i, x_t=img, target=target,
                clip_denoised=clip_denoised, generator=generator,
            )
            img = result.prev_sample
        return self._format_output(img, output_type)

    # ------------------------------------------------------------------

    @staticmethod
    def _format_output(images: torch.Tensor, output_type: str) -> BiBBDMPipelineOutput:
        if output_type == "pt":
            return BiBBDMPipelineOutput(images=images)
        if output_type == "pil":
            return BiBBDMPipelineOutput(images=pt_to_pil(images))
        images_np = ((images + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        images_np = images_np.permute(0, 2, 3, 1).cpu().numpy()
        return BiBBDMPipelineOutput(images=images_np)
