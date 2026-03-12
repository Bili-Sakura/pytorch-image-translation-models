# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""BBDM inference pipeline for one-way image-to-image translation.

BBDM performs reverse Brownian-bridge sampling from source ``y`` to target
``x0``. This pipeline is intentionally one-directional and separate from
``BiBBDM``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from PIL import Image
from tqdm.auto import tqdm

from diffusers import DiffusionPipeline
from diffusers.utils import BaseOutput

from src.models.unet import BBDMUNet
from src.schedulers.bbdm import BBDMScheduler


@dataclass
class BBDMPipelineOutput(BaseOutput):
    """Output of the BBDM pipeline."""

    images: Any


class BBDMPipeline(DiffusionPipeline):
    """One-way source -> target pipeline for BBDM."""

    def __init__(self, unet: torch.nn.Module, scheduler: BBDMScheduler) -> None:
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
    ) -> "BBDMPipeline":
        """Load BBDM pipeline from local checkpoint directories."""
        unet = BBDMUNet.from_pretrained(pretrained_model_name_or_path, subfolder=subfolder)
        try:
            scheduler = BBDMScheduler.from_pretrained(
                pretrained_model_name_or_path, subfolder=scheduler_subfolder
            )
        except Exception:
            scheduler = BBDMScheduler()
        unet = unet.eval().to(device=device)
        if torch_dtype is not None:
            unet = unet.to(dtype=torch_dtype)
        return cls(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        source_image: torch.Tensor,
        num_inference_steps: Optional[int] = None,
        clip_denoised: bool = False,
        output_type: str = "pt",
        generator: Optional[torch.Generator] = None,
    ) -> BBDMPipelineOutput:
        """Run reverse BBDM sampling from source to target."""
        if num_inference_steps is not None:
            self.scheduler.set_timesteps(num_inference_steps)
        steps = self.scheduler.steps
        if steps is None:
            raise RuntimeError("Scheduler steps not initialised; call set_timesteps().")

        img = source_image.clone()
        p = next(self.unet.parameters())
        model_device, model_dtype = p.device, p.dtype
        context = source_image

        for i in tqdm(range(len(steps)), desc="BBDM sampling", total=len(steps)):
            t = torch.full((img.shape[0],), int(steps[i].item()), device=img.device, dtype=torch.long)
            model_output = self.unet(
                img.to(device=model_device, dtype=model_dtype),
                t,
                context=context.to(device=model_device, dtype=model_dtype),
            )
            model_output = model_output.to(device=img.device, dtype=img.dtype)
            result = self.scheduler.step(
                model_output=model_output,
                step_index=i,
                x_t=img,
                source=source_image,
                clip_denoised=clip_denoised,
                generator=generator,
            )
            img = result.prev_sample

        return self._format_output(img, output_type)

    @staticmethod
    def _format_output(images: torch.Tensor, output_type: str) -> BBDMPipelineOutput:
        if output_type == "pt":
            return BBDMPipelineOutput(images=images)
        images_np = ((images + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        images_np = images_np.permute(0, 2, 3, 1).cpu().numpy()
        if output_type == "np":
            return BBDMPipelineOutput(images=images_np)
        pil_images: list[Image.Image] = []
        for arr in images_np:
            if arr.shape[2] == 1:
                arr = arr.squeeze(2)
            pil_images.append(Image.fromarray(arr))
        return BBDMPipelineOutput(images=pil_images)
