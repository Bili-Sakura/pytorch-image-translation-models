# Copyright (c) 2026 EarthBridge Team.
# Credits: img2img-turbo from Parmar et al. 2024 —
# https://github.com/GaParmar/img2img-turbo

"""CycleGAN-Turbo one-step unpaired inference pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image

from diffusers import DiffusionPipeline
from diffusers.utils import BaseOutput, pt_to_pil

from src.models.img2img_turbo import CycleGANTurbo, build_transform


@dataclass
class CycleGANTurboPipelineOutput(BaseOutput):
    images: Union[List[Image.Image], np.ndarray, torch.Tensor]


class CycleGANTurboPipeline(DiffusionPipeline):
    """One-step unpaired translation with CycleGAN-Turbo (SD-Turbo + LoRA)."""

    def __init__(self, model: CycleGANTurbo, *, image_prep: str = "resize_512x512") -> None:
        super().__init__()
        self.register_modules(model=model)
        self.image_prep = image_prep
        self._transform = build_transform(image_prep)

    @classmethod
    def from_pretrained(
        cls,
        *,
        pretrained_name: Optional[str] = None,
        pretrained_path: Optional[str | Path] = None,
        ckpt_folder: str = "checkpoints",
        device: str | torch.device = "cpu",
        image_prep: str = "resize_512x512",
        **kwargs,
    ) -> "CycleGANTurboPipeline":
        if (pretrained_name is None) == (pretrained_path is None):
            raise ValueError("Provide exactly one of pretrained_name or pretrained_path")
        model = CycleGANTurbo(
            pretrained_name=pretrained_name,
            pretrained_path=str(pretrained_path) if pretrained_path else None,
            ckpt_folder=ckpt_folder,
            device=device,
            **kwargs,
        )
        model.eval()
        return cls(model=model, image_prep=image_prep)

    @property
    def device(self) -> torch.device:
        return self.model.device

    def prepare_inputs(
        self,
        image: Union[torch.Tensor, Image.Image, List[Image.Image]],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if isinstance(image, Image.Image):
            image = [image]
        if isinstance(image, list) and isinstance(image[0], Image.Image):
            tensors = []
            for img in image:
                img = self._transform(img.convert("RGB"))
                t = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
                t = (t - 0.5) / 0.5
                tensors.append(t)
            image = torch.stack(tensors)
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        if image.ndim == 3:
            image = image.unsqueeze(0)
        return image.to(device=device, dtype=dtype)

    @torch.no_grad()
    def __call__(
        self,
        source_image: Union[torch.Tensor, Image.Image, List[Image.Image]],
        *,
        direction: Optional[str] = None,
        prompt: Optional[str] = None,
        output_type: str = "pil",
        return_dict: bool = True,
    ) -> Union[CycleGANTurboPipelineOutput, tuple]:
        dtype = next(self.model.unet.parameters()).dtype
        x = self.prepare_inputs(source_image, self.device, dtype)
        out = self.model(x, direction=direction, caption=prompt)
        if output_type == "pt":
            images = out
        elif output_type == "np":
            images = (out * 0.5 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
        else:
            images = pt_to_pil((out * 0.5 + 0.5).clamp(0, 1))
        if not return_dict:
            return (images,)
        return CycleGANTurboPipelineOutput(images=images)


__all__ = ["CycleGANTurboPipeline", "CycleGANTurboPipelineOutput"]
