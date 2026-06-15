# Copyright (c) 2026 EarthBridge Team.
# Credits: img2img-turbo from Parmar et al. 2024 —
# https://github.com/GaParmar/img2img-turbo

"""pix2pix-turbo one-step paired inference pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from diffusers import DiffusionPipeline
from diffusers.utils import BaseOutput, pt_to_pil

from src.models.img2img_turbo import Pix2PixTurbo, canny_from_pil


@dataclass
class Pix2PixTurboPipelineOutput(BaseOutput):
    images: Union[List[Image.Image], np.ndarray, torch.Tensor]


class Pix2PixTurboPipeline(DiffusionPipeline):
    """One-step paired translation with pix2pix-turbo (SD-Turbo + LoRA)."""

    def __init__(self, model: Pix2PixTurbo) -> None:
        super().__init__()
        self.register_modules(model=model)

    @classmethod
    def from_pretrained(
        cls,
        *,
        pretrained_name: Optional[str] = None,
        pretrained_path: Optional[str | Path] = None,
        ckpt_folder: str = "checkpoints",
        device: str | torch.device = "cpu",
        **kwargs,
    ) -> "Pix2PixTurboPipeline":
        if (pretrained_name is None) == (pretrained_path is None):
            raise ValueError("Provide exactly one of pretrained_name or pretrained_path")
        model = Pix2PixTurbo(
            pretrained_name=pretrained_name,
            pretrained_path=str(pretrained_path) if pretrained_path else None,
            ckpt_folder=ckpt_folder,
            device=device,
            **kwargs,
        )
        model.set_eval()
        return cls(model=model)

    @property
    def device(self) -> torch.device:
        return self.model.device

    @staticmethod
    def _resize_to_multiple_of_8(image: Image.Image) -> Image.Image:
        w = image.width - image.width % 8
        h = image.height - image.height % 8
        return image.resize((w, h), Image.LANCZOS)

    @torch.no_grad()
    def __call__(
        self,
        source_image: Union[torch.Tensor, Image.Image],
        *,
        prompt: str,
        model_mode: Optional[str] = None,
        gamma: float = 0.4,
        seed: Optional[int] = None,
        canny_low: int = 100,
        canny_high: int = 200,
        output_type: str = "pil",
        return_dict: bool = True,
    ) -> Union[Pix2PixTurboPipelineOutput, tuple]:
        mode = model_mode or self.model.pretrained_name
        dtype = next(self.model.unet.parameters()).dtype

        if isinstance(source_image, Image.Image):
            source_image = self._resize_to_multiple_of_8(source_image.convert("RGB"))

        if mode == "edge_to_image" and isinstance(source_image, Image.Image):
            canny = canny_from_pil(source_image, canny_low, canny_high)
            c_t = TF.to_tensor(canny).unsqueeze(0).to(device=self.device, dtype=dtype)
        elif mode == "sketch_to_image_stochastic" and isinstance(source_image, Image.Image):
            c_t = (TF.to_tensor(source_image) < 0.5).unsqueeze(0).float().to(device=self.device, dtype=dtype)
            if seed is not None:
                torch.manual_seed(seed)
            _, _, h, w = c_t.shape
            noise = torch.randn((1, 4, h // 8, w // 8), device=self.device, dtype=dtype)
            out = self.model(c_t, prompt=prompt, deterministic=False, r=gamma, noise_map=noise)
            return self._format_output(out, output_type, return_dict)
        elif isinstance(source_image, Image.Image):
            c_t = TF.to_tensor(source_image).unsqueeze(0).to(device=self.device, dtype=dtype)
        else:
            c_t = source_image.to(device=self.device, dtype=dtype)
            if c_t.ndim == 3:
                c_t = c_t.unsqueeze(0)

        out = self.model(c_t, prompt=prompt, deterministic=True)
        return self._format_output(out, output_type, return_dict)

    def _format_output(self, out, output_type, return_dict):
        if output_type == "pt":
            images = out
        elif output_type == "np":
            images = (out * 0.5 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
        else:
            images = pt_to_pil((out * 0.5 + 0.5).clamp(0, 1))
        if not return_dict:
            return (images,)
        return Pix2PixTurboPipelineOutput(images=images)


__all__ = ["Pix2PixTurboPipeline", "Pix2PixTurboPipelineOutput"]
