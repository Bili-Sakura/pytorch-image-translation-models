# Credits: StarGAN (Choi et al., CVPR 2018) - https://github.com/yunjey/stargan
#
"""StarGAN single-pass inference pipeline and checkpoint loader."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Union

import numpy as np
import torch
from PIL import Image

from diffusers import DiffusionPipeline
from diffusers.utils import BaseOutput

from src.models.stargan import StarGANGenerator


@dataclass
class StarGANPipelineOutput(BaseOutput):
    """Output of StarGAN pipeline."""

    images: Any


class StarGANPipeline(DiffusionPipeline):
    """Single-pass StarGAN inference pipeline."""

    def __init__(self, generator: StarGANGenerator) -> None:
        super().__init__()
        self.register_modules(generator=generator)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        subfolder: str = "generator",
        device: str | torch.device = "cpu",
        torch_dtype: torch.dtype | None = None,
        **kwargs,
    ) -> "StarGANPipeline":
        """Load StarGAN pipeline from local config + safetensors."""
        model_dir = Path(pretrained_model_name_or_path)
        if subfolder:
            model_dir = model_dir / subfolder

        config_path = model_dir / "config.json"
        weights_path = model_dir / "diffusion_pytorch_model.safetensors"
        if not (config_path.exists() and weights_path.exists()):
            return super().from_pretrained(pretrained_model_name_or_path, subfolder=subfolder, **kwargs)

        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)

        generator = StarGANGenerator(
            conv_dim=config.get("conv_dim", 64),
            c_dim=config.get("c_dim", 5),
            repeat_num=config.get("repeat_num", 6),
        )
        from safetensors.torch import load_file

        state_dict = load_file(str(weights_path), device="cpu")
        generator.load_state_dict(state_dict, strict=True)
        generator = generator.eval().to(device=device)
        if torch_dtype is not None:
            generator = generator.to(dtype=torch_dtype)
        return cls(generator=generator)

    @property
    def device(self) -> torch.device:
        return next(self.generator.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.generator.parameters()).dtype

    @staticmethod
    def prepare_inputs(image, device, dtype):
        """Convert inputs to tensors in [-1, 1]."""
        if isinstance(image, Image.Image):
            image = [image]
        if isinstance(image, list) and isinstance(image[0], Image.Image):
            images = []
            for img in image:
                arr = np.array(img).astype(np.float32) / 255.0
                if arr.ndim == 2:
                    t = torch.from_numpy(arr).unsqueeze(0)
                else:
                    t = torch.from_numpy(arr).permute(2, 0, 1)
                images.append(t)
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
        target_labels: torch.Tensor,
        output_type: str = "pil",
        return_dict: bool = True,
    ) -> Union[StarGANPipelineOutput, tuple]:
        """Translate source image(s) toward target domain labels."""
        device = self.device
        dtype = self.dtype
        x = self.prepare_inputs(source_image, device, dtype)
        labels = target_labels.to(device=device, dtype=dtype)
        images = self.generator(x, labels).clamp(-1, 1)

        if output_type == "pil":
            images = self._convert_to_pil(images)
        elif output_type == "np":
            images = self._convert_to_numpy(images)

        if not return_dict:
            return (images,)
        return StarGANPipelineOutput(images=images)

    @staticmethod
    def _convert_to_pil(images: torch.Tensor) -> List[Image.Image]:
        images = (images + 1) / 2
        images = images.clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).round().astype(np.uint8)
        pil_images = []
        for img in images:
            if img.shape[2] == 1:
                pil_images.append(Image.fromarray(img.squeeze(2), mode="L"))
            else:
                pil_images.append(Image.fromarray(img))
        return pil_images

    @staticmethod
    def _convert_to_numpy(images: torch.Tensor) -> np.ndarray:
        images = (images + 1) / 2
        return images.clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()


def load_stargan_pipeline(
    checkpoint_path: str | Path,
    *,
    conv_dim: int = 64,
    c_dim: int = 5,
    repeat_num: int = 6,
    device: str = "cuda",
) -> StarGANPipeline:
    """Load StarGAN generator checkpoint into a native pipeline."""
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    generator = StarGANGenerator(conv_dim=conv_dim, c_dim=c_dim, repeat_num=repeat_num)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)

    if isinstance(state, dict):
        if "G" in state and isinstance(state["G"], dict):
            state = state["G"]
        elif "generator" in state and isinstance(state["generator"], dict):
            state = state["generator"]

    if not isinstance(state, dict):
        raise ValueError("Unsupported checkpoint format. Expected a state_dict-like object.")

    generator.load_state_dict(state, strict=False)
    generator = generator.to(device).eval()
    return StarGANPipeline(generator=generator)

