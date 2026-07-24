# Credits: SPADE from Park et al. "Semantic Image Synthesis with Spatially-Adaptive Normalization" CVPR 2019.
#
"""SPADE single-pass inference pipeline and checkpoint loader."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Union

import numpy as np
import torch
from PIL import Image

from diffusers import DiffusionPipeline
from diffusers.utils import BaseOutput, pt_to_pil

from src.models.spade import SPADEGenerator


@dataclass
class SPADEPipelineOutput(BaseOutput):
    """Output of SPADE pipeline."""

    images: Any


class SPADEPipeline(DiffusionPipeline):
    """Single-pass SPADE inference pipeline for semantic image synthesis."""

    def __init__(self, generator: SPADEGenerator) -> None:
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
        strict: bool = False,
        **kwargs,
    ) -> "SPADEPipeline":
        """Load SPADE pipeline from local config + safetensors."""
        model_dir = Path(pretrained_model_name_or_path)
        if subfolder:
            model_dir = model_dir / subfolder

        config_path = model_dir / "config.json"
        weights_path = model_dir / "diffusion_pytorch_model.safetensors"
        if not (config_path.exists() and weights_path.exists()):
            return super().from_pretrained(pretrained_model_name_or_path, subfolder=subfolder, **kwargs)

        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)

        generator = SPADEGenerator.from_config(config)
        from safetensors.torch import load_file

        state_dict = load_file(str(weights_path), device="cpu")
        generator.load_state_dict(state_dict, strict=strict)
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
    def prepare_segmap(segmap, device, dtype):
        """Convert inputs to tensors in [-1, 1] or [0, 1] depending on range."""
        if isinstance(segmap, Image.Image):
            segmap = [segmap]
        if isinstance(segmap, list) and isinstance(segmap[0], Image.Image):
            tensors = []
            for img in segmap:
                arr = np.array(img).astype(np.float32) / 255.0
                if arr.ndim == 2:
                    t = torch.from_numpy(arr).unsqueeze(0)
                else:
                    t = torch.from_numpy(arr).permute(2, 0, 1)
                tensors.append(t)
            segmap = torch.stack(tensors)
        if isinstance(segmap, np.ndarray):
            segmap = torch.from_numpy(segmap)
        if segmap.max() > 1.0:
            segmap = segmap / 255.0
        return segmap.to(device=device, dtype=dtype)

    @torch.no_grad()
    def __call__(
        self,
        segmap: Union[torch.Tensor, Image.Image, List[Image.Image]],
        z: torch.Tensor | None = None,
        output_type: str = "pil",
        return_dict: bool = True,
    ) -> Union[SPADEPipelineOutput, tuple]:
        device = self.device
        dtype = self.dtype
        seg = self.prepare_segmap(segmap, device, dtype)
        images = self.generator(seg, z=z).clamp(-1, 1)

        if output_type == "pil":
            images = pt_to_pil(images)
        elif output_type == "np":
            images = self._convert_to_numpy(images)

        if not return_dict:
            return (images,)
        return SPADEPipelineOutput(images=images)

    @staticmethod
    def _convert_to_numpy(images: torch.Tensor) -> np.ndarray:
        images = (images + 1) / 2
        return images.clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()


def load_spade_pipeline(
    checkpoint_path: str | Path,
    *,
    device: str = "cuda",
    **generator_kwargs,
) -> SPADEPipeline:
    """Load a SPADE generator checkpoint into a pipeline."""
    ckpt_path = Path(checkpoint_path)
    if ckpt_path.is_dir() and (ckpt_path / "generator" / "config.json").exists():
        return SPADEPipeline.from_pretrained(ckpt_path, device=device)

    config_path = ckpt_path.parent / "generator" / "config.json"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        generator_kwargs = {**config, **generator_kwargs}

    generator = SPADEGenerator(**generator_kwargs)
    if ckpt_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state_dict = load_file(str(ckpt_path), device=device)
    else:
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
        if isinstance(state_dict, dict) and "generator" in state_dict:
            state_dict = state_dict["generator"]
    generator.load_state_dict(state_dict, strict=True)
    generator = generator.to(device).eval()
    return SPADEPipeline(generator=generator)
