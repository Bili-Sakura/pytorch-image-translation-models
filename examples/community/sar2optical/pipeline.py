# Credits: SAR2Optical (https://github.com/yuuIind/SAR2Optical), pix2pix (Isola et al.).
"""Inference pipeline for SAR2Optical pix2pix generator."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from PIL import Image

from diffusers import DiffusionPipeline
from diffusers.utils import BaseOutput, pt_to_pil

from examples.community.sar2optical.model import SAR2OpticalGenerator


@dataclass
class SAR2OpticalPipelineOutput(BaseOutput):
    """Output container for SAR2Optical pipeline inference."""

    images: Union[List[Image.Image], np.ndarray, torch.Tensor]


class SAR2OpticalPipeline(DiffusionPipeline):
    """Single-pass SAR-to-optical translation pipeline."""

    model_cpu_offload_seq = "generator"

    def __init__(self, generator: SAR2OpticalGenerator) -> None:
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
    ) -> "SAR2OpticalPipeline":
        """Load a SAR2Optical pipeline from local config + safetensors.

        Expected layout:
        ``<path>/<subfolder>/config.json`` and
        ``<path>/<subfolder>/diffusion_pytorch_model.safetensors``.
        """
        model_dir = Path(pretrained_model_name_or_path)
        if subfolder:
            model_dir = model_dir / subfolder

        config_path = model_dir / "config.json"
        weights_path = model_dir / "diffusion_pytorch_model.safetensors"
        if not (config_path.exists() and weights_path.exists()):
            return super().from_pretrained(pretrained_model_name_or_path, subfolder=subfolder, **kwargs)

        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)

        generator = SAR2OpticalGenerator(
            c_in=config.get("c_in", 3),
            c_out=config.get("c_out", 3),
            use_upsampling=config.get("use_upsampling", False),
            mode=config.get("mode", "nearest"),
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

    def prepare_inputs(
        self,
        image: Union[torch.Tensor, Image.Image, List[Image.Image], np.ndarray],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Convert input images to tensor in [-1, 1]."""
        if isinstance(image, Image.Image):
            image = [image]

        if isinstance(image, list) and image and isinstance(image[0], Image.Image):
            images = []
            for img in image:
                img_array = np.array(img, dtype=np.float32)
                if img_array.max() > 1.0:
                    img_array = img_array / 255.0
                if img_array.ndim == 2:
                    img_array = img_array[:, :, np.newaxis]
                images.append(torch.from_numpy(img_array).permute(2, 0, 1))
            image = torch.stack(images)

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)

        if image.ndim == 3:
            image = image.unsqueeze(0)

        # HWC -> BCHW for ndarray/tensor inputs coming from image arrays
        if image.ndim == 4 and image.shape[-1] in (1, 3, 4) and image.shape[1] not in (1, 3, 4):
            image = image.permute(0, 3, 1, 2)

        if image.min() >= 0 and image.max() <= 1.0:
            image = image * 2 - 1
        elif image.max() > 1.0:
            image = image / 255.0 * 2 - 1

        return image.to(device=device, dtype=dtype)

    @torch.no_grad()
    def __call__(
        self,
        source_image: Union[torch.Tensor, Image.Image, List[Image.Image], np.ndarray],
        output_type: str = "pil",
        return_dict: bool = True,
    ) -> Union[SAR2OpticalPipelineOutput, tuple]:
        x = self.prepare_inputs(source_image, self.device, self.dtype)
        self.generator.eval()
        images = self.generator(x).clamp(-1, 1)

        if output_type == "pil":
            images = pt_to_pil(images)
        elif output_type == "np":
            images = self._convert_to_numpy(images)

        if not return_dict:
            return (images,)
        return SAR2OpticalPipelineOutput(images=images)

    @staticmethod
    def _convert_to_numpy(images: torch.Tensor) -> np.ndarray:
        return ((images + 1) / 2).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()


def load_sar2optical_pipeline(
    checkpoint_path: str,
    c_in: int = 3,
    c_out: int = 3,
    use_upsampling: bool = False,
    mode: str = "nearest",
    device: str = "cpu",
) -> SAR2OpticalPipeline:
    """Load SAR2Optical generator checkpoint into a pipeline."""
    generator = SAR2OpticalGenerator(
        c_in=c_in,
        c_out=c_out,
        use_upsampling=use_upsampling,
        mode=mode,
    )
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)

    if isinstance(state, dict):
        if "generator" in state and isinstance(state["generator"], dict):
            state = state["generator"]
        elif "gen" in state and isinstance(state["gen"], dict):
            state = state["gen"]

    if isinstance(state, dict) and any(k.startswith("gen.") for k in state):
        state = {k.removeprefix("gen."): v for k, v in state.items()}

    if not isinstance(state, dict):
        raise ValueError("Unsupported checkpoint format. Expected a state_dict-like object.")

    generator.load_state_dict(state, strict=False)
    generator = generator.to(device).eval()
    return SAR2OpticalPipeline(generator=generator)
