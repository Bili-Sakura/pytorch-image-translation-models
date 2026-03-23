# Credits: pix2pixHD (Wang et al., CVPR 2018) - High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs.
#
"""pix2pixHD single-pass inference pipeline and checkpoint loader."""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from diffusers import DiffusionPipeline
from diffusers.utils import BaseOutput, pt_to_pil

from src.models.pix2pixhd import Pix2PixHDGenerator


@dataclass
class Pix2PixHDPipelineOutput(BaseOutput):
    """Output of pix2pixHD pipeline."""

    images: Any


class Pix2PixHDPipeline(DiffusionPipeline):
    """Single-pass pix2pixHD inference pipeline."""

    def __init__(self, generator: nn.Module) -> None:
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
    ) -> "Pix2PixHDPipeline":
        """Load pix2pixHD pipeline from local config + safetensors."""
        model_dir = Path(pretrained_model_name_or_path)
        if subfolder:
            model_dir = model_dir / subfolder

        config_path = model_dir / "config.json"
        weights_path = model_dir / "diffusion_pytorch_model.safetensors"
        if not (config_path.exists() and weights_path.exists()):
            return super().from_pretrained(pretrained_model_name_or_path, subfolder=subfolder, **kwargs)

        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)

        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        generator = Pix2PixHDGenerator(
            input_nc=config.get("input_nc", 3),
            output_nc=config.get("output_nc", 3),
            ngf=config.get("ngf", 64),
            n_downsampling=config.get("n_downsampling", 4),
            n_blocks=config.get("n_blocks", 9),
            norm_layer=norm_layer,
        )
        from safetensors.torch import load_file

        state_dict = load_file(str(weights_path), device="cpu")
        try:
            generator.load_state_dict(state_dict, strict=strict)
        except RuntimeError:
            # Fallback for keys saved without top-level "model." prefix.
            generator.load_state_dict(_normalize_pix2pixhd_keys(state_dict), strict=strict)
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
        output_type: str = "pil",
        return_dict: bool = True,
    ) -> Union[Pix2PixHDPipelineOutput, tuple]:
        device = self.device
        dtype = self.dtype
        x = self.prepare_inputs(source_image, device, dtype)
        images = self.generator(x).clamp(-1, 1)

        if output_type == "pil":
            images = pt_to_pil(images)
        elif output_type == "np":
            images = self._convert_to_numpy(images)

        if not return_dict:
            return (images,)
        return Pix2PixHDPipelineOutput(images=images)

    @staticmethod
    def _convert_to_numpy(images: torch.Tensor) -> np.ndarray:
        images = (images + 1) / 2
        return images.clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()


def _strip_prefixes(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Normalize checkpoint keys from wrapped training runs."""
    prefixes = ("module.", "model.")
    for prefix in prefixes:
        if all(key.startswith(prefix) for key in state_dict):
            return {key[len(prefix):]: value for key, value in state_dict.items()}
    return state_dict


def _normalize_pix2pixhd_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map common pix2pixHD key patterns to this project's generator module."""
    state_dict = _strip_prefixes(state_dict)
    if state_dict and all(not key.startswith("model.") for key in state_dict):
        # NVIDIA pix2pixHD often stores keys from a plain nn.Sequential like
        # "1.weight", while this project wraps it under "model".
        state_dict = {f"model.{key}": value for key, value in state_dict.items()}
    return state_dict


def _resolve_checkpoint_file(path: str | Path, epoch: str) -> Path:
    checkpoint_path = Path(path)
    if checkpoint_path.is_file():
        return checkpoint_path

    candidates = [checkpoint_path / f"{epoch}_net_G.pth", checkpoint_path / "latest_net_G.pth", checkpoint_path / "net_G.pth"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find pix2pixHD generator checkpoint under: {checkpoint_path}")


def load_pix2pixhd_pipeline(
    checkpoint_path: str | Path,
    *,
    epoch: str = "latest",
    input_nc: int = 3,
    output_nc: int = 3,
    ngf: int = 64,
    n_downsampling: int = 4,
    n_blocks: int = 9,
    device: str = "cuda",
) -> Pix2PixHDPipeline:
    """Load pix2pixHD ``netG`` checkpoint into a native pipeline.

    If ``generator/config.json`` exists beside the checkpoint directory, these
    architecture values are read from it and override function defaults.
    """

    ckpt_path = _resolve_checkpoint_file(checkpoint_path, epoch=epoch)

    config_path = ckpt_path.parent / "generator" / "config.json"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        input_nc = config.get("input_nc", input_nc)
        output_nc = config.get("output_nc", output_nc)
        ngf = config.get("ngf", ngf)
        n_downsampling = config.get("n_downsampling", n_downsampling)
        n_blocks = config.get("n_blocks", n_blocks)

    norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    generator = Pix2PixHDGenerator(
        input_nc=input_nc,
        output_nc=output_nc,
        ngf=ngf,
        n_downsampling=n_downsampling,
        n_blocks=n_blocks,
        norm_layer=norm_layer,
    )

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict):
        if "netG" in checkpoint:
            state_dict = checkpoint["netG"]
        elif "generator" in checkpoint:
            state_dict = checkpoint["generator"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    state_dict = _normalize_pix2pixhd_keys(state_dict)
    generator.load_state_dict(state_dict, strict=True)
    generator = generator.to(device).eval()

    return Pix2PixHDPipeline(generator=generator)

