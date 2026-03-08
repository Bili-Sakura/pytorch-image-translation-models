# Copyright (c) 2026 EarthBridge Team.
# Credits: Adapted from Parallel-GAN (Wang et al., TGRS 2022).

"""Inference pipeline for Parallel-GAN.

Provides :class:`ParallelGANPipeline` for running inference with a trained
Parallel-GAN generator, following the diffusers-style pipeline pattern.
"""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from diffusers import DiffusionPipeline
from diffusers.utils import BaseOutput

from examples.community.parallel_gan.model import ParaGAN


@dataclass
class ParallelGANPipelineOutput(BaseOutput):
    """Output of the Parallel-GAN pipeline.

    Attributes
    ----------
    images : list[PIL.Image.Image] | np.ndarray | torch.Tensor
        Translated images in the requested format.
    """

    images: Union[List[Image.Image], np.ndarray, torch.Tensor]


class ParallelGANPipeline(DiffusionPipeline):
    """Single-pass inference pipeline for Parallel-GAN SAR-to-Optical translation.

    Parallel-GAN performs image translation in a single forward pass through
    the generator network (Stage 2 translation generator).  This pipeline
    wraps the generator with a consistent API for loading, preprocessing,
    and postprocessing.

    Inherits from :class:`~diffusers.DiffusionPipeline` so that checkpoints
    can be loaded via ``from_pretrained`` following the HuggingFace
    *diffusers* convention.

    Parameters
    ----------
    generator : ParaGAN
        Trained Parallel-GAN translation generator.

    Example
    -------
    ::

        from examples.community.parallel_gan import ParaGAN, ParallelGANPipeline

        gen = ParaGAN(input_nc=3, output_nc=3)
        pipeline = ParallelGANPipeline(generator=gen)
        output = pipeline(source_image=sar_tensor, output_type="pt")
    """

    def __init__(self, generator: ParaGAN) -> None:
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
    ) -> "ParallelGANPipeline":
        """Load Parallel-GAN from local config + safetensors."""
        model_dir = Path(pretrained_model_name_or_path)
        if subfolder:
            model_dir = model_dir / subfolder

        config_path = model_dir / "config.json"
        weights_path = model_dir / "diffusion_pytorch_model.safetensors"
        if not (config_path.exists() and weights_path.exists()):
            return super().from_pretrained(pretrained_model_name_or_path, subfolder=subfolder, **kwargs)

        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)

        norm_name = cfg.get("norm", "batch")
        if norm_name == "instance":
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        elif norm_name == "batch":
            norm_layer = nn.BatchNorm2d
        else:
            raise ValueError(f"Unsupported norm in Parallel-GAN config: {norm_name}")

        generator = ParaGAN(
            input_nc=cfg.get("input_nc", 3),
            output_nc=cfg.get("output_nc", 3),
            channel=cfg.get("channel", 2048),
            norm_layer=norm_layer,
            use_dropout=cfg.get("use_dropout", False),
            n_blocks=cfg.get("n_blocks", 6),
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
        """Get the device of the pipeline."""
        return next(self.generator.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Get the dtype of the pipeline."""
        return next(self.generator.parameters()).dtype

    def prepare_inputs(
        self,
        image: Union[torch.Tensor, Image.Image, List[Image.Image]],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Prepare input images for the pipeline.

        Converts PIL images or numpy arrays to normalised tensors in
        ``[-1, 1]`` range.
        """
        if isinstance(image, Image.Image):
            image = [image]

        if isinstance(image, list) and isinstance(image[0], Image.Image):
            images = []
            for img in image:
                img_array = np.array(img, dtype=np.float32)
                if img_array.max() > 1.0:
                    img_array = img_array / 255.0
                if img_array.ndim == 2:
                    img_array = img_array[:, :, np.newaxis]
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
                images.append(img_tensor)
            image = torch.stack(images)

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)

        # Ensure image is in [-1, 1] range
        if image.min() >= 0 and image.max() <= 1.0:
            image = image * 2 - 1  # [0, 1] → [-1, 1]
        elif image.max() > 1.0:
            image = image / 255.0 * 2 - 1  # [0, 255] → [-1, 1]

        return image.to(device=device, dtype=dtype)

    @torch.no_grad()
    def __call__(
        self,
        source_image: Union[torch.Tensor, Image.Image, List[Image.Image]],
        output_type: str = "pil",
        return_dict: bool = True,
    ) -> Union[ParallelGANPipelineOutput, tuple]:
        """Generate translated images via a single forward pass.

        Parameters
        ----------
        source_image : torch.Tensor or PIL.Image.Image or list of PIL.Image.Image
            Source (SAR) images for translation.  Tensors should be
            ``(B, C, H, W)`` in ``[0, 1]`` or ``[-1, 1]`` range.
        output_type : str
            ``"pil"``, ``"np"``, or ``"pt"`` (default ``"pil"``).
        return_dict : bool
            If ``True``, return a :class:`ParallelGANPipelineOutput`.

        Returns
        -------
        ParallelGANPipelineOutput or tuple
            Translated images.
        """
        device = self.device
        dtype = self.dtype

        x = self.prepare_inputs(source_image, device, dtype)

        # Single forward pass – Parallel-GAN is a feed-forward generator
        features = self.generator(x)
        images = features[-1].clamp(-1, 1)

        if output_type == "pil":
            images = self._convert_to_pil(images)
        elif output_type == "np":
            images = self._convert_to_numpy(images)
        # else: output_type == "pt", return tensor as-is

        if not return_dict:
            return (images,)

        return ParallelGANPipelineOutput(images=images)

    @staticmethod
    def _convert_to_pil(images: torch.Tensor) -> List[Image.Image]:
        """Convert tensor in [-1, 1] to PIL images."""
        images = (images + 1) / 2  # [-1, 1] → [0, 1]
        images = images.clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()
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
        """Convert tensor in [-1, 1] to numpy array in [0, 1]."""
        images = (images + 1) / 2
        images = images.clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        return images
