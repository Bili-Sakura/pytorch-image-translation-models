# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""CUT single-pass inference pipeline.

Provides :class:`CUTPipeline` for running inference with a trained CUT
generator, following the diffusers-style pipeline pattern established
in :mod:`src.pipelines.ddbm`.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from PIL import Image

from diffusers import DiffusionPipeline
from diffusers.utils import BaseOutput, pt_to_pil

from src.models.cut import CUTGenerator


@dataclass
class CUTPipelineOutput(BaseOutput):
    """Output class for CUT pipeline.

    Attributes
    ----------
    images : list of PIL.Image.Image, np.ndarray, or torch.Tensor
        Generated images after translation.
    """

    images: Union[List[Image.Image], np.ndarray, torch.Tensor]


class CUTPipeline(DiffusionPipeline):
    """Single-pass inference pipeline for Contrastive Unpaired Translation.

    Unlike diffusion-based pipelines, CUT performs image translation in a
    single forward pass through the generator network.  This pipeline wraps
    the generator with a consistent API for loading, preprocessing, and
    postprocessing.

    Inherits from :class:`~diffusers.DiffusionPipeline` so that checkpoints
    can be loaded via ``from_pretrained`` following the HuggingFace
    *diffusers* convention.

    Example
    -------
    ::

        from src.pipelines.cut import CUTPipeline

        pipeline = CUTPipeline.from_pretrained("./ckpt/cut/checkpoint-epoch-400")
        output = pipeline(source_image=my_tensor)
    """

    def __init__(self, generator: CUTGenerator) -> None:
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
    ) -> "CUTPipeline":
        """Load CUT pipeline from local generator checkpoint."""
        generator = CUTGenerator.from_pretrained(
            pretrained_model_name_or_path,
            subfolder=subfolder,
            **kwargs,
        )
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
    ) -> Union[CUTPipelineOutput, tuple]:
        """Generate translated images via a single forward pass.

        Parameters
        ----------
        source_image : torch.Tensor or PIL.Image.Image or list of PIL.Image.Image
            Source images for translation.  Tensors should be ``(B, C, H, W)``
            in ``[0, 1]`` or ``[-1, 1]`` range.
        output_type : str
            ``"pil"``, ``"np"``, or ``"pt"`` (default ``"pil"``).
        return_dict : bool
            If ``True``, return a :class:`CUTPipelineOutput`.

        Returns
        -------
        CUTPipelineOutput or tuple
            Translated images.
        """
        device = self.device
        dtype = self.dtype

        x = self.prepare_inputs(source_image, device, dtype)

        # Single forward pass – CUT is a feed-forward GAN
        fake = self.generator(x)
        images = fake.clamp(-1, 1)

        if output_type == "pil":
            images = pt_to_pil(images)
        elif output_type == "np":
            images = self._convert_to_numpy(images)
        # else: output_type == "pt", return tensor as-is

        if not return_dict:
            return (images,)

        return CUTPipelineOutput(images=images)

    @staticmethod
    def _convert_to_numpy(images: torch.Tensor) -> np.ndarray:
        """Convert tensor in [-1, 1] to numpy array in [0, 1]."""
        images = (images + 1) / 2
        images = images.clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        return images
