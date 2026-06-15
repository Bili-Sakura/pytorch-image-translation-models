# Credits: pix2pix (Isola et al., CVPR 2017) — https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

"""Pix2Pix single-pass inference pipelines."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from diffusers import DiffusionPipeline
from diffusers.utils import BaseOutput, pt_to_pil

from src.models.cyclegan_pix2pix import Pix2PixGenerator, create_generator

logger = logging.getLogger(__name__)


@dataclass
class Pix2PixPipelineOutput(BaseOutput):
    """Output class for pix2pix pipeline."""

    images: Union[List[Image.Image], np.ndarray, torch.Tensor]


class Pix2PixPipeline(DiffusionPipeline):
    """Single-pass inference pipeline for pix2pix paired image translation."""

    def __init__(self, generator: Pix2PixGenerator | nn.Module) -> None:
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
    ) -> "Pix2PixPipeline":
        """Load pix2pix pipeline from HF-style generator checkpoint."""
        generator = Pix2PixGenerator.from_pretrained(
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
        return next(self.generator.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.generator.parameters()).dtype

    def prepare_inputs(
        self,
        image: Union[torch.Tensor, Image.Image, List[Image.Image]],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
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

        if image.min() >= 0 and image.max() <= 1.0:
            image = image * 2 - 1
        elif image.max() > 1.0:
            image = image / 255.0 * 2 - 1

        return image.to(device=device, dtype=dtype)

    @torch.no_grad()
    def __call__(
        self,
        source_image: Union[torch.Tensor, Image.Image, List[Image.Image]],
        output_type: str = "pil",
        return_dict: bool = True,
    ) -> Union[Pix2PixPipelineOutput, tuple]:
        device = self.device
        dtype = self.dtype
        x = self.prepare_inputs(source_image, device, dtype)
        fake = self.generator(x).clamp(-1, 1)

        if output_type == "pil":
            images = pt_to_pil(fake)
        elif output_type == "np":
            images = self._convert_to_numpy(fake)
        else:
            images = fake

        if not return_dict:
            return (images,)
        return Pix2PixPipelineOutput(images=images)

    @staticmethod
    def _convert_to_numpy(images: torch.Tensor) -> np.ndarray:
        images = (images + 1) / 2
        images = images.clamp(0, 1)
        return images.cpu().permute(0, 2, 3, 1).numpy()


def load_pix2pix_pipeline(
    checkpoint_dir: Union[str, Path],
    *,
    netG: str = "unet_256",
    norm: str = "batch",
    input_nc: int = 3,
    output_nc: int = 3,
    ngf: int = 64,
    no_dropout: bool = False,
    device: str | torch.device = "cpu",
    torch_dtype: torch.dtype | None = None,
) -> Pix2PixPipeline:
    """Load pix2pix pipeline from HF checkpoint or upstream ``latest_net_G.pth``."""
    from src.models.cyclegan_pix2pix import load_upstream_generator_state

    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    gen_dir = checkpoint_dir / "generator"
    if gen_dir.exists() and (gen_dir / "config.json").exists():
        return Pix2PixPipeline.from_pretrained(
            checkpoint_dir,
            device=device,
            torch_dtype=torch_dtype,
        )

    generator = create_generator(
        input_nc=input_nc,
        output_nc=output_nc,
        ngf=ngf,
        netG=netG,
        norm=norm,
        use_dropout=not no_dropout,
    )

    ckpt_path = checkpoint_dir / "latest_net_G.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No pix2pix generator checkpoint found in {checkpoint_dir}. "
            "Expected generator/ or latest_net_G.pth"
        )
    load_upstream_generator_state(generator, ckpt_path)

    generator = generator.eval().to(device=device)
    if torch_dtype is not None:
        generator = generator.to(dtype=torch_dtype)
    return Pix2PixPipeline(generator=generator)


class ImageTranslator:
    """High-level inference wrapper for a trained generator.

    Parameters
    ----------
    generator:
        A trained generator module.
    device:
        Device to run inference on.
    image_size:
        Target image size for the input transform.
    normalize:
        Whether inputs are normalised to [-1, 1].
    """

    def __init__(
        self,
        generator: nn.Module,
        device: str | torch.device = "cpu",
        image_size: int = 256,
        normalize: bool = True,
    ) -> None:
        self.device = torch.device(device)
        self.generator = generator.to(self.device).eval()
        self.normalize = normalize

        transform_list: list[transforms.transforms.Transform] = [
            transforms.Resize(
                (image_size, image_size),
                transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
        ]
        if normalize:
            transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform_list)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        subfolder: str = "generator",
        device: str = "cpu",
        image_size: int = 256,
        **kwargs,
    ) -> "ImageTranslator":
        """Load a translator from an HF-style checkpoint (config.json + safetensors).

        Parameters
        ----------
        pretrained_model_name_or_path:
            Path to checkpoint dir (e.g. ``checkpoint-epoch-10`` or ``latest``).
        subfolder:
            Subfolder containing generator (default ``generator``).
        device:
            Target device.
        image_size:
            Input image size.
        **kwargs:
            Extra keyword arguments forwarded to the constructor.
        """
        from src.models.generators import UNetGenerator
        import json
        from safetensors.torch import load_file

        root = Path(pretrained_model_name_or_path)
        gen_dir = root / subfolder
        cfg_path = gen_dir / "config.json"
        weights_path = gen_dir / "diffusion_pytorch_model.safetensors"
        if not cfg_path.exists() or not weights_path.exists():
            raise FileNotFoundError(
                f"Expected {subfolder}/config.json and {subfolder}/diffusion_pytorch_model.safetensors in {root}"
            )
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        generator = UNetGenerator(**{k: v for k, v in cfg.items() if k in ("in_channels", "out_channels", "num_downs", "base_filters", "use_dropout")})
        generator.load_state_dict(load_file(str(weights_path), device="cpu"), strict=True)
        logger.info("Loaded generator from %s (HF format)", root)
        return cls(generator, device=device, image_size=image_size, **kwargs)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        generator: nn.Module | None = None,
        device: str = "cpu",
        **kwargs,
    ) -> "ImageTranslator":
        """Load a translator from a checkpoint. Supports HF format (dir) or legacy .pt.

        Parameters
        ----------
        checkpoint_path:
            Path to HF checkpoint dir or a ``.pt`` file.
        generator:
            An uninitialised generator (only needed for legacy .pt).
        device:
            Target device.
        **kwargs:
            Extra keyword arguments forwarded to the constructor.
        """
        path = Path(checkpoint_path)
        if path.is_dir() and (path / "generator" / "config.json").exists():
            return cls.from_pretrained(path, device=device, **kwargs)
        if generator is None:
            raise ValueError("generator must be provided when loading legacy .pt checkpoint")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        generator.load_state_dict(ckpt["generator"])
        logger.info("Loaded generator from %s", checkpoint_path)
        return cls(generator, device=device, **kwargs)

    @torch.no_grad()
    def predict(self, image: Image.Image) -> Image.Image:
        """Translate a single PIL image.

        Parameters
        ----------
        image:
            Input RGB image.

        Returns
        -------
        PIL.Image.Image:
            Translated RGB image.
        """
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        output = self.generator(tensor).squeeze(0).cpu()
        return self._tensor_to_pil(output)

    @torch.no_grad()
    def predict_batch(self, images: list[Image.Image]) -> list[Image.Image]:
        """Translate a batch of PIL images.

        Parameters
        ----------
        images:
            List of input RGB images.

        Returns
        -------
        list[PIL.Image.Image]:
            List of translated images.
        """
        tensors = torch.stack(
            [self.transform(img.convert("RGB")) for img in images]
        ).to(self.device)
        outputs = self.generator(tensors).cpu()
        return [self._tensor_to_pil(t) for t in outputs]

    def predict_file(
        self, input_path: str | Path, output_path: str | Path
    ) -> None:
        """Translate an image file and save the result.

        Parameters
        ----------
        input_path:
            Path to the input image.
        output_path:
            Where to save the translated image.
        """
        image = Image.open(input_path).convert("RGB")
        result = self.predict(image)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        result.save(output_path)
        logger.info("Saved translated image to %s", output_path)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        if self.normalize:
            tensor = tensor * 0.5 + 0.5  # [-1, 1] -> [0, 1]
        tensor = tensor.clamp(0, 1)
        return transforms.ToPILImage()(tensor)


__all__ = [
    "Pix2PixPipeline",
    "Pix2PixPipelineOutput",
    "load_pix2pix_pipeline",
    "ImageTranslator",
]
