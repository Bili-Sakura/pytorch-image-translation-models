# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Pix2Pix single-pass inference pipeline for basic GAN generators."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)


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
