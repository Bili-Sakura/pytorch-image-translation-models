"""Inference pipeline for image translation."""

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
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        generator: nn.Module,
        device: str = "cpu",
        **kwargs,
    ) -> "ImageTranslator":
        """Load a translator from a training checkpoint.

        Parameters
        ----------
        checkpoint_path:
            Path to a ``.pt`` checkpoint saved by
            :class:`~src.training.trainer.Pix2PixTrainer`.
        generator:
            An **uninitialised** generator of the same architecture.
        device:
            Target device.
        **kwargs:
            Extra keyword arguments forwarded to the constructor.
        """
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
