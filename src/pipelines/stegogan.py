# Copyright (c) 2026 EarthBridge Team.
# Credits: Adapted from StegoGAN (Wu et al., CVPR 2024).
# Original: https://github.com/sian-wusidi/StegoGAN

"""StegoGAN inference pipeline for non-bijective image-to-image translation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from PIL import Image

from src.models.stegogan.generators import (
    ResnetMaskV1Generator,
    ResnetMaskV3Generator,
)


@dataclass
class StegoGANPipelineOutput:
    """Container for StegoGAN pipeline results.

    Attributes
    ----------
    images : Tensor | list[PIL.Image.Image] | np.ndarray
        Translated images in the requested format.
    masks : Tensor | None
        Per-pixel matchability masks (only for B→A direction).
    """

    images: Any
    masks: Any = None


class StegoGANPipeline:
    """End-to-end inference pipeline for a trained StegoGAN model.

    Wraps the two CycleGAN-style generators (``G_A``: A→B,
    ``G_B``: B→A with masking) and exposes a single ``__call__``
    interface for bidirectional translation.

    Parameters
    ----------
    netG_A : ResnetMaskV1Generator
        Generator for A → B translation (with optional stego injection).
    netG_B : ResnetMaskV3Generator
        Generator for B → A translation (produces matchability mask).
    """

    def __init__(
        self,
        netG_A: ResnetMaskV1Generator,
        netG_B: ResnetMaskV3Generator,
    ) -> None:
        self.netG_A = netG_A
        self.netG_B = netG_B

    @torch.no_grad()
    def __call__(
        self,
        source: torch.Tensor,
        direction: str = "a2b",
        output_type: str = "pt",
    ) -> StegoGANPipelineOutput:
        """Run StegoGAN inference.

        Parameters
        ----------
        source : Tensor ``[B, C, H, W]``
            Source images.
        direction : ``"a2b"`` | ``"b2a"``
            Translation direction.  ``"a2b"`` uses ``G_A`` (domain A → B),
            ``"b2a"`` uses ``G_B`` (domain B → A, with masking).
        output_type : ``"pt"`` | ``"pil"`` | ``"np"``
            Desired output format.

        Returns
        -------
        StegoGANPipelineOutput
            Translated images and, for ``"b2a"`` direction, the
            matchability masks.
        """
        if direction == "a2b":
            translated = self.netG_A(source)
            return self._format_output(translated, masks=None, output_type=output_type)

        if direction == "b2a":
            translated, _feat_disc, mask_sum = self.netG_B(source)
            return self._format_output(translated, masks=mask_sum, output_type=output_type)

        raise ValueError(
            f"Unknown direction: {direction!r}. Expected 'a2b' or 'b2a'."
        )

    # ------------------------------------------------------------------
    # Output formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_output(
        tensor: torch.Tensor,
        masks: torch.Tensor | None,
        output_type: str,
    ) -> StegoGANPipelineOutput:
        if output_type == "pt":
            return StegoGANPipelineOutput(images=tensor, masks=masks)

        if output_type == "np":
            np_masks = masks.cpu().numpy() if masks is not None else None
            return StegoGANPipelineOutput(
                images=tensor.cpu().numpy(), masks=np_masks,
            )

        if output_type == "pil":
            images: list[Image.Image] = []
            # Tanh output is in [-1, 1] — map to [0, 255]
            arr = ((tensor + 1) / 2).clamp(0, 1).cpu()
            for i in range(arr.shape[0]):
                img_np = arr[i].permute(1, 2, 0).numpy()
                img_np = (img_np * 255).astype(np.uint8)
                if img_np.shape[2] == 1:
                    img_np = img_np[:, :, 0]
                images.append(Image.fromarray(img_np))
            np_masks = masks.cpu().numpy() if masks is not None else None
            return StegoGANPipelineOutput(images=images, masks=np_masks)

        raise ValueError(f"Unknown output_type: {output_type!r}")
