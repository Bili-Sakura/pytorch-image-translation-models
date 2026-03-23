# Copyright (c) 2026 EarthBridge Team.
# Credits: Adapted from StegoGAN (Wu et al., CVPR 2024).
# Original: https://github.com/sian-wusidi/StegoGAN

"""StegoGAN inference pipeline for non-bijective image-to-image translation."""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from diffusers.utils import pt_to_pil

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

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        subfolder_a: str = "generator_A",
        subfolder_b: str = "generator_B",
        device: str | torch.device = "cpu",
        torch_dtype: torch.dtype | None = None,
    ) -> "StegoGANPipeline":
        """Load StegoGAN generators from local config + safetensors."""
        root = Path(pretrained_model_name_or_path)
        dir_a = root / subfolder_a
        dir_b = root / subfolder_b

        cfg_a_path = dir_a / "config.json"
        cfg_b_path = dir_b / "config.json"
        w_a_path = dir_a / "diffusion_pytorch_model.safetensors"
        w_b_path = dir_b / "diffusion_pytorch_model.safetensors"
        if not (cfg_a_path.exists() and cfg_b_path.exists() and w_a_path.exists() and w_b_path.exists()):
            raise FileNotFoundError(
                "Expected StegoGAN layout with generator_A/generator_B config + safetensors."
            )

        with open(cfg_a_path, encoding="utf-8") as f:
            cfg_a = json.load(f)
        with open(cfg_b_path, encoding="utf-8") as f:
            cfg_b = json.load(f)

        norm_name = cfg_a.get("norm", "instance")
        if norm_name == "instance":
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        elif norm_name == "batch":
            norm_layer = nn.BatchNorm2d
        else:
            raise ValueError(f"Unsupported norm in StegoGAN config: {norm_name}")

        netG_A = ResnetMaskV1Generator(
            input_nc=cfg_a.get("input_nc", 3),
            output_nc=cfg_a.get("output_nc", 3),
            ngf=cfg_a.get("ngf", 64),
            norm_layer=norm_layer,
            use_dropout=cfg_a.get("use_dropout", False),
            n_blocks=cfg_a.get("n_blocks", 9),
            resnet_layer=cfg_a.get("resnet_layer", 1),
            fusionblock=cfg_a.get("fusionblock", False),
        )
        netG_B = ResnetMaskV3Generator(
            input_nc=cfg_b.get("input_nc", 3),
            output_nc=cfg_b.get("output_nc", 3),
            ngf=cfg_b.get("ngf", 64),
            norm_layer=norm_layer,
            use_dropout=cfg_b.get("use_dropout", False),
            n_blocks=cfg_b.get("n_blocks", 9),
            input_dim=cfg_b.get("input_dim", 256),
            out_dim=cfg_b.get("out_dim", 256),
            resnet_layer=cfg_b.get("resnet_layer", 1),
        )

        from safetensors.torch import load_file

        netG_A.load_state_dict(load_file(str(w_a_path), device="cpu"), strict=True)
        netG_B.load_state_dict(load_file(str(w_b_path), device="cpu"), strict=True)

        netG_A = netG_A.eval().to(device=device)
        netG_B = netG_B.eval().to(device=device)
        if torch_dtype is not None:
            netG_A = netG_A.to(dtype=torch_dtype)
            netG_B = netG_B.to(dtype=torch_dtype)
        return cls(netG_A=netG_A, netG_B=netG_B)

    def to(self, device: str | torch.device) -> "StegoGANPipeline":
        """Move both generators to a target device."""
        self.netG_A = self.netG_A.to(device)
        self.netG_B = self.netG_B.to(device)
        return self

    @property
    def device(self) -> torch.device:
        """Get the current pipeline device."""
        return next(self.netG_A.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Get the current pipeline dtype."""
        return next(self.netG_A.parameters()).dtype

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
        source = source.to(device=self.device, dtype=self.dtype)

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
            np_masks = masks.cpu().numpy() if masks is not None else None
            return StegoGANPipelineOutput(images=pt_to_pil(tensor), masks=np_masks)

        raise ValueError(f"Unknown output_type: {output_type!r}")
