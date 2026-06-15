# Copyright (c) 2026 EarthBridge Team.
# Credits: img2img-turbo from Parmar et al. 2024 —
# https://github.com/GaParmar/img2img-turbo

"""CycleGAN-Turbo community pipeline loader."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch

from src.pipelines.cyclegan_turbo import CycleGANTurboPipeline, CycleGANTurboPipelineOutput


def load_cyclegan_turbo_pipeline(
    *,
    pretrained_name: Optional[str] = None,
    pretrained_path: Optional[Union[str, Path]] = None,
    ckpt_folder: str = "checkpoints",
    device: str | torch.device = "cpu",
    image_prep: str = "resize_512x512",
    torch_dtype: torch.dtype | None = None,
) -> CycleGANTurboPipeline:
    """Load a CycleGAN-Turbo inference pipeline.

    Parameters
    ----------
    pretrained_name : str, optional
        One of ``day_to_night``, ``night_to_day``, ``clear_to_rainy``, ``rainy_to_clear``.
    pretrained_path : str or Path, optional
        Local ``.pkl`` checkpoint from training.
    ckpt_folder : str
        Directory for downloading pretrained weights.
    device : str or torch.device
        Target device.
    image_prep : str
        Preprocessing preset (``resize_512x512``, etc.).
    """
    if (pretrained_name is None) == (pretrained_path is None):
        raise ValueError("Provide exactly one of pretrained_name or pretrained_path")
    pipe = CycleGANTurboPipeline.from_pretrained(
        pretrained_name=pretrained_name,
        pretrained_path=pretrained_path,
        ckpt_folder=ckpt_folder,
        device=device,
        image_prep=image_prep,
    )
    if torch_dtype is not None:
        pipe.model.to(dtype=torch_dtype)
    return pipe


__all__ = ["CycleGANTurboPipeline", "CycleGANTurboPipelineOutput", "load_cyclegan_turbo_pipeline"]
