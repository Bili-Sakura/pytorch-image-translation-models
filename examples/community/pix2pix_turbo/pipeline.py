# Copyright (c) 2026 EarthBridge Team.
# Credits: img2img-turbo from Parmar et al. 2024 —
# https://github.com/GaParmar/img2img-turbo

"""pix2pix-turbo community pipeline loader."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch

from src.pipelines.pix2pix_turbo import Pix2PixTurboPipeline, Pix2PixTurboPipelineOutput


def load_pix2pix_turbo_pipeline(
    *,
    pretrained_name: Optional[str] = None,
    pretrained_path: Optional[Union[str, Path]] = None,
    ckpt_folder: str = "checkpoints",
    device: str | torch.device = "cpu",
    torch_dtype: torch.dtype | None = None,
) -> Pix2PixTurboPipeline:
    """Load a pix2pix-turbo inference pipeline.

    Parameters
    ----------
    pretrained_name : str, optional
        One of ``edge_to_image``, ``sketch_to_image_stochastic``.
    pretrained_path : str or Path, optional
        Local ``.pkl`` checkpoint from training.
    ckpt_folder : str
        Directory for downloading pretrained weights.
    device : str or torch.device
        Target device.
    """
    if (pretrained_name is None) == (pretrained_path is None):
        raise ValueError("Provide exactly one of pretrained_name or pretrained_path")
    pipe = Pix2PixTurboPipeline.from_pretrained(
        pretrained_name=pretrained_name,
        pretrained_path=pretrained_path,
        ckpt_folder=ckpt_folder,
        device=device,
    )
    if torch_dtype is not None:
        pipe.model.to(dtype=torch_dtype)
    return pipe


__all__ = ["Pix2PixTurboPipeline", "Pix2PixTurboPipelineOutput", "load_pix2pix_turbo_pipeline"]
