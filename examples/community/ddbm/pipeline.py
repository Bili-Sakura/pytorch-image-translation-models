# Copyright (c) 2026 EarthBridge Team.
# Credits: alexzhou907/DDBM (Zhou et al., ICLR 2024).

"""DDBM community pipeline for OpenAI-style checkpoints (BiliSakura/DDBM-ckpt).

Uses :class:`OpenAIDDBMUNet` and reuses :class:`DDBMScheduler` + sampling logic
from the core DDBM pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from examples.community.ddbm.model import OpenAIDDBMUNet
from src.pipelines.ddbm import DDBMPipeline, DDBMPipelineOutput
from src.schedulers.ddbm import DDBMScheduler


def load_ddbm_community_pipeline(
    pretrained_model_name_or_path: str | Path,
    *,
    subfolder: str = "unet",
    device: str | Union[str, "torch.device"] = None,
    torch_dtype: "torch.dtype | None" = None,
) -> DDBMPipeline:
    """Load DDBM pipeline for OpenAI-style checkpoints (unet/ format).

    Use this for BiliSakura/DDBM-ckpt and other checkpoints that use
    the improved_diffusion architecture (input_blocks / middle_block / output_blocks).

    Parameters
    ----------
    pretrained_model_name_or_path : str | Path
        Path to the model directory containing unet/config.json and
        unet/diffusion_pytorch_model.safetensors.
    subfolder : str
        Subfolder containing the UNet (default ``"unet"``).
    device : str | torch.device
        Device to load the model on.
    torch_dtype : torch.dtype | None
        Optional dtype for the model.

    Returns
    -------
    DDBMPipeline
        Pipeline ready for inference.

    Example
    -------
    >>> from examples.community.ddbm import load_ddbm_community_pipeline
    >>> pipe = load_ddbm_community_pipeline("/path/to/DDBM-ckpt/edges2handbags-vp")
    >>> out = pipe(source_image=image, num_inference_steps=40, output_type="pil")
    """
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    path = Path(pretrained_model_name_or_path)
    unet = OpenAIDDBMUNet.from_pretrained(
        path, subfolder=subfolder, device=device
    )
    if torch_dtype is not None:
        unet = unet.to(dtype=torch_dtype)

    scheduler_cfg = None
    for candidate in (
        path / "scheduler" / "scheduler_config.json",
        path / "scheduler_config.json",
    ):
        if candidate.exists():
            with open(candidate, encoding="utf-8") as f:
                scheduler_cfg = json.load(f)
            break
    scheduler = (
        DDBMScheduler(**scheduler_cfg)
        if scheduler_cfg is not None
        else DDBMScheduler()
    )

    return DDBMPipeline(unet=unet, scheduler=scheduler)
