# Copyright (c) 2026 EarthBridge Team.
# Credits: DDIB (Su et al., ICLR 2023) https://github.com/suxuann/ddib

"""DDIB community pipeline for OpenAI/guided_diffusion-style checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from examples.community.ddib.model import OpenAIDDIBUNet
from src.pipelines.ddib import DDIBPipeline, DDIBPipelineOutput
from src.schedulers.ddib import DDIBScheduler


def load_ddib_community_pipeline(
    pretrained_model_name_or_path: str | Path,
    *,
    source_subfolder: str = "source_unet",
    target_subfolder: str = "target_unet",
    device: str | Union[str, "torch.device"] = None,
    torch_dtype: "torch.dtype | None" = None,
) -> DDIBPipeline:
    """Load DDIB pipeline for OpenAI-style checkpoints (source_unet/ + target_unet/).

    Use this for DDIB checkpoints that use the guided_diffusion architecture
    (input_blocks / middle_block / output_blocks), converted via
    convert_pt_to_ddib.

    Parameters
    ----------
    pretrained_model_name_or_path : str | Path
        Path to the model directory containing source_unet/ and target_unet/.
    source_subfolder : str
        Subfolder for the source domain UNet (default ``"source_unet"``).
    target_subfolder : str
        Subfolder for the target domain UNet (default ``"target_unet"``).
    device : str | torch.device
        Device to load models on.
    torch_dtype : torch.dtype | None
        Optional dtype for the models.

    Returns
    -------
    DDIBPipeline
        Pipeline ready for inference.

    Example
    -------
    >>> from examples.community.ddib import load_ddib_community_pipeline
    >>> pipe = load_ddib_community_pipeline("/path/to/DDIB-ckpt/Synthetic-log2D0-to-log2D1")
    >>> out = pipe(source_image=image, num_inference_steps=250, output_type="pil")
    """
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    path = Path(pretrained_model_name_or_path)

    source_unet = OpenAIDDIBUNet.from_pretrained(
        path, subfolder=source_subfolder, device=device
    )
    target_unet = OpenAIDDIBUNet.from_pretrained(
        path, subfolder=target_subfolder, device=device
    )

    if torch_dtype is not None:
        source_unet = source_unet.to(dtype=torch_dtype)
        target_unet = target_unet.to(dtype=torch_dtype)

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
        DDIBScheduler(**scheduler_cfg)
        if scheduler_cfg is not None
        else DDIBScheduler()
    )

    return DDIBPipeline(
        source_unet=source_unet,
        target_unet=target_unet,
        scheduler=scheduler,
    )
