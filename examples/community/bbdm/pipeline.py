# Copyright (c) 2026 EarthBridge Team.
# Credits: xuekt98/BBDM (Li et al., CVPR 2023).

"""BBDM community pipeline loader for OpenAI-style checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from examples.community.bbdm.model import OpenAIBBDMUNet
from src.pipelines.bbdm import BBDMPipeline
from src.schedulers.bbdm import BBDMScheduler


def load_bbdm_community_pipeline(
    pretrained_model_name_or_path: str | Path,
    *,
    subfolder: str = "unet",
    device: str | Union[str, "torch.device"] = "cpu",
    torch_dtype: "torch.dtype | None" = None,
) -> BBDMPipeline:
    """Load BBDM pipeline for OpenAI-style BBDM checkpoints."""
    import torch

    path = Path(pretrained_model_name_or_path)
    unet = OpenAIBBDMUNet.from_pretrained(path, subfolder=subfolder, device=device)
    if torch_dtype is not None:
        unet = unet.to(dtype=torch_dtype)

    scheduler_cfg = None
    scheduler_path = path / "scheduler" / "scheduler_config.json"
    if scheduler_path.exists():
        with open(scheduler_path, encoding="utf-8") as f:
            scheduler_cfg = json.load(f)
    scheduler = (
        BBDMScheduler(**scheduler_cfg)
        if scheduler_cfg is not None
        else BBDMScheduler()
    )
    return BBDMPipeline(unet=unet, scheduler=scheduler)
