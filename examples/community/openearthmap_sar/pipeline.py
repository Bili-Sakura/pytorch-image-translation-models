# Copyright (c) 2026 EarthBridge Team.
# Credits: OpenEarthMap-SAR CUT models. Architecture from Park et al. "Contrastive Learning for Unpaired Image-to-Image Translation" ECCV 2020.

"""Loader for OpenEarthMap-SAR CUT checkpoints.

Provides :func:`load_openearthmap_sar_pipeline` to load pre-trained checkpoints
and return a :class:`CUTPipeline` ready for inference.
"""

from __future__ import annotations

import functools
import json
import logging
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn

from src.pipelines.cut import CUTPipeline

from .model import OpenEarthMapSARGenerator

logger = logging.getLogger(__name__)

OPENEARTHMAP_SAR_MODELS: dict[str, str] = {
    "opt2sar": "20",
    "sar2opt": "15",
    "seman2opt": "25",
    "seman2opt_pesudo": "195",
    "seman2sar": "25",
    "seman2sar_pesudo": "200",
}


def load_openearthmap_sar_pipeline(
    checkpoint_dir: str | Path,
    model_name: Literal["opt2sar", "sar2opt", "seman2opt", "seman2opt_pesudo", "seman2sar", "seman2sar_pesudo"] = "sar2opt",
    epoch: str = "latest",
    in_channels: int = 3,
    out_channels: int = 3,
    base_filters: int = 64,
    n_blocks: int = 9,
    device: str = "cuda",
) -> CUTPipeline:
    """Load an OpenEarthMap-SAR CUT checkpoint for inference.

    Prefers safetensors + config.json when available; otherwise falls back to .pth.

    Parameters
    ----------
    checkpoint_dir : str or Path
        Root directory containing model subdirs (e.g. CUT-OpenEarthMap-SAR).
    model_name : str
        One of: opt2sar, sar2opt, seman2opt, seman2opt_pesudo, seman2sar, seman2sar_pesudo.
    epoch : str
        ``"latest"`` or a specific epoch number (for .pth fallback).
    in_channels, out_channels, base_filters, n_blocks : int
        Generator architecture (used when config.json is missing).
    device : str
        Target device.

    Returns
    -------
    CUTPipeline
        Pipeline ready for inference.
    """
    checkpoint_dir = Path(checkpoint_dir)
    model_dir = checkpoint_dir / model_name
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    gen_path = model_dir / "generator"
    safetensors_path = gen_path / "diffusion_pytorch_model.safetensors"
    config_path = gen_path / "config.json"

    if safetensors_path.exists() and config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        cfg_in = config.get("in_channels", in_channels)
        cfg_out = config.get("out_channels", out_channels)
        cfg_base = config.get("base_filters", base_filters)
        cfg_blocks = config.get("n_blocks", n_blocks)
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        generator = OpenEarthMapSARGenerator(
            in_channels=cfg_in,
            out_channels=cfg_out,
            base_filters=cfg_base,
            n_blocks=cfg_blocks,
            norm_layer=norm_layer,
            use_dropout=config.get("use_dropout", False),
            padding_type=config.get("padding_type", "reflect"),
            no_antialias=config.get("no_antialias", False),
            no_antialias_up=config.get("no_antialias_up", False),
        )
        from safetensors.torch import load_file

        state_dict = load_file(str(safetensors_path), device=device)
        generator.load_state_dict(state_dict, strict=True)
        generator = generator.to(device).eval()
        logger.info("Loaded OpenEarthMap-SAR %s from %s", model_name, safetensors_path)
        return CUTPipeline(generator=generator)

    if epoch == "latest":
        ckpt_path = model_dir / "latest_net_G.pth"
        if not ckpt_path.exists():
            ckpt_path = model_dir / "generator.pt"
    else:
        ckpt_path = model_dir / f"{epoch}_net_G.pth"
        if not ckpt_path.exists():
            ckpt_path = model_dir / "generator.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found in {model_dir}")

    norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    generator = OpenEarthMapSARGenerator(
        in_channels=in_channels,
        out_channels=out_channels,
        base_filters=base_filters,
        n_blocks=n_blocks,
        norm_layer=norm_layer,
        use_dropout=False,
        no_antialias=False,
        no_antialias_up=False,
    )

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    if isinstance(ckpt, dict) and "generator" in ckpt:
        state_dict = ckpt["generator"]
    else:
        state_dict = ckpt

    generator.load_state_dict(state_dict, strict=True)
    generator = generator.to(device).eval()
    logger.info("Loaded OpenEarthMap-SAR %s from %s", model_name, ckpt_path)

    return CUTPipeline(generator=generator)
