#!/usr/bin/env python
# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Unified inference script for all supported bridge-diffusion methods.

This script imports all components from ``src/`` — no duplicated code.
Supports: DDBM, DDIB, I2SB, BiBBDM, BDBM, DBIM, CDTSDE, LBM.

Usage
-----
.. code-block:: bash

    python examples/inference/run_inference.py \\
        --model ddbm \\
        --checkpoint_dir /path/to/checkpoint \\
        --input_dir /path/to/input_images \\
        --output_dir /path/to/output_images \\
        --num_steps 40

Or from Python:

.. code-block:: python

    from examples.inference.run_inference import load_pipeline
    pipeline = load_pipeline("ddbm", "/path/to/checkpoint")
    result = pipeline(source_image, num_inference_steps=40, output_type="pil")
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.models.unet.diffusers_wrappers import (
    BDBMUNet,
    BiBBDMUNet,
    CDTSDEUNet,
    DBIMUNet,
    DDBMUNet,
    DDIBUNet,
    I2SBDiffusersUNet,
    LBMUNet,
)
from src.schedulers import (
    BDBMScheduler,
    BiBBDMScheduler,
    CDTSDEScheduler,
    DBIMScheduler,
    DDBMScheduler,
    DDIBScheduler,
    I2SBScheduler,
    LBMScheduler,
)
from src.pipelines import (
    BDBMPipeline,
    BiBBDMPipeline,
    CDTSDEPipeline,
    DBIMPipeline,
    DDBMPipeline,
    DDIBPipeline,
    I2SBPipeline,
    LBMPipeline,
)

SUPPORTED_MODELS = ("ddbm", "ddib", "i2sb", "bibbdm", "bdbm", "dbim", "cdtsde", "lbm")


def _load_unet(model_name: str, checkpoint_dir: str, **kwargs):
    """Load a diffusers UNet wrapper from a checkpoint directory."""
    unet_classes = {
        "ddbm": DDBMUNet,
        "ddib": DDIBUNet,
        "i2sb": I2SBDiffusersUNet,
        "bibbdm": BiBBDMUNet,
        "bdbm": BDBMUNet,
        "dbim": DBIMUNet,
        "cdtsde": CDTSDEUNet,
        "lbm": LBMUNet,
    }
    cls = unet_classes[model_name]
    subfolder = kwargs.pop("subfolder", "ema_unet")
    return cls.from_pretrained(checkpoint_dir, subfolder=subfolder, **kwargs)


def _load_scheduler(model_name: str, checkpoint_dir: str, **kwargs):
    """Load the appropriate scheduler for a model."""
    scheduler_classes = {
        "ddbm": DDBMScheduler,
        "ddib": DDIBScheduler,
        "i2sb": I2SBScheduler,
        "bibbdm": BiBBDMScheduler,
        "bdbm": BDBMScheduler,
        "dbim": DBIMScheduler,
        "cdtsde": CDTSDEScheduler,
        "lbm": LBMScheduler,
    }
    return scheduler_classes[model_name](**kwargs)


def load_pipeline(model_name: str, checkpoint_dir: str, device: str = "cpu"):
    """Load a complete inference pipeline.

    Parameters
    ----------
    model_name : str
        One of: ddbm, ddib, i2sb, bibbdm, bdbm, dbim, cdtsde, lbm.
    checkpoint_dir : str
        Path to the pretrained checkpoint directory.
    device : str
        Device to run on.

    Returns
    -------
    pipeline
        An inference pipeline that can be called with ``pipeline(source, ...)``.
    """
    model_name = model_name.lower()
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unknown model: {model_name}. Supported: {SUPPORTED_MODELS}")

    pipeline_classes = {
        "ddbm": DDBMPipeline,
        "ddib": DDIBPipeline,
        "i2sb": I2SBPipeline,
        "bibbdm": BiBBDMPipeline,
        "bdbm": BDBMPipeline,
        "dbim": DBIMPipeline,
        "cdtsde": CDTSDEPipeline,
        "lbm": LBMPipeline,
    }

    unet = _load_unet(model_name, checkpoint_dir)
    scheduler = _load_scheduler(model_name, checkpoint_dir)
    pipeline_cls = pipeline_classes[model_name]

    if model_name == "ddib":
        source_unet = _load_unet(model_name, checkpoint_dir, subfolder="source_unet")
        target_unet = _load_unet(model_name, checkpoint_dir, subfolder="target_unet")
        pipeline = pipeline_cls(source_unet=source_unet, target_unet=target_unet, scheduler=scheduler)
    else:
        pipeline = pipeline_cls(unet=unet, scheduler=scheduler)

    return pipeline


def load_image(path: str) -> torch.Tensor:
    """Load an image and convert to a [-1, 1] tensor."""
    img = Image.open(path).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor * 2 - 1


def save_image(tensor: torch.Tensor, path: str) -> None:
    """Save a [-1, 1] tensor as an image."""
    arr = ((tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def main():
    parser = argparse.ArgumentParser(description="Unified inference for bridge-diffusion image translation")
    parser.add_argument("--model", type=str, required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_steps", type=int, default=40)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_type", type=str, default="pil", choices=["pil", "np", "pt"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    pipeline = load_pipeline(args.model, args.checkpoint_dir, args.device)

    input_dir = Path(args.input_dir)
    for img_path in sorted(input_dir.glob("*.png")) + sorted(input_dir.glob("*.jpg")):
        source = load_image(str(img_path)).to(args.device)
        result = pipeline(source, num_inference_steps=args.num_steps, output_type=args.output_type)

        if args.output_type == "pil":
            for i, img in enumerate(result.images):
                out_path = Path(args.output_dir) / f"{img_path.stem}_translated_{i}.png"
                img.save(str(out_path))
        elif args.output_type == "pt":
            for i in range(result.images.shape[0]):
                out_path = Path(args.output_dir) / f"{img_path.stem}_translated_{i}.png"
                save_image(result.images[i:i + 1], str(out_path))

        print(f"Processed {img_path.name} → {args.output_dir}")


if __name__ == "__main__":
    main()
