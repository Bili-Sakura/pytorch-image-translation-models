#!/usr/bin/env python
# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Standalone inference script using self-contained pipelines.

No external project code (src/) required. Uses only diffusers and examples.pipelines.

Usage:
    # DDBM
    python -m examples.pipelines.run_inference \
        --model ddbm \
        --checkpoint ./checkpoints/ddbm/sar2eo/checkpoint-10000 \
        --input_dir ./path/to/images \
        --output_dir ./outputs \
        --num_steps 1000

    # I2SB
    python -m examples.pipelines.run_inference \
        --model i2sb \
        --checkpoint ./checkpoints/i2sb/sar2eo/checkpoint-10000 \
        --input_dir ./path/to/images \
        --output_dir ./outputs \
        --num_steps 100 \
        --deterministic
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Ensure project root in path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def load_ddbm_pipeline(checkpoint: str, device: str = "cuda", use_ema: bool = True):
    """Load DDBM pipeline from checkpoint."""
    from examples.pipelines.ddbm.pipeline import DDBMPipeline, DDBMUNet, DDBMScheduler

    ckpt = Path(checkpoint)
    unet_subfolder = "ema_unet" if use_ema and (ckpt / "ema_unet").is_dir() else "unet"
    unet = DDBMUNet.from_pretrained(str(ckpt), subfolder=unet_subfolder)
    scheduler = DDBMScheduler.from_pretrained(str(ckpt), subfolder="scheduler")
    pipeline = DDBMPipeline(unet=unet, scheduler=scheduler)
    return pipeline.to(device)


def load_ddib_pipeline(checkpoint: str, device: str = "cuda"):
    """Load DDIB pipeline from combined checkpoint directory."""
    from examples.pipelines.ddib.pipeline import DDIBPipeline, DDIBUNet, DDIBScheduler

    ckpt = Path(checkpoint)
    src_sub = "source_ema_unet" if (ckpt / "source_ema_unet").is_dir() else "source_unet"
    tgt_sub = "target_ema_unet" if (ckpt / "target_ema_unet").is_dir() else "target_unet"
    source_unet = DDIBUNet.from_pretrained(str(ckpt), subfolder=src_sub)
    target_unet = DDIBUNet.from_pretrained(str(ckpt), subfolder=tgt_sub)
    scheduler = DDIBScheduler.from_pretrained(str(ckpt), subfolder="scheduler")
    pipeline = DDIBPipeline(
        source_unet=source_unet,
        target_unet=target_unet,
        scheduler=scheduler,
    )
    return pipeline.to(device)


def load_i2sb_pipeline(checkpoint: str, device: str = "cuda", use_ema: bool = True):
    """Load I2SB pipeline from checkpoint."""
    from examples.pipelines.i2sb.pipeline import I2SBPipeline, I2SBUNet, I2SBScheduler

    ckpt = Path(checkpoint)
    unet_subfolder = "ema_unet" if use_ema and (ckpt / "ema_unet").is_dir() else "unet"
    unet = I2SBUNet.from_pretrained(str(ckpt), subfolder=unet_subfolder)
    scheduler = I2SBScheduler.from_pretrained(str(ckpt), subfolder="scheduler")
    pipeline = I2SBPipeline(unet=unet, scheduler=scheduler)
    return pipeline.to(device)


def load_bibbdm_pipeline(checkpoint: str, device: str = "cuda", use_ema: bool = True):
    """Load BiBBDM pipeline from checkpoint."""
    from examples.pipelines.bibbdm.pipeline import BiBBDMPipeline, BiBBDMUNet, BiBBDMScheduler

    ckpt = Path(checkpoint)
    unet_subfolder = "ema_unet" if use_ema and (ckpt / "ema_unet").is_dir() else "unet"
    unet = BiBBDMUNet.from_pretrained(str(ckpt), subfolder=unet_subfolder)
    scheduler = BiBBDMScheduler.from_pretrained(str(ckpt), subfolder="scheduler")
    pipeline = BiBBDMPipeline(unet=unet, scheduler=scheduler)
    return pipeline.to(device)


def main():
    parser = argparse.ArgumentParser(description="Run inference with self-contained pipelines.")
    parser.add_argument("--model", choices=["ddbm", "ddib", "i2sb", "bibbdm"], required=True)
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint directory")
    parser.add_argument("--input_dir", default=None, help="Directory of input images")
    parser.add_argument("--input_image", default=None, help="Single input image path")
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--num_steps", type=int, default=1000, help="Inference steps")
    parser.add_argument("--direction", default="b2a", choices=["b2a", "a2b"], help="BiBBDM only")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic sampling (churn/eta/ot_ode overrides).",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_type", default="pil", choices=["pil", "np", "pt"])
    args = parser.parse_args()

    # Load pipeline
    if args.model == "ddbm":
        pipe = load_ddbm_pipeline(args.checkpoint, args.device)
    elif args.model == "ddib":
        pipe = load_ddib_pipeline(args.checkpoint, args.device)
    elif args.model == "i2sb":
        pipe = load_i2sb_pipeline(args.checkpoint, args.device)
    elif args.model == "bibbdm":
        pipe = load_bibbdm_pipeline(args.checkpoint, args.device)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    # Gather inputs
    inputs = []
    if args.input_image:
        inputs.append(Path(args.input_image))
    if args.input_dir:
        for p in Path(args.input_dir).glob("*.png"):
            inputs.append(p)
        for p in Path(args.input_dir).glob("*.jpg"):
            inputs.append(p)
    if not inputs:
        print("No input images. Pass --input_image or --input_dir")
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.deterministic and args.model == "bibbdm":
        pipe.scheduler.eta = 0.0
        pipe.scheduler.config.eta = 0.0

    # Run inference
    for i, img_path in enumerate(inputs):
        img = Image.open(img_path).convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0
        if arr.ndim == 2:
            tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        else:
            tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1,C,H,W)
        tensor = tensor.to(args.device) * 2 - 1  # [-1,1]

        churn = 0.0 if args.deterministic else 0.33
        eta = 0.0 if args.deterministic else 1.0
        ot_ode = args.deterministic

        if args.model == "ddbm":
            out = pipe(
                source_image=tensor,
                num_inference_steps=args.num_steps,
                guidance=1.0,
                churn_step_ratio=churn,
                output_type=args.output_type,
            )
        elif args.model == "ddib":
            out = pipe(
                source_image=tensor,
                num_inference_steps=args.num_steps,
                clip_denoised=True,
                eta=eta,
                output_type=args.output_type,
            )
        elif args.model == "i2sb":
            out = pipe(
                source_image=tensor,
                nfe=args.num_steps,
                ot_ode=ot_ode,
                clip_denoise=False,
                output_type=args.output_type,
            )
        elif args.model == "bibbdm":
            out = pipe(
                source_image=tensor,
                direction=args.direction,
                num_inference_steps=args.num_steps,
                clip_denoised=False,
                output_type=args.output_type,
            )

        images = out.images
        if args.output_type == "pil":
            for j, pil_img in enumerate(images):
                out_path = out_dir / f"{img_path.stem}_{i}_{j}.png"
                pil_img.save(out_path)
                print(f"Saved {out_path}")
        elif args.output_type == "pt":
            torch.save(images, out_dir / f"{img_path.stem}_{i}.pt")

    print(f"Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
