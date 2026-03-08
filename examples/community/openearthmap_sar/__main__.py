# Copyright (c) 2026 EarthBridge Team.
# Credits: OpenEarthMap-SAR CUT models. Architecture from Park et al. "Contrastive Learning for Unpaired Image-to-Image Translation" ECCV 2020.

"""CLI entry point for OpenEarthMap-SAR inference.

Usage:
    python -m examples.community.openearthmap_sar --checkpoint-dir PATH [--model sar2opt] [--input IMG] [--output OUT]
"""

from __future__ import annotations

import argparse

import numpy as np
from PIL import Image

from examples.community.openearthmap_sar.pipeline import (
    OPENEARTHMAP_SAR_MODELS,
    load_openearthmap_sar_pipeline,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenEarthMap-SAR inference")
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="Root directory (e.g. CUT-OpenEarthMap-SAR)")
    parser.add_argument("--model", type=str, default="sar2opt", choices=list(OPENEARTHMAP_SAR_MODELS))
    parser.add_argument("--input", "-i", type=str, default=None)
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    pipeline = load_openearthmap_sar_pipeline(
        checkpoint_dir=args.checkpoint_dir,
        model_name=args.model,
        device=args.device,
    )
    if args.input:
        img = Image.open(args.input).convert("RGB")
    else:
        img = Image.fromarray((np.random.rand(256, 256, 3) * 255).astype(np.uint8))
    out = pipeline(source_image=img, output_type="pil")
    result = out.images[0] if isinstance(out.images, list) else out.images
    if args.output:
        result.save(args.output)
        print(f"Saved to {args.output}")
    else:
        print(f"Inference OK: input {img.size} -> output {result.size}")


if __name__ == "__main__":
    main()
