# Copyright (c) 2026 EarthBridge Team.
# Credits: CDTSDE/PSCDE (solar defect identification, projects/CDTSDE).
"""CLI for CDTSDE inference.

Usage:
    python -m examples.community.cdtsde \\
        --checkpoint /path/to/CDTSDE-ckpt/solar-defect-pscde \\
        --input electroluminescence.png \\
        --output semantic_mask.png

Requires diffusers-style checkpoint (unet/, controlnet/, vae/, etc.).
Run convert_ckpt_to_cdtsde first for raw .ckpt conversion.
"""

from __future__ import annotations

import argparse

from PIL import Image

from examples.community.cdtsde import load_cdtsde_community_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="CDTSDE/PSCDE solar defect inference")
    parser.add_argument("--checkpoint", "-c", type=str, required=True)
    parser.add_argument("--input", "-i", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, required=True)
    parser.add_argument("--cdtsde-src", type=str, default=None)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    pipe = load_cdtsde_community_pipeline(
        args.checkpoint,
        cdtsde_src_path=args.cdtsde_src,
        device=args.device,
    )
    pipe.to(args.device)

    img = Image.open(args.input).convert("RGB")
    out = pipe(
        source_image=img,
        num_inference_steps=args.steps,
        size=args.size,
        output_type="pil",
    )
    out.images[0].save(args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
