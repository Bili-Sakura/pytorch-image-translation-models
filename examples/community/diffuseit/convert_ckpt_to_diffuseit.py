#!/usr/bin/env python3
# Copyright (c) 2026 EarthBridge Team.
# Credits: DiffuseIT (Kwon & Ye, ICLR 2023) - https://github.com/cyclomon/DiffuseIT

"""Convert raw DiffuseIT .pt checkpoints to BiliSakura layout.

Usage:
    python -m examples.community.diffuseit.convert_ckpt_to_diffuseit \\
        --raw-root /root/worksapce/models/raw/DiffuseIT-ckpt-raw \\
        --output-root /root/worksapce/models/BiliSakura/DiffuseIT-ckpt
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

# DiffuseIT guided_diffusion configs per model
DIFFUSEIT_CONFIGS = {
    "256x256_diffusion_uncond.pt": {
        "image_size": 256,
        "num_channels": 256,
        "num_res_blocks": 2,
        "channel_mult": [1, 1, 2, 2, 4, 4],
        "attention_resolutions": [8, 16, 32],  # 256//32, 256//16, 256//8
        "out_channels": 6,
        "learn_sigma": True,
    },
    "ffhq_10m.pt": {
        "image_size": 256,
        "num_channels": 128,
        "num_res_blocks": 1,
        "channel_mult": [1, 1, 2, 2, 4, 4],
        "attention_resolutions": [16],
        "out_channels": 6,
        "learn_sigma": True,
    },
    "512x512_diffusion.pt": {
        "image_size": 512,
        "num_channels": 256,
        "num_res_blocks": 2,
        "channel_mult": [0.5, 1, 1, 2, 2, 4, 4],
        "attention_resolutions": [8, 16, 32],
        "out_channels": 6,
        "learn_sigma": True,
    },
}


def _infer_config(state: dict, ckpt_name: str) -> dict:
    """Infer config from state dict; fallback to known configs."""
    if ckpt_name in DIFFUSEIT_CONFIGS:
        cfg = DIFFUSEIT_CONFIGS[ckpt_name].copy()
    else:
        w = state["input_blocks.0.0.weight"]
        model_ch = int(w.shape[0])
        out_w = state.get("out.2.weight", state.get("out.0.weight"))
        out_ch = int(out_w.shape[0]) if out_w is not None else 3
        cfg = {
            "num_channels": model_ch,
            "out_channels": out_ch,
            "num_res_blocks": 2,
            "channel_mult": [1, 1, 2, 2, 4, 4],
            "attention_resolutions": [8, 16, 32],
            "learn_sigma": out_ch == 6,
        }
    cfg["_class_name"] = "DiffuseITGuidedDiffusionUNet"
    cfg["_converted_from"] = ckpt_name
    return cfg


def convert_checkpoint(
    raw_pt_path: Path,
    output_dir: Path,
    model_name: str,
) -> None:
    """Convert single .pt to unet/ layout (safetensors only, no .pt)."""
    state = torch.load(str(raw_pt_path), map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "ema_model" in state:
        state = state["ema_model"]

    config = _infer_config(dict(state), raw_pt_path.name)
    config_to_save = {k: v for k, v in config.items() if not k.startswith("_")}
    config_to_save["_class_name"] = config["_class_name"]
    config_to_save["_converted_from"] = config["_converted_from"]

    out_unet = output_dir / "unet"
    out_unet.mkdir(parents=True, exist_ok=True)

    config_path = out_unet / "config.json"
    safetensors_path = out_unet / "diffusion_pytorch_model.safetensors"

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_to_save, f, indent=2)

    save_file(state, str(safetensors_path))

    print(f"Converted {raw_pt_path.name} -> {output_dir}/")
    print(f"  config: {config_path}")
    print(f"  safetensors: {safetensors_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert DiffuseIT raw .pt checkpoints to BiliSakura layout"
    )
    parser.add_argument(
        "--raw-root",
        type=str,
        default="/root/worksapce/models/raw/DiffuseIT-ckpt-raw",
        help="Path to raw checkpoint directory",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="/root/worksapce/models/BiliSakura/DiffuseIT-ckpt",
        help="Output root (e.g. BiliSakura/DiffuseIT-ckpt)",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["imagenet256-uncond", "ffhq-256"],
        help="Output subfolder names: imagenet256-uncond, ffhq-256, imagenet512",
    )
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    output_root = Path(args.output_root)

    # Maps output subfolder to raw checkpoint filename
    model_files = {
        "imagenet256-uncond": "256x256_diffusion_uncond.pt",
        "ffhq-256": "ffhq_10m.pt",
        "imagenet512": "512x512_diffusion.pt",
    }

    id_src = raw_root / "model_ir_se50.pth"
    id_state = None
    if id_src.exists():
        id_state = torch.load(str(id_src), map_location="cpu", weights_only=True)
        if isinstance(id_state, dict) and "state_dict" in id_state:
            id_state = id_state["state_dict"]

    for name in args.models:
        fname = model_files.get(name)
        if not fname:
            print(f"Skipping {name} (unknown model)")
            continue
        raw_path = raw_root / fname
        if not raw_path.exists():
            print(f"Skipping {name} ({fname} not found)")
            continue
        out_dir = output_root / name
        convert_checkpoint(raw_path, out_dir, name)

        # Copy id_model into ffhq-256 for self-contained subfolder
        if name == "ffhq-256" and id_state is not None:
            id_dst = out_dir / "id_model"
            id_dst.mkdir(parents=True, exist_ok=True)
            save_file(id_state, str(id_dst / "model_ir_se50.safetensors"))
            with open(id_dst / "config.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "_class_name": "ArcFaceIR_SE50",
                        "_converted_from": "model_ir_se50.pth",
                    },
                    f,
                    indent=2,
                )
            print(f"  + id_model/ (self-contained)")


if __name__ == "__main__":
    main()
