#!/usr/bin/env python3
# Copyright (c) 2026 EarthBridge Team.
# Credits: alexzhou907/DDBM (Zhou et al., ICLR 2024).
"""Convert raw DDBM .pt checkpoint to unet/ format (safetensors + config.json).

Usage:
    python -m examples.community.ddbm.convert_pt_to_unet \\
        /path/to/DDBM-ckpt/edges2handbags-vp \\
        --checkpoint e2h_ema_0.9999_420000.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file


def _infer_config(state: dict) -> dict:
    w = state["input_blocks.0.0.weight"]
    unet_in_ch = int(w.shape[1])
    model_ch = int(w.shape[0])
    base_ch = unet_in_ch // 2 if unet_in_ch == 6 else unet_in_ch
    out_w = state.get("out.2.weight", state.get("out.0.weight"))
    out_ch = int(out_w.shape[0]) if out_w is not None else 3
    return {
        "in_channels": base_ch,
        "model_channels": model_ch,
        "out_channels": out_ch,
        "num_res_blocks": 3,
        "attention_resolutions": [2, 4, 8],
        "channel_mult": [1, 2, 3, 4],
        "conv_resample": False,
        "condition_mode": "concat",
        "_class_name": "OpenAIDDBMUNet",
        "_converted_from": "pt",
    }


def convert_pt_to_unet(
    model_dir: str | Path,
    *,
    checkpoint_name: str | None = None,
    output_subfolder: str = "unet",
) -> None:
    """Convert .pt checkpoint to unet/config.json + unet/diffusion_pytorch_model.safetensors."""
    path = Path(model_dir)
    if checkpoint_name is None:
        pt_files = list(path.glob("*.pt"))
        if not pt_files:
            raise FileNotFoundError(f"No .pt checkpoint found in {path}")
        ckpt_path = pt_files[0]
    else:
        ckpt_path = path / checkpoint_name
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "ema_model" in state:
        state = state["ema_model"]

    config = _infer_config(state)
    config_to_save = {k: v for k, v in config.items() if not k.startswith("_")}
    config_to_save["_class_name"] = "OpenAIDDBMUNet"
    config_to_save["_converted_from"] = ckpt_path.name

    out_dir = path / output_subfolder
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / "config.json"
    weights_path = out_dir / "diffusion_pytorch_model.safetensors"

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_to_save, f, indent=2)

    save_file(state, str(weights_path))
    print(f"Converted {ckpt_path.name} -> {out_dir}/")
    print(f"  config: {config_path}")
    print(f"  weights: {weights_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert DDBM .pt checkpoint to unet/ format"
    )
    parser.add_argument(
        "model_dir",
        type=str,
        help="Path to model directory (e.g. DDBM-ckpt/edges2handbags-vp)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint filename (default: first .pt in directory)",
    )
    parser.add_argument(
        "--output-subfolder",
        type=str,
        default="unet",
        help="Output subfolder (default: unet)",
    )
    args = parser.parse_args()
    convert_pt_to_unet(
        args.model_dir,
        checkpoint_name=args.checkpoint,
        output_subfolder=args.output_subfolder,
    )


if __name__ == "__main__":
    main()
