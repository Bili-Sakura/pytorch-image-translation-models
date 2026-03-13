#!/usr/bin/env python3
# Copyright (c) 2026 EarthBridge Team.
# Credits: Bosch Research LDDBM (Multimodal-Distribution-Translation-MDT).
"""Convert raw MDT .pt checkpoints to project format (config.json + safetensors per subfolder).

Usage:
    python -m examples.community.mdt.convert_pt_to_mdt \\
        /path/to/mdt-checkpoints \\
        --encoder-x encoder_x.pt --encoder-y encoder_y.pt \\
        --decoder-x decoder_x.pt --bridge bridge.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file


def _extract_state_dict(obj) -> dict:
    """Extract state dict from various checkpoint formats."""
    if isinstance(obj, dict):
        if "state_dict" in obj:
            return obj["state_dict"]
        if "ema_model" in obj:
            return obj["ema_model"]
        if "model" in obj:
            return obj["model"]
        # Assume it is the state dict itself (all values are tensors)
        if obj and isinstance(next(iter(obj.values())), torch.Tensor):
            return obj
    return obj


def convert_pt_to_mdt(
    output_dir: str | Path,
    *,
    encoder_x: str | Path | None = None,
    encoder_y: str | Path | None = None,
    decoder_x: str | Path | None = None,
    bridge: str | Path | None = None,
    task: str = "sr_16_to_128",
) -> None:
    """Convert .pt checkpoints to encoder_x/, encoder_y/, decoder_x/, bridge/ subfolders
    each with config.json + diffusion_pytorch_model.safetensors.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    components = {
        "encoder_x": encoder_x,
        "encoder_y": encoder_y,
        "decoder_x": decoder_x,
        "bridge": bridge,
    }
    for name, path in components.items():
        if path is None:
            # Look for {name}.pt in output_dir's parent (sibling of output)
            parent = Path(output_dir).parent
            default = (parent / f"{name}.pt") if parent else Path(f"{name}.pt")
            if default.exists():
                path = default
            else:
                raise FileNotFoundError(f"Missing {name}: provide --{name.replace('_', '-')} or {name}.pt")
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
        state_dict = _extract_state_dict(ckpt)
        if not isinstance(state_dict, dict):
            raise TypeError(f"{name}: expected state dict, got {type(state_dict)}")

        subdir = out / name
        subdir.mkdir(parents=True, exist_ok=True)
        config = {
            "task": task,
            "component": name,
            "_class_name": "MDTComponent",
            "_converted_from": path.name,
        }
        config_path = subdir / "config.json"
        weights_path = subdir / "diffusion_pytorch_model.safetensors"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        save_file(state_dict, str(weights_path))
        print(f"Converted {name}: {path} -> {subdir}/")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert MDT .pt checkpoints to config.json + safetensors"
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Output directory (e.g. mdt-checkpoints/ or path/to/mdt-safetensors)",
    )
    parser.add_argument("--encoder-x", type=str, default=None, help="Path to encoder_x.pt")
    parser.add_argument("--encoder-y", type=str, default=None, help="Path to encoder_y.pt")
    parser.add_argument("--decoder-x", type=str, default=None, help="Path to decoder_x.pt")
    parser.add_argument("--bridge", type=str, default=None, help="Path to bridge.pt")
    parser.add_argument("--task", type=str, default="sr_16_to_128", help="Task name for config")
    args = parser.parse_args()

    convert_pt_to_mdt(
        args.output_dir,
        encoder_x=args.encoder_x,
        encoder_y=args.encoder_y,
        decoder_x=args.decoder_x,
        bridge=args.bridge,
        task=args.task,
    )


if __name__ == "__main__":
    main()
