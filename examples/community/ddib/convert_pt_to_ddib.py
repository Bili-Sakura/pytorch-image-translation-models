#!/usr/bin/env python3
# Copyright (c) 2026 EarthBridge Team.
"""Convert raw DDIB .pt checkpoints to BiliSakura format (source_unet/ + target_unet/).

Supports:
- Synthetic (log2D0, log2D1, ...): 2-channel unconditional diffusion
- ImageNet256: 3-channel, strips label_emb for unconditional use

Usage:
    # Convert Synthetic log2D0 -> log2D1 translation
    python -m examples.community.ddib.convert_pt_to_ddib \\
        --source-pt models/raw/DDIB-ckpt-raw/Synthetic/log2D0/model230000.pt \\
        --target-pt models/raw/DDIB-ckpt-raw/Synthetic/log2D1/ema_0.9999_200000.pt \\
        --output-dir models/BiliSakura/DDIB-ckpt/Synthetic-log2D0-to-log2D1

    # Convert ImageNet256 (same model as source and target for identity/experiment)
    python -m examples.community.ddib.convert_pt_to_ddib \\
        --source-pt models/raw/DDIB-ckpt-raw/ImageNet256/256x256_diffusion.pt \\
        --target-pt models/raw/DDIB-ckpt-raw/ImageNet256/256x256_diffusion.pt \\
        --output-dir models/BiliSakura/DDIB-ckpt/ImageNet256 \\
        --strip-class-embed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file


# OpenAI-style key roots (exclude label_emb for unconditional)
OPENAI_ROOTS = ("input_blocks.", "middle_block.", "output_blocks.", "out.", "time_embed.")


def _extract_state(pt_path: Path, strip_label_emb: bool = False) -> dict:
    """Load and extract UNet state dict from .pt checkpoint."""
    raw = torch.load(str(pt_path), map_location="cpu", weights_only=True)
    state = raw
    for key in ("state_dict", "model", "ema_model"):
        if key in state and isinstance(state[key], dict):
            state = state[key]
            break

    extracted = {}
    for k, v in state.items():
        if not torch.is_tensor(v):
            continue
        if strip_label_emb and (k.startswith("label_emb") or "label" in k):
            continue
        if any(k.startswith(r) for r in OPENAI_ROOTS):
            extracted[k] = v.detach().cpu().contiguous()

    if not extracted:
        raise RuntimeError(
            f"No OpenAI-style weights found in {pt_path}. "
            "Expected keys like input_blocks.*, middle_block.*, output_blocks.*"
        )
    return extracted


def _infer_config(state: dict) -> dict:
    """Infer UNet config from state dict keys and shapes."""
    w0 = state["input_blocks.0.0.weight"]
    in_channels = int(w0.shape[1])
    model_channels = int(w0.shape[0])

    out_w = state.get("out.2.weight", state.get("out.0.weight"))
    out_channels = int(out_w.shape[0]) if out_w is not None else in_channels

    # Time embed dim from first ResBlock emb_layers
    ted = 256
    for k in state:
        if "emb_layers.1.weight" in k and "input_blocks" in k:
            ted = int(state[k].shape[0])
            break

    # use_scale_shift_norm: emb_layers outputs 2*ch
    use_scale_shift_norm = False
    for k, v in state.items():
        if "emb_layers.1.weight" in k and "input_blocks.1.0" in k:
            out_dim = v.shape[0]
            # find corresponding resblock out channels
            ob_key = k.replace("emb_layers.1.weight", "out_layers.3.weight")
            if ob_key in state:
                ch = state[ob_key].shape[0]
                use_scale_shift_norm = out_dim == 2 * ch
            break

    # Count num_res_blocks and channel_mult from structure
    num_res_blocks = 3
    channel_mult = [1, 2, 2, 4]
    # Infer from input_blocks
    ib_count = len(set(k.split(".")[1] for k in state if k.startswith("input_blocks.")))
    if ib_count == 2:  # 0=conv, 1=resblocks
        num_res_blocks = len(
            set(k.split(".")[2] for k in state if k.startswith("input_blocks.1."))
        )
        channel_mult = [1]  # single level

    # No attention if no qkv
    has_attn = any("qkv" in k for k in state)
    attention_resolutions = (2, 4, 8) if has_attn else ()

    return {
        "in_channels": in_channels,
        "model_channels": model_channels,
        "out_channels": out_channels,
        "num_res_blocks": num_res_blocks,
        "attention_resolutions": list(attention_resolutions),
        "channel_mult": channel_mult,
        "time_embed_dim": ted,
        "use_scale_shift_norm": use_scale_shift_norm,
        "conv_resample": False,
    }


def convert_pt_to_ddib(
    source_pt: str | Path,
    target_pt: str | Path,
    output_dir: str | Path,
    *,
    strip_label_emb: bool = False,
) -> None:
    """Convert source and target .pt checkpoints to DDIB BiliSakura layout."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_state = _extract_state(Path(source_pt), strip_label_emb=strip_label_emb)
    target_state = _extract_state(Path(target_pt), strip_label_emb=strip_label_emb)

    config = _infer_config(source_state)
    config_target = _infer_config(target_state)
    if config != config_target:
        raise ValueError(
            "Source and target checkpoints must have the same architecture. "
            f"Got {config} vs {config_target}"
        )

    # Save source_unet
    source_dir = output_dir / "source_unet"
    source_dir.mkdir(exist_ok=True)
    config_path = source_dir / "config.json"
    weights_path = source_dir / "diffusion_pytorch_model.safetensors"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    save_file(source_state, str(weights_path))
    print(f"Saved source_unet/ from {Path(source_pt).name}")

    # Save target_unet
    target_dir = output_dir / "target_unet"
    target_dir.mkdir(exist_ok=True)
    config_path = target_dir / "config.json"
    weights_path = target_dir / "diffusion_pytorch_model.safetensors"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    save_file(target_state, str(weights_path))
    print(f"Saved target_unet/ from {Path(target_pt).name}")

    # Optional scheduler
    sched_dir = output_dir / "scheduler"
    sched_dir.mkdir(exist_ok=True)
    sched_config = {
        "num_train_timesteps": 1000,
        "noise_schedule": "linear",
        "learn_sigma": False,
        "predict_xstart": False,
        "rescale_timesteps": False,
    }
    with open(sched_dir / "scheduler_config.json", "w", encoding="utf-8") as f:
        json.dump(sched_config, f, indent=2)
    print(f"Converted -> {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Convert DDIB .pt checkpoints to BiliSakura format"
    )
    parser.add_argument("--source-pt", type=str, required=True, help="Source domain .pt checkpoint")
    parser.add_argument("--target-pt", type=str, required=True, help="Target domain .pt checkpoint")
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory (e.g. models/BiliSakura/DDIB-ckpt/dataset-name)",
    )
    parser.add_argument(
        "--strip-class-embed",
        action="store_true",
        help="Strip label_emb for class-conditional checkpoints (ImageNet256)",
    )
    args = parser.parse_args()
    convert_pt_to_ddib(
        args.source_pt,
        args.target_pt,
        args.output_dir,
        strip_label_emb=args.strip_class_embed,
    )


if __name__ == "__main__":
    main()
