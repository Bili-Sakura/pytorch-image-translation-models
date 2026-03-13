#!/usr/bin/env python3
# Copyright (c) 2026 EarthBridge Team.
# Credits: xuekt98/BBDM (Li et al., CVPR 2023).
"""Convert original BBDM checkpoints to community ``unet/`` layout.

Usage:
    python -m examples.community.bbdm.convert_ckpt_to_unet \
      --raw-root "/root/worksapce/models/raw/BBDM Checkpoints" \
      --output-root "/root/worksapce/models/BiliSakura/BBDM-ckpt"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required for BBDM conversion. Install with: pip install pyyaml"
        ) from exc
    with open(path, encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def _try_list_checkpoints(folder: Path) -> list[Path]:
    patterns = ("*.pt", "*.pth", "*.ckpt")
    out: list[Path] = []
    for pat in patterns:
        out.extend(sorted(folder.glob(pat)))
    return out


def _extract_model_state(raw_state: Any) -> dict[str, torch.Tensor]:
    if not isinstance(raw_state, dict):
        raise TypeError(f"Unexpected checkpoint object type: {type(raw_state)}")

    state = raw_state
    for key in ("state_dict", "model", "ema_model"):
        if key in state and isinstance(state[key], dict):
            state = state[key]

    if not isinstance(state, dict):
        raise TypeError("Could not resolve a state_dict from checkpoint")

    denoise_prefixes = (
        "denoise_fn.",
        "model.denoise_fn.",
        "module.denoise_fn.",
        "model.module.denoise_fn.",
    )
    openai_roots = ("input_blocks.", "middle_block.", "output_blocks.", "out.", "time_embed.")

    extracted: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        new_key = None
        for prefix in denoise_prefixes:
            if key.startswith(prefix):
                new_key = key[len(prefix) :]
                break
        if new_key is None:
            if key.startswith(openai_roots):
                new_key = key
            elif key.startswith("module.") and key[len("module.") :].startswith(openai_roots):
                new_key = key[len("module.") :]
        if new_key is not None and torch.is_tensor(value):
            extracted[new_key] = value.detach().cpu().contiguous()

    if not extracted:
        # fallback: checkpoint may already be bare UNet state_dict
        if any(k.startswith(openai_roots) for k in state.keys()):
            extracted = {
                k: v.detach().cpu().contiguous()
                for k, v in state.items()
                if torch.is_tensor(v) and k.startswith(openai_roots)
            }
    if not extracted:
        raise RuntimeError(
            "No OpenAI-style denoise_fn weights found. Expected keys like "
            "'denoise_fn.input_blocks.0.0.weight'."
        )
    return extracted


def _attention_to_ds(attn: list[int] | tuple[int, ...], image_size: int) -> list[int]:
    out: list[int] = []
    for v in attn:
        if v <= 0:
            continue
        # Original configs often store resolutions (32,16,8). Convert to ds (2,4,8).
        if v > 8 and image_size % v == 0:
            out.append(image_size // v)
        else:
            out.append(v)
    # preserve order, remove duplicates
    seen = set()
    uniq = []
    for v in out:
        if v not in seen:
            uniq.append(v)
            seen.add(v)
    return uniq or [2, 4, 8]


def _infer_attention_ds_from_keys(
    *,
    denoise_keys: list[str],
    channel_mult: list[int],
    num_res_blocks: int,
) -> list[int]:
    """Infer UNet attention ds locations directly from OpenAI-style weight keys."""
    if not denoise_keys:
        return []

    input_block_ds: dict[int, int] = {}
    idx = 1  # input_blocks.0 is the initial conv block
    ds = 1
    for level, _ in enumerate(channel_mult):
        for _ in range(num_res_blocks):
            input_block_ds[idx] = ds
            idx += 1
        if level != len(channel_mult) - 1:
            idx += 1  # downsample block
            ds *= 2

    output_block_ds: dict[int, int] = {}
    ds = 2 ** (len(channel_mult) - 1)
    out_idx = 0
    for rev_level, _ in enumerate(channel_mult[::-1]):
        for i in range(num_res_blocks + 1):
            output_block_ds[out_idx] = ds
            if rev_level != len(channel_mult) - 1 and i == num_res_blocks:
                ds //= 2
            out_idx += 1

    attn_ds: list[int] = []
    seen: set[int] = set()
    for key in denoise_keys:
        m = None
        if ".1.qkv.weight" in key and key.startswith("input_blocks."):
            m = key.split(".", 2)[1]
            if m.isdigit():
                block_idx = int(m)
                ds_val = input_block_ds.get(block_idx)
                if ds_val is not None and ds_val not in seen:
                    seen.add(ds_val)
                    attn_ds.append(ds_val)
        elif ".1.qkv.weight" in key and key.startswith("output_blocks."):
            m = key.split(".", 2)[1]
            if m.isdigit():
                block_idx = int(m)
                ds_val = output_block_ds.get(block_idx)
                if ds_val is not None and ds_val not in seen:
                    seen.add(ds_val)
                    attn_ds.append(ds_val)
    return attn_ds


def _build_unet_config(
    yaml_cfg: dict[str, Any],
    ckpt_name: str,
    *,
    denoise_keys: list[str] | None = None,
) -> dict[str, Any]:
    bb_params = yaml_cfg["model"]["BB"]["params"]
    unet = bb_params["UNetParams"]
    image_size = int(unet.get("image_size", 64))
    attn = unet.get("attention_resolutions", [32, 16, 8])
    channel_mult = list(unet.get("channel_mult", [1, 4, 8]))
    num_res_blocks = int(unet.get("num_res_blocks", 2))
    attn_ds = _attention_to_ds(list(attn), image_size=image_size)
    if denoise_keys is not None:
        inferred = _infer_attention_ds_from_keys(
            denoise_keys=denoise_keys,
            channel_mult=channel_mult,
            num_res_blocks=num_res_blocks,
        )
        if inferred:
            attn_ds = inferred
        else:
            # Many LBBDM checkpoints only keep middle attention.
            attn_ds = []

    raw_condition_key = unet.get("condition_key", "nocond")
    if raw_condition_key is None or str(raw_condition_key).strip() == "":
        condition_key = "nocond"
    else:
        condition_key = str(raw_condition_key)
    condition_mode = "concat" if condition_key != "nocond" else "none"
    return {
        "image_size": image_size,
        "in_channels": int(unet.get("in_channels", 3)),
        "model_channels": int(unet.get("model_channels", 128)),
        "out_channels": int(unet.get("out_channels", 3)),
        "num_res_blocks": num_res_blocks,
        "attention_resolutions": attn_ds,
        "dropout": float(unet.get("dropout", 0.0)),
        "channel_mult": channel_mult,
        "conv_resample": bool(unet.get("conv_resample", True)),
        "use_scale_shift_norm": bool(unet.get("use_scale_shift_norm", True)),
        "num_heads": int(unet.get("num_heads", 8)),
        "num_head_channels": int(unet.get("num_head_channels", 64)),
        "resblock_updown": bool(unet.get("resblock_updown", True)),
        "use_spatial_transformer": bool(unet.get("use_spatial_transformer", False)),
        "condition_key": condition_key,
        "condition_mode": condition_mode,
        "_class_name": "OpenAIBBDMUNet",
        "_converted_from": ckpt_name,
    }


def _build_scheduler_config(yaml_cfg: dict[str, Any]) -> dict[str, Any]:
    p = yaml_cfg["model"]["BB"]["params"]
    return {
        "num_timesteps": int(p.get("num_timesteps", 1000)),
        "mt_type": str(p.get("mt_type", "linear")),
        "eta": float(p.get("eta", 1.0)),
        "max_var": float(p.get("max_var", 1.0)),
        "skip_sample": bool(p.get("skip_sample", True)),
        "sample_step": int(p.get("sample_step", 200)),
        "sample_step_type": str(p.get("sample_type", "linear")),
        "objective": str(p.get("objective", "grad")),
    }


def _check_src_bbdm_compat(weights: dict[str, torch.Tensor]) -> tuple[bool, str]:
    # src BBDMUNet expects diffusers-style keys under `unet.*` rather than OpenAI-style.
    openai_like = any(k.startswith("input_blocks.") for k in weights.keys())
    if openai_like:
        return (
            False,
            "OpenAI-style key layout detected (input_blocks/middle_block/output_blocks), "
            "not diffusers UNet2DModel format expected by src.BBDMUNet.",
        )
    return True, "No obvious incompatibility detected from key layout."


def _select_checkpoint(ckpts: list[Path], yaml_stem: str) -> Path:
    if not ckpts:
        raise ValueError("No checkpoints to select from")
    stem_lower = yaml_stem.lower()
    for ckpt in ckpts:
        if stem_lower in ckpt.stem.lower():
            return ckpt
    return ckpts[0]


def convert_one(raw_subdir: Path, yaml_path: Path, output_root: Path) -> None:
    yaml_stem = yaml_path.stem
    if "-f" in yaml_stem:
        name = yaml_stem
    else:
        name = raw_subdir.name
    out_dir = output_root / name
    out_unet = out_dir / "unet"
    out_sched = out_dir / "scheduler"
    out_unet.mkdir(parents=True, exist_ok=True)
    out_sched.mkdir(parents=True, exist_ok=True)

    yaml_cfg = _load_yaml(yaml_path)
    model_cfg = yaml_cfg.get("model", {})
    if "BB" not in model_cfg:
        print(f"[skip] {yaml_path} does not look like a BBDM/LBBDM config.")
        return
    ckpts = _try_list_checkpoints(raw_subdir)

    scheduler_cfg = _build_scheduler_config(yaml_cfg)
    with open(out_sched / "scheduler_config.json", "w", encoding="utf-8") as f:
        json.dump(scheduler_cfg, f, indent=2)

    if not ckpts:
        status = {
            "status": "pending_weights",
            "message": "No checkpoint file (*.pt/*.pth/*.ckpt) found in source folder.",
            "source_folder": str(raw_subdir),
            "expected_output": str(out_unet / "diffusion_pytorch_model.safetensors"),
        }
        with open(out_dir / "conversion_status.json", "w", encoding="utf-8") as f:
            json.dump(status, f, indent=2)
        # still write a config inferred from YAML so layout is ready.
        with open(out_unet / "config.json", "w", encoding="utf-8") as f:
            json.dump(_build_unet_config(yaml_cfg, ckpt_name="N/A"), f, indent=2)
        print(f"[{name}] no checkpoint found; wrote scaffold config + scheduler.")
        return

    ckpt = _select_checkpoint(ckpts, yaml_stem=yaml_stem)
    state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    denoise = _extract_model_state(state)

    unet_cfg = _build_unet_config(
        yaml_cfg,
        ckpt.name,
        denoise_keys=list(denoise.keys()),
    )
    with open(out_unet / "config.json", "w", encoding="utf-8") as f:
        json.dump(unet_cfg, f, indent=2)
    save_file(denoise, str(out_unet / "diffusion_pytorch_model.safetensors"))

    compatible, reason = _check_src_bbdm_compat(denoise)
    with open(out_dir / "conversion_status.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "status": "converted",
                "source_checkpoint": str(ckpt),
                "src_bbdm_compatible": compatible,
                "src_bbdm_check_reason": reason,
                "recommended_loader": (
                    "src.BBDMPipeline.from_pretrained"
                    if compatible
                    else "examples.community.bbdm.load_bbdm_community_pipeline"
                ),
            },
            f,
            indent=2,
        )
    print(f"[{name}] converted {ckpt.name}")
    print(f"  src.BBDM compatibility: {compatible} ({reason})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert BBDM checkpoints to unet/ format")
    parser.add_argument("--raw-root", type=str, required=True, help="Path to raw BBDM checkpoints root")
    parser.add_argument("--output-root", type=str, required=True, help="Path to converted output root")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    yaml_files = sorted(raw_root.glob("*/*.yaml"))
    if not yaml_files:
        raise FileNotFoundError(f"No YAML config files found under {raw_root}")

    for yaml_path in yaml_files:
        convert_one(yaml_path.parent, yaml_path, output_root)


if __name__ == "__main__":
    main()
