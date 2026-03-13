#!/usr/bin/env python3
# Copyright (c) 2026 EarthBridge Team.
# Credits: xuekt98/BBDM (Li et al., CVPR 2023).
"""Convert legacy BBDM VQGAN checkpoints to Diffusers VQModel layout.

Usage:
    conda activate rsgen
    python -m examples.community.bbdm.convert_ckpt_to_vqmodel \
      --raw-dir "/root/worksapce/models/raw/BBDM Checkpoints/vqgan_f8" \
      --raw-dir "/root/worksapce/models/raw/BBDM Checkpoints/vqgan_f16_16384" \
      --raw-dir "/root/worksapce/models/raw/BBDM Checkpoints/vqgan-f4-8192" \
      --in-place \
      --subfolder "vqvae"
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import torch
from diffusers import VQModel


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required for VQGAN conversion. Install with: pip install pyyaml"
        ) from exc
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _find_checkpoint(folder: Path) -> Path:
    preferred = ("model.ckpt", "vq-f8.ckpt", "diffusion_model.pt")
    for name in preferred:
        p = folder / name
        if p.exists():
            return p
    all_ckpts: list[Path] = []
    for pat in ("*.ckpt", "*.pt", "*.pth"):
        all_ckpts.extend(sorted(folder.glob(pat)))
    if not all_ckpts:
        raise FileNotFoundError(f"No checkpoint file found in {folder}")
    return all_ckpts[0]


def _load_model_state(ckpt_path: Path) -> dict[str, torch.Tensor]:
    raw = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "state_dict" in raw and isinstance(raw["state_dict"], dict):
        state = raw["state_dict"]
    elif isinstance(raw, dict):
        state = raw
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(raw)}")

    out: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not torch.is_tensor(value):
            continue
        if key.startswith("loss.") or key.startswith("discriminator."):
            continue
        out[key] = value.detach().cpu().contiguous()
    return out


def _parse_config_from_yaml(path: Path) -> dict[str, Any]:
    cfg = _load_yaml(path)
    params = cfg["model"]["params"]
    dd = params["ddconfig"]
    ch_mult = list(dd["ch_mult"])
    attn_resolutions = set(dd.get("attn_resolutions", []))
    resolution = int(dd["resolution"])
    down_block_types = []
    up_block_types = []
    for idx in range(len(ch_mult)):
        ds = 2**idx
        cur_res = resolution // ds
        use_attn = cur_res in attn_resolutions
        down_block_types.append("AttnDownEncoderBlock2D" if use_attn else "DownEncoderBlock2D")
    for idx in range(len(ch_mult)):
        old_up_idx = len(ch_mult) - 1 - idx
        ds = 2**old_up_idx
        cur_res = resolution // ds
        use_attn = cur_res in attn_resolutions
        up_block_types.append("AttnUpDecoderBlock2D" if use_attn else "UpDecoderBlock2D")
    return {
        "in_channels": int(dd["in_channels"]),
        "out_channels": int(dd["out_ch"]),
        "down_block_types": down_block_types,
        "up_block_types": up_block_types,
        "block_out_channels": [int(dd["ch"] * mult) for mult in ch_mult],
        "layers_per_block": int(dd["num_res_blocks"]),
        "act_fn": "silu",
        "latent_channels": int(dd["z_channels"]),
        "sample_size": resolution,
        "num_vq_embeddings": int(params["n_embed"]),
        "norm_num_groups": 32,
        "vq_embed_dim": int(params["embed_dim"]),
        "scaling_factor": 0.18215,
    }


def _parse_config_from_json(path: Path) -> dict[str, Any]:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if "down_block_types" in cfg:
        return {
            "in_channels": int(cfg["in_channels"]),
            "out_channels": int(cfg.get("out_channels", cfg.get("out_ch", 3))),
            "down_block_types": list(cfg["down_block_types"]),
            "up_block_types": list(cfg["up_block_types"]),
            "block_out_channels": [int(x) for x in cfg["block_out_channels"]],
            "layers_per_block": int(cfg["layers_per_block"]),
            "act_fn": str(cfg.get("act_fn", "silu")),
            "latent_channels": int(cfg.get("latent_channels", cfg.get("z_channels", 3))),
            "sample_size": int(cfg.get("sample_size", cfg.get("resolution", 256))),
            "num_vq_embeddings": int(cfg.get("num_vq_embeddings", cfg.get("n_embed", 8192))),
            "norm_num_groups": int(cfg.get("norm_num_groups", 32)),
            "vq_embed_dim": int(cfg.get("vq_embed_dim", cfg.get("embed_dim", 3))),
            "scaling_factor": float(cfg.get("scaling_factor", 0.18215)),
        }
    ch_mult = [int(x) for x in cfg["ch_mult"]]
    attn_indices = set(int(x) for x in cfg.get("attn_resolutions", []))
    down_block_types = [
        "AttnDownEncoderBlock2D" if i in attn_indices else "DownEncoderBlock2D"
        for i in range(len(ch_mult))
    ]
    up_block_types = []
    for i in range(len(ch_mult)):
        old_i = len(ch_mult) - 1 - i
        up_block_types.append("AttnUpDecoderBlock2D" if old_i in attn_indices else "UpDecoderBlock2D")
    return {
        "in_channels": int(cfg["in_channels"]),
        "out_channels": int(cfg.get("out_channels", cfg.get("out_ch", 3))),
        "down_block_types": down_block_types,
        "up_block_types": up_block_types,
        "block_out_channels": [int(cfg["ch"] * m) for m in ch_mult],
        "layers_per_block": int(cfg["num_res_blocks"]),
        "act_fn": "silu",
        "latent_channels": int(cfg.get("latent_channels", cfg.get("z_channels", 3))),
        "sample_size": int(cfg.get("sample_size", cfg.get("resolution", 256))),
        "num_vq_embeddings": int(cfg.get("num_vq_embeddings", cfg.get("n_embed", 8192))),
        "norm_num_groups": int(cfg.get("norm_num_groups", 32)),
        "vq_embed_dim": int(cfg.get("vq_embed_dim", cfg.get("embed_dim", 3))),
        "scaling_factor": float(cfg.get("scaling_factor", 0.18215)),
    }


def _infer_config_from_state(state: dict[str, torch.Tensor]) -> dict[str, Any]:
    down_levels = sorted(
        {int(m.group(1)) for k in state for m in [re.match(r"encoder\.down\.(\d+)\.", k)] if m}
    )
    if not down_levels:
        raise RuntimeError("Unable to infer VQ config: no encoder.down.* keys found")

    base_ch = int(state["encoder.conv_in.weight"].shape[0])
    block_out_channels = []
    for level in down_levels:
        key = f"encoder.down.{level}.block.0.conv2.weight"
        block_out_channels.append(int(state[key].shape[0]))
    ch_mult = [max(c // base_ch, 1) for c in block_out_channels]

    num_res_blocks = max(
        int(m.group(2))
        for k in state
        for m in [re.match(r"encoder\.down\.(\d+)\.block\.(\d+)\.", k)]
        if m
    ) + 1

    enc_attn_old = {
        int(m.group(1)) for k in state for m in [re.match(r"encoder\.down\.(\d+)\.attn\.", k)] if m
    }
    dec_attn_old = {
        int(m.group(1)) for k in state for m in [re.match(r"decoder\.up\.(\d+)\.attn\.", k)] if m
    }
    n_blocks = len(down_levels)

    down_block_types = [
        "AttnDownEncoderBlock2D" if i in enc_attn_old else "DownEncoderBlock2D"
        for i in range(n_blocks)
    ]
    up_block_types = []
    for new_i in range(n_blocks):
        old_i = n_blocks - 1 - new_i
        up_block_types.append("AttnUpDecoderBlock2D" if old_i in dec_attn_old else "UpDecoderBlock2D")

    quant_embed = state["quantize.embedding.weight"]
    n_embed = int(quant_embed.shape[0])
    embed_dim = int(quant_embed.shape[1])
    z_channels = int(state["quant_conv.weight"].shape[1])
    in_channels = int(state["encoder.conv_in.weight"].shape[1])
    out_channels = int(state["decoder.conv_out.weight"].shape[0])

    return {
        "in_channels": in_channels,
        "out_channels": out_channels,
        "down_block_types": down_block_types,
        "up_block_types": up_block_types,
        "block_out_channels": block_out_channels,
        "layers_per_block": num_res_blocks,
        "act_fn": "silu",
        "latent_channels": z_channels,
        "sample_size": 256,
        "num_vq_embeddings": n_embed,
        "norm_num_groups": 32,
        "vq_embed_dim": embed_dim,
        "scaling_factor": 0.18215,
    }


def _load_vq_config(folder: Path, state: dict[str, torch.Tensor]) -> tuple[dict[str, Any], str]:
    yaml_cfg = folder / "config.yaml"
    json_cfg = folder / "config.json"
    if yaml_cfg.exists():
        return _parse_config_from_yaml(yaml_cfg), str(yaml_cfg)
    if json_cfg.exists():
        return _parse_config_from_json(json_cfg), str(json_cfg)
    return _infer_config_from_state(state), "inferred_from_checkpoint"


def _map_attention_part(part: str) -> str:
    if part == "norm":
        return "group_norm"
    if part == "q":
        return "to_q"
    if part == "k":
        return "to_k"
    if part == "v":
        return "to_v"
    if part == "proj_out":
        return "to_out.0"
    raise ValueError(f"Unknown attention sub-key: {part}")


def _convert_state_dict(state: dict[str, torch.Tensor], n_blocks: int) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        new_key: str | None = None

        if key in {
            "encoder.conv_in.weight",
            "encoder.conv_in.bias",
            "encoder.conv_out.weight",
            "encoder.conv_out.bias",
            "decoder.conv_in.weight",
            "decoder.conv_in.bias",
            "decoder.conv_out.weight",
            "decoder.conv_out.bias",
            "quant_conv.weight",
            "quant_conv.bias",
            "post_quant_conv.weight",
            "post_quant_conv.bias",
            "quantize.embedding.weight",
        }:
            new_key = key
        elif key.startswith("encoder.norm_out."):
            new_key = key.replace("encoder.norm_out.", "encoder.conv_norm_out.")
        elif key.startswith("decoder.norm_out."):
            new_key = key.replace("decoder.norm_out.", "decoder.conv_norm_out.")
        else:
            m = re.match(r"encoder\.down\.(\d+)\.block\.(\d+)\.(.+)$", key)
            if m:
                i, j, rest = int(m.group(1)), int(m.group(2)), m.group(3)
                rest = rest.replace("nin_shortcut", "conv_shortcut")
                new_key = f"encoder.down_blocks.{i}.resnets.{j}.{rest}"

            m = m or re.match(r"encoder\.down\.(\d+)\.downsample\.conv\.(.+)$", key)
            if m and new_key is None:
                i, rest = int(m.group(1)), m.group(2)
                new_key = f"encoder.down_blocks.{i}.downsamplers.0.conv.{rest}"

            m = m or re.match(r"encoder\.down\.(\d+)\.attn\.(\d+)\.(norm|q|k|v|proj_out)\.(.+)$", key)
            if m and new_key is None:
                i, j, part, rest = int(m.group(1)), int(m.group(2)), m.group(3), m.group(4)
                mapped = _map_attention_part(part)
                new_key = f"encoder.down_blocks.{i}.attentions.{j}.{mapped}.{rest}"

            m = m or re.match(r"encoder\.mid\.block_(\d+)\.(.+)$", key)
            if m and new_key is None:
                block_idx = int(m.group(1)) - 1
                rest = m.group(2).replace("nin_shortcut", "conv_shortcut")
                new_key = f"encoder.mid_block.resnets.{block_idx}.{rest}"

            m = m or re.match(r"encoder\.mid\.attn_1\.(norm|q|k|v|proj_out)\.(.+)$", key)
            if m and new_key is None:
                part, rest = m.group(1), m.group(2)
                mapped = _map_attention_part(part)
                new_key = f"encoder.mid_block.attentions.0.{mapped}.{rest}"

            m = m or re.match(r"decoder\.mid\.block_(\d+)\.(.+)$", key)
            if m and new_key is None:
                block_idx = int(m.group(1)) - 1
                rest = m.group(2).replace("nin_shortcut", "conv_shortcut")
                new_key = f"decoder.mid_block.resnets.{block_idx}.{rest}"

            m = m or re.match(r"decoder\.mid\.attn_1\.(norm|q|k|v|proj_out)\.(.+)$", key)
            if m and new_key is None:
                part, rest = m.group(1), m.group(2)
                mapped = _map_attention_part(part)
                new_key = f"decoder.mid_block.attentions.0.{mapped}.{rest}"

            m = m or re.match(r"decoder\.up\.(\d+)\.block\.(\d+)\.(.+)$", key)
            if m and new_key is None:
                old_i, j, rest = int(m.group(1)), int(m.group(2)), m.group(3)
                new_i = n_blocks - 1 - old_i
                rest = rest.replace("nin_shortcut", "conv_shortcut")
                new_key = f"decoder.up_blocks.{new_i}.resnets.{j}.{rest}"

            m = m or re.match(r"decoder\.up\.(\d+)\.upsample\.conv\.(.+)$", key)
            if m and new_key is None:
                old_i, rest = int(m.group(1)), m.group(2)
                new_i = n_blocks - 1 - old_i
                new_key = f"decoder.up_blocks.{new_i}.upsamplers.0.conv.{rest}"

            m = m or re.match(r"decoder\.up\.(\d+)\.attn\.(\d+)\.(norm|q|k|v|proj_out)\.(.+)$", key)
            if m and new_key is None:
                old_i, j, part, rest = int(m.group(1)), int(m.group(2)), m.group(3), m.group(4)
                new_i = n_blocks - 1 - old_i
                mapped = _map_attention_part(part)
                new_key = f"decoder.up_blocks.{new_i}.attentions.{j}.{mapped}.{rest}"

        if new_key is not None:
            # Legacy taming attention uses Conv2d(1x1); Diffusers attention uses Linear.
            if new_key.endswith((".to_q.weight", ".to_k.weight", ".to_v.weight", ".to_out.0.weight")):
                if value.ndim == 4 and value.shape[-2:] == (1, 1):
                    value = value[:, :, 0, 0]
            out[new_key] = value

    return out


def convert_one(raw_dir: Path, out_dir: Path) -> None:
    ckpt_path = _find_checkpoint(raw_dir)
    state = _load_model_state(ckpt_path)
    config, config_source = _load_vq_config(raw_dir, state)
    n_blocks = len(config["block_out_channels"])
    converted = _convert_state_dict(state, n_blocks=n_blocks)

    model = VQModel(**config)
    missing, unexpected = model.load_state_dict(converted, strict=False)
    if missing:
        raise RuntimeError(f"Missing {len(missing)} keys after conversion: {missing[:10]}")
    if unexpected:
        raise RuntimeError(f"Unexpected {len(unexpected)} keys after conversion: {unexpected[:10]}")

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir), safe_serialization=True)

    status = {
        "status": "converted",
        "source_checkpoint": str(ckpt_path),
        "config_source": config_source,
        "output_dir": str(out_dir),
        "class_name": "VQModel",
    }
    with open(out_dir.parent / "vqgan_conversion_status.json", "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)
    print(f"[ok] {raw_dir.name}: {ckpt_path.name} -> {out_dir}")


def _prepare_raw_source(raw_arg: str) -> tuple[Path, str, tempfile.TemporaryDirectory[str] | None]:
    """Return usable raw folder path, source name, and optional temp directory handle."""
    src = Path(raw_arg)
    if src.is_dir():
        return src, src.name, None

    if src.is_file() and src.suffix.lower() == ".zip":
        tmp = tempfile.TemporaryDirectory(prefix="vqgan_raw_")
        with zipfile.ZipFile(src, "r") as zf:
            zf.extractall(tmp.name)
        return Path(tmp.name), src.stem, tmp

    raise FileNotFoundError(f"Raw source must be a folder or .zip file: {src}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert legacy VQGAN checkpoints to Diffusers VQModel")
    parser.add_argument(
        "--raw-dir",
        action="append",
        required=True,
        help="Raw VQGAN folder path or .zip file path",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Root output folder; if unset with --in-place, writes inside each raw folder",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Write converted VQModel under each raw folder (default subfolder: vqvae)",
    )
    parser.add_argument("--subfolder", type=str, default="vqvae", help="Subfolder name for VQModel files")
    args = parser.parse_args()

    if not args.in_place and args.output_root is None:
        raise ValueError("Specify either --in-place or --output-root")

    output_root = Path(args.output_root) if args.output_root else None
    if output_root is not None:
        output_root.mkdir(parents=True, exist_ok=True)

    for raw_dir_str in args.raw_dir:
        raw_dir, source_name, tmp_handle = _prepare_raw_source(raw_dir_str)
        try:
            if args.in_place:
                if tmp_handle is not None:
                    raise ValueError("--in-place is not supported when --raw-dir points to a .zip file")
                out_dir = raw_dir / args.subfolder
            else:
                out_dir = output_root / source_name / args.subfolder
            convert_one(raw_dir=raw_dir, out_dir=out_dir)
        finally:
            if tmp_handle is not None:
                tmp_handle.cleanup()


if __name__ == "__main__":
    main()
