#!/usr/bin/env python3
# Copyright (c) 2026 EarthBridge Team.
# Credits: CDTSDE/PSCDE (solar defect identification, projects/CDTSDE).
"""Convert raw CDTSDE/PSCDE .ckpt checkpoints to standard diffusers layout.

Supports ControlLDM checkpoints from projects/CDTSDE (e.g. solar defect PSCDE).
Produces a diffusers-style directory structure:
    unet/
    controlnet/
    vae/
    text_encoder/
    cond_encoder/
    nonlinear_lambda/
    scheduler/

Usage:
    python -m examples.community.cdtsde.convert_ckpt_to_cdtsde \\
        --ckpt /path/to/PSCDE.ckpt \\
        --output-dir /path/to/CDTSDE-ckpt/solar-defect-pscde

    # With model config (optional, for reproducibility)
    python -m examples.community.cdtsde.convert_ckpt_to_cdtsde \\
        --ckpt /root/worksapce/models/raw/CDTSDE-ckpt-raw/PSCDE.ckpt \\
        --output-dir /root/worksapce/models/BiliSakura/CDTSDE-ckpt/solar-defect-pscde \\
        --model-config /root/worksapce/projects/CDTSDE/configs/model/cldm_v21_dynamic.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file


# Component prefixes -> subfolder (diffusers layout)
COMPONENT_PREFIXES = {
    "model.": "unet",
    "control_model.": "controlnet",
    "first_stage_model.": "vae",
    "cond_stage_model.": "text_encoder",
    "cond_encoder.": "cond_encoder",
    "nonlinear_lambda.": "nonlinear_lambda",
}

DIFFUSERS_WEIGHTS_NAME = "diffusion_pytorch_model.safetensors"

# CDTSDE paper & lineage (Wang et al., ICLR 2026; derived from Stable Diffusion)
CDTSDE_META = {
    "paper": "Adaptive Domain Shift in Diffusion Models for Cross-Modality Image Translation",
    "authors": "Wang et al.",
    "venue": "ICLR 2026",
    "arxiv": "https://arxiv.org/html/2601.18623v2",
    "source": "https://laplace.center/CDTSDE/",
    "lineage": "Stable Diffusion 2.1 (Stability-AI)",
    "backbone": "Latent diffusion (Rombach et al.) + ControlNet",
}

# Per-component config augmentation (diffusers + robust metadata)
COMPONENT_CONFIGS = {
    "unet": {
        "_class_name": "CDTSDE.unet",
        "_diffusers_version": "1.0",
        "model_type": "ControlledUnetModel",
        "architecture": "Stable-Diffusion UNet (Rombach et al. 2022) with ControlNet conditioning",
        "in_channels": 4,
        "out_channels": 4,
        "model_channels": 320,
        "channel_mult": [1, 2, 4, 4],
        "attention_resolutions": [4, 2, 1],
        "context_dim": 1024,
        "latent_channels": 4,
        "latent_size": 32,
    },
    "controlnet": {
        "_class_name": "CDTSDE.controlnet",
        "_diffusers_version": "1.0",
        "model_type": "ControlNet",
        "architecture": "ControlNet for source-image conditioning",
        "in_channels": 4,
        "hint_channels": 4,
        "model_channels": 320,
    },
    "vae": {
        "_class_name": "CDTSDE.vae",
        "_diffusers_version": "1.0",
        "model_type": "AutoencoderKL",
        "architecture": "Stable Diffusion KL-autoencoder (4ch latent, 32× downscale)",
        "in_channels": 3,
        "out_channels": 3,
        "latent_channels": 4,
        "scale_factor": 0.18215,
    },
    "text_encoder": {
        "_class_name": "CDTSDE.text-encoder",
        "_diffusers_version": "1.0",
        "model_type": "FrozenOpenCLIPEmbedder",
        "architecture": "OpenCLIP ViT-H-14 (laion2b_s32b_b79k)",
        "context_dim": 1024,
    },
    "cond_encoder": {
        "_class_name": "CDTSDE.cond-encoder",
        "_diffusers_version": "1.0",
        "model_type": "VAE encoder copy",
        "architecture": "Source-image encoder (frozen VAE encoder + quant_conv) for latent conditioning",
    },
    "nonlinear_lambda": {
        "_class_name": "CDTSDE.nonlinear-lambda",
        "_diffusers_version": "1.0",
        "model_type": "LearnableNonlinearLambdaEnhanced",
        "architecture": "Adaptive domain-shift schedule (spatial + channel modulation)",
        "channels": 4,
        "degree": 5,
    },
}


def _load_state_dict(ckpt_path: Path) -> dict:
    """Load and extract model state dict from .ckpt checkpoint."""
    raw = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    state = raw
    for key in ("state_dict", "model"):
        if key in state and isinstance(state[key], dict):
            state = state[key]
            break

    if not isinstance(state, dict):
        raise TypeError(
            f"Could not resolve state_dict from {ckpt_path}. "
            "Expected a dict with 'state_dict' or 'model' key."
        )

    # Drop preprocess_model keys (not in this codebase)
    filtered = {
        k: v.detach().cpu().contiguous()
        for k, v in state.items()
        if torch.is_tensor(v) and not k.startswith("preprocess_model.")
    }
    dropped = len(state) - len(filtered)
    if dropped:
        print(f"[INFO] Dropped {dropped} preprocess_model keys")

    if not filtered:
        raise RuntimeError(
            f"No model weights found in {ckpt_path}. "
            "Expected keys like model.diffusion_model.*, control_model.*, etc."
        )

    return filtered


def _require_keys(state: dict, prefixes: tuple[str, ...]) -> None:
    """Verify at least one key exists for each required component."""
    for prefix in prefixes:
        found = any(k.startswith(prefix) for k in state)
        if not found:
            raise RuntimeError(
                f"Missing component '{prefix}*' in checkpoint. "
                f"Expected ControlLDM layout (model, control_model, first_stage_model, etc.)."
            )


def _load_model_config(config_path: Path | None) -> dict | None:
    """Load model config YAML as dict (for JSON serialization)."""
    if config_path is None or not config_path.exists():
        return None
    try:
        import yaml
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cfg
    except ImportError:
        print("[WARN] PyYAML not installed, skipping model_config copy")
        return None


def convert_ckpt_to_cdtsde(
    ckpt_path: str | Path,
    output_dir: str | Path,
    *,
    model_config_path: str | Path | None = None,
    use_safetensors: bool = True,
) -> None:
    """Convert raw CDTSDE .ckpt to BiliSakura layout for community pipeline.

    Parameters
    ----------
    ckpt_path : str | Path
        Path to raw .ckpt file (PyTorch Lightning format).
    output_dir : str | Path
        Output directory (e.g. models/BiliSakura/CDTSDE-ckpt/solar-defect-pscde).
    model_config_path : str | Path | None
        Optional path to model config YAML (cldm_v21_dynamic.yaml).
    use_safetensors : bool
        If True, save as model.safetensors; else save as model.ckpt.
    """
    ckpt_path = Path(ckpt_path)
    output_dir = Path(output_dir)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    state_dict = _load_state_dict(ckpt_path)

    # Verify ControlLDM layout (model=DiffusionWrapper.diffusion_model, control_model, first_stage_model)
    _require_keys(
        state_dict,
        (
            "model.",
            "control_model.",
            "first_stage_model.",
        ),
    )

    # Split by component and save in diffusers layout (unet/, controlnet/, vae/, etc.)
    component_state: dict[str, dict] = {subfolder: {} for subfolder in COMPONENT_PREFIXES.values()}
    for key, value in state_dict.items():
        assigned = False
        for prefix, subfolder in COMPONENT_PREFIXES.items():
            if key.startswith(prefix):
                component_state[subfolder][key] = value
                assigned = True
                break
        if not assigned:
            # Buffers / params without prefix (e.g. betas, scale_factor) -> unet
            component_state["unet"][key] = value

    for subfolder, comp_state in component_state.items():
        if not comp_state:
            continue
        sub_dir = output_dir / subfolder
        sub_dir.mkdir(parents=True, exist_ok=True)

        if use_safetensors:
            weights_path = sub_dir / DIFFUSERS_WEIGHTS_NAME
            save_file(comp_state, str(weights_path))
        else:
            weights_path = sub_dir / "model.ckpt"
            torch.save({"state_dict": comp_state}, str(weights_path))

        # Robust config.json (diffusers + CDTSDE paper/lineage)
        config = dict(COMPONENT_CONFIGS.get(subfolder, {}))
        config["_converted_from"] = "ControlLDM"
        config["_cdtsde"] = CDTSDE_META
        with open(sub_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        print(f"[INFO] Saved {subfolder}/ ({len(comp_state)} keys)")

    # Scheduler config (CDTSDE dynamic schedule params)
    sched_dir = output_dir / "scheduler"
    sched_dir.mkdir(exist_ok=True)
    sched_config = {
        "linear_start": 0.00085,
        "linear_end": 0.0120,
        "num_train_timesteps": 1000,
        "schedule": "linear",
        "_cdtsde": CDTSDE_META,
        "scheduler_type": "CDTSDE dynamic shift (VP diffusion)",
    }
    with open(sched_dir / "scheduler_config.json", "w", encoding="utf-8") as f:
        json.dump(sched_config, f, indent=2)
    print(f"[INFO] Saved scheduler/scheduler_config.json")

    # Root pipeline config (robust metadata for loaders)
    pipeline_config = {
        "pipeline_class": "CDTSDECommunityPipeline",
        "_cdtsde": CDTSDE_META,
        "config_path": "configs/model/cldm_v21_dynamic.yaml",
        "shifting_sequence": "dynamic",
        "task": "Electroluminescence → Semantic mask (PSCDE solar defect)",
    }
    with open(output_dir / "pipeline_config.json", "w", encoding="utf-8") as f:
        json.dump(pipeline_config, f, indent=2)
    print(f"[INFO] Saved pipeline_config.json")

    # Model config (for reproducibility)
    model_cfg = _load_model_config(
        Path(model_config_path) if model_config_path else None
    )
    if model_cfg is not None:
        config_path = output_dir / "model_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(model_cfg, f, indent=2, default=str)
        print(f"[INFO] Saved model_config.json")
    else:
        default_config = {
            "target": "model.cldm.ControlLDM",
            "config_path": "configs/model/cldm_v21_dynamic.yaml",
            "shifting_sequence": "dynamic",
        }
        with open(output_dir / "model_config.json", "w", encoding="utf-8") as f:
            json.dump(default_config, f, indent=2)
        print(f"[INFO] Saved model_config.json (default)")

    print(f"[INFO] Converted -> {output_dir}/ (diffusers layout)")
    print(f"[INFO] Use: load_cdtsde_community_pipeline('{output_dir}')")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert CDTSDE/PSCDE .ckpt to standard diffusers layout"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to raw .ckpt file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory (creates unet/, controlnet/, vae/, etc.)",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default=None,
        help="Path to model config YAML (cldm_v21_dynamic.yaml)",
    )
    parser.add_argument(
        "--no-safetensors",
        action="store_true",
        help="Save as .ckpt instead of .safetensors",
    )
    args = parser.parse_args()

    convert_ckpt_to_cdtsde(
        args.ckpt,
        args.output_dir,
        model_config_path=args.model_config,
        use_safetensors=not args.no_safetensors,
    )


if __name__ == "__main__":
    main()
