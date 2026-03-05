# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Task-level configuration for I2SB training and inference."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TaskConfig:
    """Configuration for an I2SB image-translation task.

    Default values provide a reasonable baseline and can be overridden
    by the per-task builder functions below.
    """

    # --- task identity ---
    task_name: str = "sar2eo"

    # --- scheduler ---
    interval: int = 1000
    beta_max: float = 0.3

    # --- model ---
    condition_mode: str = "concat"
    resolution: int = 256
    in_channels: int = 3
    num_channels: int = 128
    num_res_blocks: int = 2
    attention_resolutions: str = "32,16,8"

    # --- training ---
    train_batch_size: int = 8
    learning_rate: float = 1e-4
    num_train_epochs: int = 100
    use_ema: bool = True

    # --- hub / logging ---
    push_to_hub: bool = True

    # --- latent VAE (for latent-space variants) ---
    latent_vae_path: str | None = None

    # --- representation alignment (REPA) ---
    rep_alignment_model_path: str | None = None

    # --- extra ---
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Per-task builder helpers
# ---------------------------------------------------------------------------

def _build_config(defaults: dict, overrides: dict) -> TaskConfig:
    merged = {**defaults, **overrides}
    return TaskConfig(**merged)


_COMMON_LATENT_VAE = "./models/stabilityai/sd-vae-ft-ema"


def sar2eo_config(**kwargs) -> TaskConfig:
    """SAR → Electro-Optical translation defaults."""
    return _build_config(
        {"task_name": "sar2eo", "latent_vae_path": _COMMON_LATENT_VAE},
        kwargs,
    )


def rgb2ir_config(**kwargs) -> TaskConfig:
    """RGB → Infrared translation defaults."""
    return _build_config(
        {"task_name": "rgb2ir", "latent_vae_path": _COMMON_LATENT_VAE},
        kwargs,
    )


def sar2ir_config(**kwargs) -> TaskConfig:
    """SAR → Infrared translation defaults."""
    return _build_config(
        {"task_name": "sar2ir", "latent_vae_path": _COMMON_LATENT_VAE},
        kwargs,
    )


def sar2rgb_config(**kwargs) -> TaskConfig:
    """SAR → RGB translation defaults (includes REPA alignment)."""
    return _build_config(
        {
            "task_name": "sar2rgb",
            "latent_vae_path": _COMMON_LATENT_VAE,
            "rep_alignment_model_path": "./models/BiliSakura/MaRS-Base-RGB",
        },
        kwargs,
    )
