# Copyright (c) 2026 EarthBridge Team.
# Credits: CycleDiff (Zou et al., TIP 2026) — https://github.com/ZouShilong1024/CycleDiff

"""CycleDiff model identifiers and upstream repository metadata."""

CYCLEDIFF_REPO_URL = "https://github.com/ZouShilong1024/CycleDiff"

# Scripts expected at the root of a valid CycleDiff checkout.
CYCLEDIFF_TRAIN_SCRIPT = "train_uncond_ldm_cycle.py"
CYCLEDIFF_TRANSLATION_SCRIPT = "translation_uncond_ldm_cycle.py"
CYCLEDIFF_VAE_TRAIN_SCRIPT = "train_vae.py"
CYCLEDIFF_LDM_TRAIN_SCRIPT = "train_uncond_ldm.py"

__all__ = [
    "CYCLEDIFF_REPO_URL",
    "CYCLEDIFF_TRAIN_SCRIPT",
    "CYCLEDIFF_TRANSLATION_SCRIPT",
    "CYCLEDIFF_VAE_TRAIN_SCRIPT",
    "CYCLEDIFF_LDM_TRAIN_SCRIPT",
]
