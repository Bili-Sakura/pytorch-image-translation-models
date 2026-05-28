# Copyright (c) 2026 EarthBridge Team.
# Credits: CycleDiff (Zou et al., TIP 2026) — https://github.com/ZouShilong1024/CycleDiff

"""CycleDiff: cycle diffusion models for unpaired image-to-image translation."""

CYCLEDIFF_REPO_URL = "https://github.com/ZouShilong1024/CycleDiff"
CYCLEDIFF_PAPER_URL = "https://arxiv.org/abs/2508.06625"

from src.models.cyclediff.config_loader import load_cfg_node, load_yaml_config
from src.models.cyclediff.factory import build_all_models, build_latent_diffusion, load_checkpoint_weights
from src.models.cyclediff.inference import is_a2b_task, translate_batch, translate_for_task

__all__ = [
    "CYCLEDIFF_REPO_URL",
    "CYCLEDIFF_PAPER_URL",
    "load_cfg_node",
    "load_yaml_config",
    "build_all_models",
    "build_latent_diffusion",
    "load_checkpoint_weights",
    "is_a2b_task",
    "translate_batch",
    "translate_for_task",
]
