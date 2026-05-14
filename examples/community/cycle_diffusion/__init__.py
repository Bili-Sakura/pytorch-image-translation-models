# Copyright (c) 2026 EarthBridge Team.
# Credits: CycleDiffusion (Wu & De la Torre, ICCV 2023) — https://github.com/humansensinglab/cycle-diffusion

"""CycleDiffusion: zero-shot image editing with stochastic diffusion models (ICCV 2023).

Upstream implementation: https://github.com/humansensinglab/cycle-diffusion

This package provides path resolution and a thin CLI to run upstream ``main.py`` (and other
scripts) from a local clone. Install upstream dependencies (see their ``environment.yml``)
before running training or evaluation.
"""

from examples.community.cycle_diffusion.model import CYCLE_DIFFUSION_REPO_URL
from examples.community.cycle_diffusion.pipeline import inject_cycle_diffusion_sys_path, resolve_cycle_diffusion_root

__all__ = [
    "CYCLE_DIFFUSION_REPO_URL",
    "inject_cycle_diffusion_sys_path",
    "resolve_cycle_diffusion_root",
]
