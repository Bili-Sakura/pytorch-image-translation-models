# Copyright (c) 2026 EarthBridge Team.
# Credits: CycleDiff (Zou et al., TIP 2026) — https://github.com/ZouShilong1024/CycleDiff

"""CycleDiff: cycle diffusion models for unpaired image-to-image translation.

Upstream implementation: https://github.com/ZouShilong1024/CycleDiff

This package provides path resolution and a thin CLI to run upstream training
and evaluation scripts from a local clone. Install CycleDiff dependencies in
your environment (see their ``requirement.txt``) before running training.
"""

from examples.community.cyclediff.model import CYCLEDIFF_REPO_URL
from examples.community.cyclediff.pipeline import inject_cyclediff_sys_path, resolve_cyclediff_root

__all__ = [
    "CYCLEDIFF_REPO_URL",
    "inject_cyclediff_sys_path",
    "resolve_cyclediff_root",
]
