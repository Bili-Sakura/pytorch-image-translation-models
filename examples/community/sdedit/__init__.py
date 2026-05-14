# Copyright (c) 2026 EarthBridge Team.
# Credits: SDEdit (Meng et al., ICLR 2022) — https://github.com/ermongroup/SDEdit

"""SDEdit: guided image synthesis and editing with stochastic differential equations (ICLR 2022).

Upstream implementation: https://github.com/ermongroup/SDEdit

This package provides path resolution and a thin CLI to run upstream ``main.py`` from a local
clone. Install upstream dependencies (see their ``requirements.txt``) before sampling.
"""

from examples.community.sdedit.model import SDEDIT_REPO_URL
from examples.community.sdedit.pipeline import inject_sdedit_sys_path, resolve_sdedit_root

__all__ = [
    "SDEDIT_REPO_URL",
    "inject_sdedit_sys_path",
    "resolve_sdedit_root",
]
