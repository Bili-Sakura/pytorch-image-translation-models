# Copyright (c) 2026 EarthBridge Team.
# Credits: CycleDiff (Zou et al., TIP 2026) — https://github.com/ZouShilong1024/CycleDiff

"""CycleDiff community shim — implementation lives in :mod:`src.pipelines.cyclediff`.

Prefer::

    from src.pipelines.cyclediff import CycleDiffPipeline, load_cyclediff_pipeline
    python -m examples.cyclediff.train train --cfg ./configs/.../translation.yaml
"""

from src.models.cyclediff import CYCLEDIFF_REPO_URL
from src.pipelines.cyclediff import (
    CycleDiffPipeline,
    CycleDiffPipelineOutput,
    inject_cyclediff_sys_path,
    load_cyclediff_pipeline,
    resolve_cyclediff_root,
    run_cyclediff_script,
)

__all__ = [
    "CYCLEDIFF_REPO_URL",
    "CycleDiffPipeline",
    "CycleDiffPipelineOutput",
    "inject_cyclediff_sys_path",
    "load_cyclediff_pipeline",
    "resolve_cyclediff_root",
    "run_cyclediff_script",
]
