# Copyright (c) 2026 EarthBridge Team.
# Credits: CycleDiff (Zou et al., TIP 2026) — https://github.com/ZouShilong1024/CycleDiff

"""Backward-compatible re-exports; prefer :mod:`src.pipelines.cyclediff`."""

from src.pipelines.cyclediff import inject_cyclediff_sys_path, resolve_cyclediff_root

__all__ = ["inject_cyclediff_sys_path", "resolve_cyclediff_root"]
