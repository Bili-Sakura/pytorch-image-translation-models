# Copyright (c) 2026 EarthBridge Team.
# Credits: CycleDiff (Zou et al., TIP 2026) — https://github.com/ZouShilong1024/CycleDiff

"""YAML configuration loading for CycleDiff."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

import yaml
from fvcore.common.config import CfgNode


def load_yaml_config(config_file: Union[str, Path], conf: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Load a CycleDiff YAML config into a nested dict."""
    conf = {} if conf is None else dict(conf)
    path = Path(config_file)
    with path.open(encoding="utf-8") as f:
        exp_conf = yaml.load(f, Loader=yaml.FullLoader)
    for key, value in exp_conf.items():
        conf[key] = value
    return conf


def load_cfg_node(config_file: Union[str, Path]) -> CfgNode:
    """Load YAML and wrap as :class:`fvcore.common.config.CfgNode`."""
    return CfgNode(load_yaml_config(config_file))


def default_configs_dir() -> Path:
    """Directory shipped with examples (``examples/cyclediff/configs``)."""
    return Path(__file__).resolve().parents[3] / "examples" / "cyclediff" / "configs"
