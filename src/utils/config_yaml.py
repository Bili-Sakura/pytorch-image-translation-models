# Copyright (c) 2026 EarthBridge Team.
"""Utility to save training configuration to YAML for checkpoint reproducibility."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def _to_yaml_serializable(obj: Any) -> Any:
    """Convert config object to YAML-serializable format."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, tuple):
        return [_to_yaml_serializable(x) for x in obj]
    if isinstance(obj, (list,)):
        return [_to_yaml_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_yaml_serializable(v) for k, v in obj.items()}
    if is_dataclass(obj) and not isinstance(obj, type):
        return _to_yaml_serializable(asdict(obj))
    if hasattr(obj, "__dict__"):
        return _to_yaml_serializable(vars(obj))
    return obj


def save_config_yaml(config: Any, path: str | Path, *, extra: dict[str, Any] | None = None) -> None:
    """Save training configuration to a YAML file for checkpoint reproducibility.

    Parameters
    ----------
    config : object
        Configuration object (dataclass, dict, or object with __dict__).
    path : str | Path
        Output file path (e.g. checkpoint_dir / "config.yaml").
    extra : dict, optional
        Additional key-value pairs to merge into the output (e.g. epoch, global_step).
    """
    try:
        import yaml
    except ImportError as err:
        raise ImportError(
            "PyYAML is required to save config.yaml. Install with: pip install pyyaml"
        ) from err

    data = _to_yaml_serializable(config)
    if not isinstance(data, dict):
        data = {"config": data}
    if extra:
        data = {**data, **extra}

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
