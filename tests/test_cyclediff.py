# Copyright (c) 2026 EarthBridge Team.
# Credits: CycleDiff (Zou et al., TIP 2026) — https://github.com/ZouShilong1024/CycleDiff

"""Tests for vendored CycleDiff integration."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]


def _load_submodule(relpath: str, mod_name: str):
    """Load a module file without importing ``src`` package root."""
    path = _REPO / relpath
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class TestCycleDiffVendored:
    def test_inference_task_direction(self):
        mod = _load_submodule("src/models/cyclediff/inference.py", "cyclediff_inference")
        assert mod.is_a2b_task("cat2dog")
        assert mod.is_a2b_task("horse2zebra")
        assert not mod.is_a2b_task("dog2cat")

    def test_config_yaml_class_names(self):
        cfg_path = _REPO / "examples/cyclediff/configs/afhq_cat2dog/translation_C_disc_timestep_ode_2.yaml"
        text = cfg_path.read_text(encoding="utf-8")
        assert "src.models.cyclediff.ddm" in text
        assert "src.models.cyclediff.unet" in text
        assert "class_name: ddm." not in text

    def test_load_yaml_config(self):
        mod = _load_submodule("src/models/cyclediff/config_loader.py", "cyclediff_config")
        cfg_path = _REPO / "examples/cyclediff/configs/afhq_cat2dog/translation_C_disc_timestep_ode_2.yaml"
        cfg = mod.load_yaml_config(cfg_path)
        assert "model1" in cfg
        assert "trainer" in cfg

    def test_packaged_configs_exist(self):
        root = _REPO / "examples/cyclediff/configs"
        assert (root / "afhq_cat2dog" / "translation_C_disc_timestep_ode_2.yaml").is_file()

    def test_examples_resolve_cfg(self):
        from examples.cyclediff.config import packaged_configs_dir
        from examples.cyclediff.train import _resolve_cfg

        rel = "afhq_cat2dog/translation_C_disc_timestep_ode_2.yaml"
        p = _resolve_cfg(str(packaged_configs_dir() / rel))
        assert p.is_file()

    def test_ddm_utils_import(self):
        mod = _load_submodule("src/models/cyclediff/ddm/utils.py", "cyclediff_ddm_utils")
        assert callable(mod.construct_class_by_name)
        assert callable(mod.safe_torch_load)
