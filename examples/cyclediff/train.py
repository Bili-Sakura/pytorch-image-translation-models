# Copyright (c) 2026 EarthBridge Team.
# Credits: CycleDiff (Zou et al., TIP 2026) — https://github.com/ZouShilong1024/CycleDiff

"""CycleDiff training and batch translation CLI.

Examples
--------
.. code-block:: bash

    python -m examples.cyclediff.train train \\
        --cfg examples/cyclediff/configs/afhq_cat2dog/translation_C_disc_timestep_ode_2.yaml

    python -m examples.cyclediff.train translate \\
        --cfg examples/cyclediff/configs/afhq_cat2dog/translation_C_disc_timestep_ode_2.yaml

    python -m examples.cyclediff.train train-vae \\
        --cfg examples/cyclediff/configs/afhq_cat2dog/cat_ae_kl_256x256_d4.yaml
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

from examples.cyclediff.config import packaged_configs_dir


def _resolve_cfg(path: str) -> Path:
    p = Path(path)
    if p.is_file():
        return p.resolve()
    alt = packaged_configs_dir() / path
    if alt.is_file():
        return alt.resolve()
    raise FileNotFoundError(f"Config not found: {path} (also tried {alt})")


def _run_script(module_name: str, cfg_path: Path, extra: list[str]) -> int:
    argv = [str(cfg_path)] + extra
    sys.argv = [module_name, "--cfg", str(cfg_path)] + extra
    runpy.run_module(module_name, run_name="__main__")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CycleDiff training and translation (vendored in src.models.cyclediff).")
    sub = parser.add_subparsers(dest="command", required=True)

    for name, help_text, mod in [
        ("train", "Cycle LDM + cycle GAN training", "src.models.cyclediff.scripts.train_uncond_ldm_cycle"),
        ("translate", "Batch translation on a test dataset", "src.models.cyclediff.scripts.translation_uncond_ldm_cycle"),
        ("train-vae", "VAE pretraining", "src.models.cyclediff.scripts.train_vae"),
        ("train-ldm", "Single-domain LDM pretraining", "src.models.cyclediff.scripts.train_uncond_ldm"),
    ]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--cfg", required=True, help="YAML config path")
        p.add_argument("extra", nargs=argparse.REMAINDER, help="Extra args passed to the script")

    args = parser.parse_args(argv)
    cfg_path = _resolve_cfg(args.cfg)
    extra = [a for a in args.extra if a != "--"]

    mod_map = {
        "train": "src.models.cyclediff.scripts.train_uncond_ldm_cycle",
        "translate": "src.models.cyclediff.scripts.translation_uncond_ldm_cycle",
        "train-vae": "src.models.cyclediff.scripts.train_vae",
        "train-ldm": "src.models.cyclediff.scripts.train_uncond_ldm",
    }
    return _run_script(mod_map[args.command], cfg_path, extra)


if __name__ == "__main__":
    raise SystemExit(main())
