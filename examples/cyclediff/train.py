# Copyright (c) 2026 EarthBridge Team.
# Credits: CycleDiff (Zou et al., TIP 2026) — https://github.com/ZouShilong1024/CycleDiff

"""Run CycleDiff upstream training or translation from ``examples/cyclediff``.

Training and inference use the upstream YAML configs and scripts in a local
CycleDiff clone. Set ``CYCLEDIFF_ROOT`` or pass ``--cyclediff-root``.

Examples
--------
.. code-block:: bash

    # Cycle LDM training (main CycleDiff recipe)
    python -m examples.cyclediff.train train --cfg ./configs/cat2dog/translation_C_disc_timestep_ode_2.yaml

    # Translation / evaluation
    python -m examples.cyclediff.train translate --cfg ./configs/cat2dog/test_translation.yaml

    # Arbitrary upstream script
    python -m examples.cyclediff.train run train_vae.py --cfg ./configs/cat2dog/vae.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.pipelines.cyclediff import (
    CYCLEDIFF_LDM_TRAIN_SCRIPT,
    CYCLEDIFF_TRAIN_SCRIPT,
    CYCLEDIFF_TRANSLATION_SCRIPT,
    CYCLEDIFF_VAE_TRAIN_SCRIPT,
    CycleDiffPipeline,
    run_cyclediff_script,
)


def _add_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cyclediff-root",
        type=Path,
        default=None,
        help="Path to CycleDiff clone (overrides CYCLEDIFF_ROOT and auto-discovery).",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="CycleDiff: run upstream training or translation via src.pipelines.cyclediff."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help=f"Run {CYCLEDIFF_TRAIN_SCRIPT}")
    _add_root(p_train)
    p_train.add_argument("--cfg", required=True, help="YAML config path (relative to CycleDiff root).")
    p_train.add_argument("extra", nargs=argparse.REMAINDER, help="Additional args passed to upstream script.")

    p_translate = sub.add_parser("translate", help=f"Run {CYCLEDIFF_TRANSLATION_SCRIPT}")
    _add_root(p_translate)
    p_translate.add_argument("--cfg", required=True, help="YAML config path (relative to CycleDiff root).")
    p_translate.add_argument("extra", nargs=argparse.REMAINDER, help="Additional upstream args.")

    p_vae = sub.add_parser("train-vae", help=f"Run {CYCLEDIFF_VAE_TRAIN_SCRIPT}")
    _add_root(p_vae)
    p_vae.add_argument("--cfg", required=True, help="VAE training YAML config.")
    p_vae.add_argument("extra", nargs=argparse.REMAINDER)

    p_ldm = sub.add_parser("train-ldm", help=f"Run {CYCLEDIFF_LDM_TRAIN_SCRIPT}")
    _add_root(p_ldm)
    p_ldm.add_argument("--cfg", required=True, help="Single-domain LDM YAML config.")
    p_ldm.add_argument("extra", nargs=argparse.REMAINDER)

    p_run = sub.add_parser("run", help="Run any script from the CycleDiff root")
    _add_root(p_run)
    p_run.add_argument(
        "script_and_args",
        nargs=argparse.REMAINDER,
        help="Script name and args, e.g. train_uncond_ldm_cycle.py --cfg ./configs/....yaml",
    )

    args = parser.parse_args(argv)
    root = args.cyclediff_root
    extra = [a for a in getattr(args, "extra", []) if a != "--"]

    if args.command == "train":
        pipe = CycleDiffPipeline.from_pretrained(cyclediff_root=root)
        return pipe.run_training(args.cfg, extra_args=extra or None)

    if args.command == "translate":
        pipe = CycleDiffPipeline.from_pretrained(cyclediff_root=root)
        return pipe.run_translation(args.cfg, extra_args=extra or None)

    if args.command == "train-vae":
        pipe = CycleDiffPipeline.from_pretrained(cyclediff_root=root)
        return pipe.run_vae_training(args.cfg, extra_args=extra or None)

    if args.command == "train-ldm":
        pipe = CycleDiffPipeline.from_pretrained(cyclediff_root=root)
        return pipe.run_ldm_training(args.cfg, extra_args=extra or None)

    if args.command == "run":
        rest = args.script_and_args
        if rest and rest[0] == "--":
            rest = rest[1:]
        if not rest:
            p_run.error("Pass script name and arguments, e.g. train_vae.py --cfg ./configs/foo/vae.yaml")
        return run_cyclediff_script(rest, cyclediff_root=root)

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
