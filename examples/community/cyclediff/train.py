# Copyright (c) 2026 EarthBridge Team.
# Credits: CycleDiff (Zou et al., TIP 2026) — https://github.com/ZouShilong1024/CycleDiff

"""Backward-compatible CLI; prefer :mod:`examples.cyclediff.train` or :mod:`src.pipelines.cyclediff`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.pipelines.cyclediff import run_cyclediff_script


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Execute a CycleDiff script (deprecated: use python -m examples.cyclediff.train)."
    )
    parser.add_argument(
        "--cyclediff-root",
        type=Path,
        default=None,
        help="Path to CycleDiff clone (overrides CYCLEDIFF_ROOT and auto-discovery).",
    )
    parser.add_argument(
        "script_and_args",
        nargs=argparse.REMAINDER,
        help="Script name and arguments, e.g. train_uncond_ldm_cycle.py --cfg ./configs/....yaml",
    )
    args = parser.parse_args(argv)

    rest = args.script_and_args
    if rest and rest[0] == "--":
        rest = rest[1:]
    if not rest:
        parser.error(
            "Pass the upstream script and its arguments after optional --cyclediff-root, "
            "e.g. train_uncond_ldm_cycle.py --cfg ./configs/foo/translation_C_disc_timestep_ode_2.yaml"
        )

    return run_cyclediff_script(rest, cyclediff_root=args.cyclediff_root)


if __name__ == "__main__":
    raise SystemExit(main())
