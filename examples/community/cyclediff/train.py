# Copyright (c) 2026 EarthBridge Team.
# Credits: CycleDiff (Zou et al., TIP 2026) — https://github.com/ZouShilong1024/CycleDiff

"""Run CycleDiff upstream training or translation scripts from a local checkout.

Example
-------
.. code-block:: bash

    python -m examples.community.cyclediff.train \\
        --cyclediff-root /path/to/CycleDiff \\
        train_uncond_ldm_cycle.py --cfg ./configs/dataset/translation_C_disc_timestep_ode_2.yaml
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from examples.community.cyclediff.pipeline import resolve_cyclediff_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Execute a CycleDiff script (train_vae.py, train_uncond_ldm_cycle.py, etc.) "
        "with working directory set to the CycleDiff root."
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

    script_name = rest[0]
    script_args = rest[1:]
    root = resolve_cyclediff_root(args.cyclediff_root)
    script_path = root / script_name
    if not script_path.is_file():
        raise FileNotFoundError(f"Script not found under CycleDiff root: {script_path}")

    cmd = [sys.executable, str(script_path), *script_args]
    return subprocess.call(cmd, cwd=str(root))


if __name__ == "__main__":
    raise SystemExit(main())
