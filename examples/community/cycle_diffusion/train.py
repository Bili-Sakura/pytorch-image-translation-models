# Copyright (c) 2026 EarthBridge Team.
# Credits: CycleDiffusion (Wu & De la Torre, ICCV 2023) — https://github.com/humansensinglab/cycle-diffusion

"""Run upstream cycle-diffusion entry points from a local checkout.

Example
-------
.. code-block:: bash

    python -m examples.community.cycle_diffusion.train \\
        --cycle-diffusion-root /path/to/cycle-diffusion \\
        main.py --cfg config/experiments/translate_text2img256_stable_diffusion_stochastic_1.cfg \\
        --run_name demo --do_eval --output_dir output/demo
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from examples.community.cycle_diffusion.pipeline import resolve_cycle_diffusion_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Execute a cycle-diffusion script (typically main.py) with working directory "
        "set to the repository root."
    )
    parser.add_argument(
        "--cycle-diffusion-root",
        type=Path,
        default=None,
        help="Path to cycle-diffusion clone (overrides CYCLE_DIFFUSION_ROOT and auto-discovery).",
    )
    parser.add_argument(
        "script_and_args",
        nargs=argparse.REMAINDER,
        help="Script name and arguments, e.g. main.py --cfg config/experiments/....cfg ...",
    )
    args = parser.parse_args(argv)

    rest = args.script_and_args
    if rest and rest[0] == "--":
        rest = rest[1:]
    if not rest:
        parser.error(
            "Pass the upstream script and its arguments after optional --cycle-diffusion-root, "
            "e.g. main.py --cfg config/experiments/translate_text2img256_stable_diffusion_stochastic_1.cfg"
        )

    script_name = rest[0]
    script_args = rest[1:]
    root = resolve_cycle_diffusion_root(args.cycle_diffusion_root)
    script_path = root / script_name
    if not script_path.is_file():
        raise FileNotFoundError(f"Script not found under cycle-diffusion root: {script_path}")

    cmd = [sys.executable, str(script_path), *script_args]
    return subprocess.call(cmd, cwd=str(root))


if __name__ == "__main__":
    raise SystemExit(main())
