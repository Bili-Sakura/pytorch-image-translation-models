# Copyright (c) 2026 EarthBridge Team.
# Credits: SDEdit (Meng et al., ICLR 2022) — https://github.com/ermongroup/SDEdit

"""Run upstream SDEdit entry points from a local checkout.

Example
-------
.. code-block:: bash

    python -m examples.community.sdedit.train \\
        --sdedit-root /path/to/SDEdit \\
        main.py --exp ./runs/ --config bedroom.yml --sample -i images \\
        --npy_name lsun_bedroom1 --sample_step 3 --t 500 --ni
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from examples.community.sdedit.pipeline import resolve_sdedit_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Execute an SDEdit script (typically main.py) with working directory "
        "set to the repository root."
    )
    parser.add_argument(
        "--sdedit-root",
        type=Path,
        default=None,
        help="Path to SDEdit clone (overrides SDEDIT_ROOT and auto-discovery).",
    )
    parser.add_argument(
        "script_and_args",
        nargs=argparse.REMAINDER,
        help="Script name and arguments, e.g. main.py --exp ./runs/ --config bedroom.yml ...",
    )
    args = parser.parse_args(argv)

    rest = args.script_and_args
    if rest and rest[0] == "--":
        rest = rest[1:]
    if not rest:
        parser.error(
            "Pass the upstream script and its arguments after optional --sdedit-root, "
            "e.g. main.py --exp ./runs/ --config bedroom.yml --sample -i images --npy_name demo --ni"
        )

    script_name = rest[0]
    script_args = rest[1:]
    root = resolve_sdedit_root(args.sdedit_root)
    script_path = root / script_name
    if not script_path.is_file():
        raise FileNotFoundError(f"Script not found under SDEdit root: {script_path}")

    cmd = [sys.executable, str(script_path), *script_args]
    return subprocess.call(cmd, cwd=str(root))


if __name__ == "__main__":
    raise SystemExit(main())
