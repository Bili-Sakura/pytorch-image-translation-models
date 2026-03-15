"""Checkpoint management utilities for training."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def rotate_checkpoints(output_dir: str | Path, total_limit: int = 1) -> None:
    """Keep only the most recent checkpoints, deleting older ones.

    Expects checkpoint subdirs named ``checkpoint-epoch-N`` or ``checkpoint-N``
    (step-based). Sorts by embedded number and removes oldest when over limit.

    Parameters
    ----------
    output_dir : str | Path
        Directory containing checkpoint subdirs.
    total_limit : int
        Maximum number of checkpoints to retain (default 1).
    """
    if total_limit <= 0:
        return

    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        return

    def _parse_ckpt_num(p: Path) -> tuple[int, bool]:
        """Return (order_key, is_valid). Lower order_key = older."""
        name = p.name
        if name.startswith("checkpoint-epoch-"):
            try:
                return (int(name.replace("checkpoint-epoch-", "")), True)
            except ValueError:
                return (0, False)
        if name.startswith("checkpoint-"):
            try:
                return (int(name.replace("checkpoint-", "").split("-")[0]), True)
            except (ValueError, IndexError):
                return (0, False)
        return (0, False)

    ckpt_dirs = [p for p in output_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")]
    valid = [(p, _parse_ckpt_num(p)) for p in ckpt_dirs]
    valid = [(p, k) for p, (k, ok) in valid if ok]
    valid.sort(key=lambda x: x[1])

    while len(valid) > total_limit:
        oldest = valid.pop(0)
        oldest_path = oldest[0]
        try:
            shutil.rmtree(oldest_path)
            logger.info("Rotated checkpoint: removed %s", oldest_path)
        except OSError as e:
            logger.warning("Failed to remove %s: %s", oldest_path, e)
