# Copyright (c) 2026 EarthBridge Team.
# Credits: NEGCUT from Wang et al. ICCV 2021 — https://github.com/WeilunWang/NEGCUT

"""NEGCUT inference pipeline loader."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import torch

from src.pipelines.cut import NEGCUTPipeline, NEGCUTPipelineOutput


def load_negcut_pipeline(
    checkpoint_dir: Union[str, Path],
    *,
    subfolder: str = "generator",
    device: str | torch.device = "cpu",
    torch_dtype: torch.dtype | None = None,
) -> NEGCUTPipeline:
    """Load a NEGCUT pipeline from a training checkpoint directory.

    Checkpoints produced by :class:`~examples.negcut.NEGCUTTrainer` store the
    generator under ``generator/`` in diffusers layout. Inference is identical
    to CUT — a single generator forward pass.

    Parameters
    ----------
    checkpoint_dir : str or Path
        Path to ``checkpoint-epoch-N`` or a directory containing ``generator/``.
    subfolder : str
        Generator subfolder name (default ``generator``).
    device : str or torch.device
        Target device.
    torch_dtype : torch.dtype, optional
        Optional dtype for the generator weights.

    Returns
    -------
    NEGCUTPipeline
        Ready-to-use inference pipeline.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    if not (checkpoint_dir / subfolder).exists():
        raise FileNotFoundError(
            f"Generator checkpoint not found: {checkpoint_dir / subfolder}"
        )

    return NEGCUTPipeline.from_pretrained(
        checkpoint_dir,
        subfolder=subfolder,
        device=device,
        torch_dtype=torch_dtype,
    )


__all__ = ["NEGCUTPipeline", "NEGCUTPipelineOutput", "load_negcut_pipeline"]
