# Copyright (c) 2026 EarthBridge Team.
# Credits: Hneg-SRC from Jung et al. CVPR 2022 — https://github.com/jcy132/Hneg_SRC

"""Hneg-SRC inference pipeline loader."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import torch

from src.pipelines.cut import HnegSRCPipeline, HnegSRCPipelineOutput


def load_hneg_src_pipeline(
    checkpoint_dir: Union[str, Path],
    *,
    subfolder: str = "generator",
    device: str | torch.device = "cpu",
    torch_dtype: torch.dtype | None = None,
) -> HnegSRCPipeline:
    """Load an Hneg-SRC pipeline from a training checkpoint directory.

    Checkpoints produced by :class:`~examples.hneg_src.HnegSRCTrainer` store the
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
    HnegSRCPipeline
        Ready-to-use inference pipeline.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    if not (checkpoint_dir / subfolder).exists():
        raise FileNotFoundError(
            f"Generator checkpoint not found: {checkpoint_dir / subfolder}"
        )

    return HnegSRCPipeline.from_pretrained(
        checkpoint_dir,
        subfolder=subfolder,
        device=device,
        torch_dtype=torch_dtype,
    )


__all__ = ["HnegSRCPipeline", "HnegSRCPipelineOutput", "load_hneg_src_pipeline"]
