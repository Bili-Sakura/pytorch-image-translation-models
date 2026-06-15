# Copyright (c) 2026 EarthBridge Team.
# Credits: F-LSeSim from Zheng et al. CVPR 2021 —
# https://github.com/lyndonzheng/F-LSeSim

"""F-LSeSim inference pipeline loader."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import torch

from src.pipelines.cut import FLSeSimPipeline, FLSeSimPipelineOutput


def load_flsesim_pipeline(
    checkpoint_dir: Union[str, Path],
    *,
    subfolder: str = "generator",
    device: str | torch.device = "cpu",
    torch_dtype: torch.dtype | None = None,
) -> FLSeSimPipeline:
    """Load an F-LSeSim pipeline from a training checkpoint directory.

    Checkpoints produced by :class:`~examples.flsesim.FLSeSimTrainer` store the
    generator under ``generator/`` in diffusers layout. Inference is a single
    generator forward pass.

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
    FLSeSimPipeline
        Ready-to-use inference pipeline.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    if not (checkpoint_dir / subfolder).exists():
        raise FileNotFoundError(
            f"Generator checkpoint not found: {checkpoint_dir / subfolder}"
        )

    return FLSeSimPipeline.from_pretrained(
        checkpoint_dir,
        subfolder=subfolder,
        device=device,
        torch_dtype=torch_dtype,
    )


__all__ = ["FLSeSimPipeline", "FLSeSimPipelineOutput", "load_flsesim_pipeline"]
