# Credits: CycleGAN (Zhu et al., ICCV 2017) — https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

"""CycleGAN inference pipeline loader."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import torch

from src.pipelines.cyclegan import CycleGANPipeline, CycleGANPipelineOutput, load_cyclegan_pipeline


def load_cyclegan_community_pipeline(
    checkpoint_dir: Union[str, Path],
    *,
    direction: str = "a2b",
    device: str | torch.device = "cpu",
    torch_dtype: torch.dtype | None = None,
    **kwargs,
) -> CycleGANPipeline:
    """Load a CycleGAN pipeline from HF or upstream junyanz checkpoints.

    Parameters
    ----------
    checkpoint_dir : str or Path
        Training checkpoint directory or upstream ``checkpoints/<name>_pretrained``.
    direction : str
        ``"a2b"`` or ``"b2a"`` when loading single-generator upstream weights.
    device : str or torch.device
        Target device.
    torch_dtype : torch.dtype, optional
        Optional dtype for generator weights.
    **kwargs
        Forwarded to :func:`load_cyclegan_pipeline` (``netG``, ``norm``, etc.).
    """
    return load_cyclegan_pipeline(
        checkpoint_dir,
        direction=direction,
        device=device,
        torch_dtype=torch_dtype,
        **kwargs,
    )


__all__ = [
    "CycleGANPipeline",
    "CycleGANPipelineOutput",
    "load_cyclegan_pipeline",
    "load_cyclegan_community_pipeline",
]
