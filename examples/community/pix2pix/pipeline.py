# Credits: pix2pix (Isola et al., CVPR 2017) — https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

"""pix2pix inference pipeline loader."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import torch

from src.pipelines.pix2pix import Pix2PixPipeline, Pix2PixPipelineOutput, load_pix2pix_pipeline


def load_pix2pix_community_pipeline(
    checkpoint_dir: Union[str, Path],
    *,
    device: str | torch.device = "cpu",
    torch_dtype: torch.dtype | None = None,
    **kwargs,
) -> Pix2PixPipeline:
    """Load a pix2pix pipeline from HF or upstream junyanz checkpoints.

    Parameters
    ----------
    checkpoint_dir : str or Path
        Training checkpoint directory or upstream ``checkpoints/<name>_pretrained``.
    device : str or torch.device
        Target device.
    torch_dtype : torch.dtype, optional
        Optional dtype for generator weights.
    **kwargs
        Forwarded to :func:`load_pix2pix_pipeline` (``netG``, ``norm``, etc.).
    """
    return load_pix2pix_pipeline(
        checkpoint_dir,
        device=device,
        torch_dtype=torch_dtype,
        **kwargs,
    )


__all__ = [
    "Pix2PixPipeline",
    "Pix2PixPipelineOutput",
    "load_pix2pix_pipeline",
    "load_pix2pix_community_pipeline",
]
