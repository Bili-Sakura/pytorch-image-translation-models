# Copyright (c) 2026 EarthBridge Team.
# Credits: Adapted from Parallel-GAN (Wang et al., TGRS 2022).

"""Inference / pipeline helpers for Parallel-GAN.

The ``ParaGAN`` generator's forward pass already returns a list of feature
tensors with the final RGB prediction at ``features[-1]``.  This module
provides a thin convenience wrapper for inference.
"""

from __future__ import annotations

import torch

from examples.community.parallel_gan.model import ParaGAN


@torch.no_grad()
def translate(
    generator: ParaGAN,
    source: torch.Tensor,
) -> torch.Tensor:
    """Translate a source (SAR) image to the target (optical) domain.

    Parameters
    ----------
    generator : ParaGAN
        A trained Parallel-GAN translation generator.
    source : Tensor ``[B, C_in, H, W]``
        Source-domain images in [-1, 1].

    Returns
    -------
    Tensor ``[B, C_out, H, W]``
        Predicted target-domain images in [-1, 1].
    """
    generator.eval()
    features = generator(source)
    return features[-1]
