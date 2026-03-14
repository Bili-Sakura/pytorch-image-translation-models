# Credits: Built on open-source libraries and papers acknowledged in README.md citations.
#
"""FID — Fréchet Inception Distance (Heusel et al., NeurIPS 2017)."""

from __future__ import annotations

import torch


def compute_fid(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    feature_dim: int = 2048,
    device: torch.device | None = None,
    **kwargs,
) -> float:
    """Compute Fréchet Inception Distance.

    Uses torchmetrics for the classic Inception-based computation.
    Will be refactored to vendored implementation in the unpaired metrics suite.

    Parameters
    ----------
    real_images, fake_images :
        Tensors of shape ``(N, C, H, W)`` in [0, 1].
    feature_dim :
        Feature dimensionality from InceptionV3 (2048).

    Returns
    -------
    float :
        FID score. Lower is better.

    References
    ----------
    .. [1] Heusel et al., "GANs Trained by a Two Time-Scale Update Rule
           Converge to a Local Nash Equilibrium", NeurIPS 2017.
    """
    from torchmetrics.image.fid import FrechetInceptionDistance

    dev = device if device is not None else real_images.device
    fid = FrechetInceptionDistance(feature=feature_dim, normalize=True).to(dev)
    fid.update(real_images, real=True)
    fid.update(fake_images, real=False)
    return fid.compute().item()
