"""KID — Kernel Inception Distance (Binkowski et al., ICLR 2018)."""

from __future__ import annotations

import torch


def compute_kid(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    feature_dim: int = 2048,
    subsets: int = 100,
    subset_size: int = 1000,
    device: torch.device | None = None,
    **kwargs,
) -> float:
    """Compute Kernel Inception Distance (MMD with polynomial kernel).

    Uses torchmetrics for the classic Inception-based computation.

    Parameters
    ----------
    real_images, fake_images :
        Tensors of shape ``(N, C, H, W)`` in [0, 1].
    feature_dim :
        Inception feature dimensionality (2048).
    subsets :
        Number of subsets for mean/std over bootstrap.
    subset_size :
        Samples per subset.

    Returns
    -------
    float :
        KID mean. Lower is better.

    References
    ----------
    .. [1] Binkowski et al., "Demystifying MMD GANs", ICLR 2018.
    """
    from torchmetrics.image.kid import KernelInceptionDistance

    dev = device if device is not None else real_images.device
    kid = KernelInceptionDistance(
        feature=feature_dim,
        subsets=subsets,
        subset_size=subset_size,
        normalize=True,
    ).to(dev)
    kid.update(real_images, real=True)
    kid.update(fake_images, real=False)
    kid_mean, _ = kid.compute()
    return kid_mean.item()
