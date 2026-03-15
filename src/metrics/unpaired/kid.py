# Credits: Built on open-source libraries and papers acknowledged in README.md citations.
#
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
    batch_size: int = 64,
    **kwargs,
) -> float:
    """Compute Kernel Inception Distance (MMD with polynomial kernel).

    Uses torchmetrics for the classic Inception-based computation.
    Processes images in batches to avoid CUDA OOM on large validation sets.

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
    batch_size :
        Maximum images per batch for Inception forward.

    Returns
    -------
    float :
        KID mean. Lower is better.

    References
    ----------
    .. [1] Binkowski et al., "Demystifying MMD GANs", ICLR 2018.
    """
    from torchmetrics.image.kid import KernelInceptionDistance

    n = min(real_images.shape[0], fake_images.shape[0])
    if n <= 1:
        return 0.0
    subset_size = min(subset_size, n - 1)

    dev = device if device is not None else real_images.device
    kid = KernelInceptionDistance(
        feature=feature_dim,
        subsets=subsets,
        subset_size=subset_size,
        normalize=True,
    ).to(dev)

    n_real = real_images.shape[0]
    n_fake = fake_images.shape[0]
    for i in range(0, n_real, batch_size):
        batch = real_images[i : i + batch_size]
        if batch.device != dev:
            batch = batch.to(dev)
        kid.update(batch, real=True)
    for i in range(0, n_fake, batch_size):
        batch = fake_images[i : i + batch_size]
        if batch.device != dev:
            batch = batch.to(dev)
        kid.update(batch, real=False)
    kid_mean, _ = kid.compute()
    return kid_mean.item()
