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
    batch_size: int = 64,
    **kwargs,
) -> float:
    """Compute Fréchet Inception Distance.

    Uses torchmetrics for the classic Inception-based computation.
    Will be refactored to vendored implementation in the unpaired metrics suite.
    Processes images in batches to avoid CUDA OOM on large validation sets.

    Parameters
    ----------
    real_images, fake_images :
        Tensors of shape ``(N, C, H, W)`` in [0, 1].
    feature_dim :
        Feature dimensionality from InceptionV3 (2048).
    batch_size :
        Maximum images per batch for Inception forward. Smaller values reduce
        GPU memory use at the cost of more passes.

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

    n_real = real_images.shape[0]
    n_fake = fake_images.shape[0]
    for i in range(0, n_real, batch_size):
        batch = real_images[i : i + batch_size]
        if batch.device != dev:
            batch = batch.to(dev)
        fid.update(batch, real=True)
    for i in range(0, n_fake, batch_size):
        batch = fake_images[i : i + batch_size]
        if batch.device != dev:
            batch = batch.to(dev)
        fid.update(batch, real=False)
    return fid.compute().item()
