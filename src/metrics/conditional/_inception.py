"""Inception feature extraction for conditional diversity metrics.

Uses the same InceptionV3 backbone as FID (torchmetrics/torch-fidelity) for
consistent feature space with other image metrics.
"""

from __future__ import annotations

import torch


def extract_inception_features(
    images: torch.Tensor,
    feature_dim: int = 2048,
    device: torch.device | None = None,
    feature_extractor_weights_path: str | None = None,
) -> torch.Tensor:
    """Extract InceptionV3 features for a batch of images.

    Uses the same feature extractor as FID for consistency. Images are
    expected to be in [0, 1] range and will be resized to 299x299.

    Parameters
    ----------
    images :
        Tensor of shape ``(N, C, H, W)`` in [0, 1].
    feature_dim :
        Inception feature dimensionality (2048).
    device :
        Device for computation.
    feature_extractor_weights_path :
        Optional path to Inception weights (FID-compat format).

    Returns
    -------
    torch.Tensor :
        Features of shape ``(N, feature_dim)``.
    """
    from torchmetrics.image.fid import FrechetInceptionDistance

    dev = device if device is not None else images.device
    fid = FrechetInceptionDistance(
        feature=feature_dim,
        normalize=True,
        feature_extractor_weights_path=feature_extractor_weights_path,
    ).to(dev)
    fid.eval()

    with torch.no_grad():
        imgs = (images * 255).byte().to(dev)
        features = fid.inception(imgs)

    return features.float()
