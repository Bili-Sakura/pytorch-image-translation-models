"""Image quality metrics for evaluating translation results."""

from __future__ import annotations

import torch
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Peak Signal-to-Noise Ratio.

    Parameters
    ----------
    pred, target:
        Tensors of shape ``(N, C, H, W)`` in [0, 1].

    Returns
    -------
    float:
        Average PSNR over the batch.
    """
    metric = PeakSignalNoiseRatio(data_range=1.0).to(pred.device)
    return metric(pred, target).item()


def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Structural Similarity Index.

    Parameters
    ----------
    pred, target:
        Tensors of shape ``(N, C, H, W)`` in [0, 1].

    Returns
    -------
    float:
        Average SSIM over the batch.
    """
    metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(pred.device)
    return metric(pred, target).item()


def compute_lpips(
    pred: torch.Tensor,
    target: torch.Tensor,
    net: str = "alex",
) -> float:
    """Learned Perceptual Image Patch Similarity.

    Requires the ``lpips`` package.

    Parameters
    ----------
    pred, target:
        Tensors of shape ``(N, C, H, W)`` in [-1, 1].
    net:
        Backbone network (``"alex"``, ``"vgg"``, ``"squeeze"``).

    Returns
    -------
    float:
        Average LPIPS distance.
    """
    import lpips as _lpips

    loss_fn = _lpips.LPIPS(net=net).to(pred.device)
    with torch.no_grad():
        score = loss_fn(pred, target)
    return score.mean().item()


def compute_fid(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    feature_dim: int = 2048,
) -> float:
    """Fréchet Inception Distance.

    Requires the ``torch-fidelity`` package for accurate computation.
    This function provides a lightweight numpy-based approximation
    when only tensors are available.

    Parameters
    ----------
    real_images, fake_images:
        Tensors of shape ``(N, C, H, W)`` in [0, 1].
    feature_dim:
        Feature dimensionality from InceptionV3.

    Returns
    -------
    float:
        FID score (lower is better).
    """
    from torchmetrics.image.fid import FrechetInceptionDistance

    fid = FrechetInceptionDistance(feature=feature_dim, normalize=True).to(
        real_images.device
    )
    fid.update(real_images, real=True)
    fid.update(fake_images, real=False)
    return fid.compute().item()
