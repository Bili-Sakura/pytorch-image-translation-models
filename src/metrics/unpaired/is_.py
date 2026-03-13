# Credits: Built on open-source libraries and papers acknowledged in README.md citations.
#
"""IS — Inception Score (Salimans et al., NeurIPS 2016)."""

from __future__ import annotations

import torch


def compute_is(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    splits: int = 10,
    device: torch.device | None = None,
    **kwargs,
) -> float:
    """Compute Inception Score (quality & diversity of generated images).

    Uses only fake_images; real_images are ignored. Uses torchmetrics
    for the classic Inception-based computation.

    Parameters
    ----------
    real_images, fake_images :
        Tensors of shape ``(N, C, H, W)`` in [0, 1]. Only fake_images are used.
    splits :
        Number of splits for mean/std over the score.

    Returns
    -------
    float :
        Inception Score mean. Higher is better.

    References
    ----------
    .. [1] Salimans et al., "Improved Techniques for Training GANs", NeurIPS 2016.
    """
    from torchmetrics.image.inception import InceptionScore

    dev = device if device is not None else fake_images.device
    is_metric = InceptionScore(splits=splits, normalize=True).to(dev)
    is_metric.update(fake_images, real=False)
    is_mean, _ = is_metric.compute()
    return is_mean.item()
