# Credits: Built on open-source libraries and papers acknowledged in README.md citations.
#
"""Structural Similarity Index (SSIM) — classic pixel-level metric."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _gaussian_kernel_2d(
    channels: int,
    size: int,
    sigma: float,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Create a 2D Gaussian kernel for SSIM."""
    coords = torch.arange(size, dtype=dtype, device=device) - (size - 1) / 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel = g.unsqueeze(0) * g.unsqueeze(1)
    kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
    return kernel


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    kernel_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
    device: torch.device | None = None,
    **kwargs,
) -> float:
    """Compute Structural Similarity Index Measure.

    Parameters
    ----------
    pred, target :
        Tensors of shape ``(N, C, H, W)`` in [0, 1].
    data_range :
        Value range (typically 1.0).
    kernel_size :
        Size of the Gaussian window.
    sigma :
        Standard deviation of the Gaussian.
    k1, k2 :
        SSIM stability constants.

    Returns
    -------
    float :
        Average SSIM over the batch. Higher is better (in [0, 1]).

    References
    ----------
    .. [1] Wang et al., "Image Quality Assessment: From Error Visibility to
           Structural Similarity", IEEE TIP 2004.
    """
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    channel = pred.size(1)
    pad = kernel_size // 2
    kernel = _gaussian_kernel_2d(
        channel, kernel_size, sigma, pred.dtype, pred.device
    )

    pred = F.pad(pred, [pad] * 4, mode="reflect")
    target = F.pad(target, [pad] * 4, mode="reflect")

    mu_pred = F.conv2d(pred, kernel, groups=channel)
    mu_target = F.conv2d(target, kernel, groups=channel)
    mu_pred_sq = mu_pred**2
    mu_target_sq = mu_target**2
    mu_pred_target = mu_pred * mu_target

    sigma_pred_sq = F.conv2d(pred * pred, kernel, groups=channel) - mu_pred_sq
    sigma_target_sq = (
        F.conv2d(target * target, kernel, groups=channel) - mu_target_sq
    )
    sigma_pred_target = (
        F.conv2d(pred * target, kernel, groups=channel) - mu_pred_target
    )

    ssim_map = (
        (2 * mu_pred_target + c1)
        * (2 * sigma_pred_target + c2)
        / ((mu_pred_sq + mu_target_sq + c1) * (sigma_pred_sq + sigma_target_sq + c2))
    )
    return ssim_map.mean().item()
