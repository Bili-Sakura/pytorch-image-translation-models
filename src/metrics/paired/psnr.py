# Credits: Built on open-source libraries and papers acknowledged in README.md citations.
#
"""Peak Signal-to-Noise Ratio (PSNR) — classic pixel-level metric."""

from __future__ import annotations

import torch


def compute_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    device: torch.device | None = None,
    **kwargs,
) -> float:
    """Compute Peak Signal-to-Noise Ratio.

    .. math::
        \\text{PSNR} = 10 \\log_{10}\\left(\\frac{\\max^2}{\\text{MSE}}\\right)

    Parameters
    ----------
    pred, target :
        Tensors of shape ``(N, C, H, W)`` in [0, 1].
    data_range :
        Value range (typically 1.0).

    Returns
    -------
    float :
        Average PSNR over the batch. Higher is better.

    References
    ----------
    .. [1] Wang et al., "Image Quality Assessment: From Error Visibility to
           Structural Similarity", IEEE TIP 2004.
    """
    mse = torch.nn.functional.mse_loss(pred, target)
    if mse.item() == 0:
        return float("inf")
    max_sq = data_range**2
    psnr = 10 * torch.log10(max_sq / mse)
    return psnr.item()
