# Credits: Built on open-source libraries and papers acknowledged in README.md citations.
"""L1 and L2 (MAE/MSE) — pixel-level error metrics."""

from __future__ import annotations

import torch


def compute_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    device: torch.device | None = None,
    **kwargs,
) -> float:
    """Compute L1 (mean absolute error, MAE) in normalized [0, 1] scale.

    .. math::
        \\text{L1} = \\frac{1}{n} \\sum |\\hat{y} - y|

    Parameters
    ----------
    pred, target :
        Tensors of shape ``(N, C, H, W)``.
    data_range :
        Value range for optional normalization. Not used in raw L1.

    Returns
    -------
    float :
        Mean absolute error over the batch. Lower is better.
    """
    return torch.nn.functional.l1_loss(pred, target).item()


def compute_l2(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    device: torch.device | None = None,
    **kwargs,
) -> float:
    """Compute L2 (mean squared error, MSE) in normalized scale.

    .. math::
        \\text{L2} = \\frac{1}{n} \\sum (\\hat{y} - y)^2

    Parameters
    ----------
    pred, target :
        Tensors of shape ``(N, C, H, W)``.
    data_range :
        Value range for optional normalization. Not used in raw L2.

    Returns
    -------
    float :
        Mean squared error over the batch. Lower is better.
    """
    return torch.nn.functional.mse_loss(pred, target).item()
