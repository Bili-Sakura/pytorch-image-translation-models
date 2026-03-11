"""LPIPS — Learned Perceptual Image Patch Similarity (Zhang et al., CVPR 2018)."""

from __future__ import annotations

import torch


def compute_lpips(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    net: str = "alex",
    normalize: bool = True,
    device: torch.device | None = None,
    model_path: str | None = None,
    **kwargs,
) -> float:
    """Compute LPIPS perceptual distance.

    Uses the ``lpips`` package. Pass ``model_path`` to load custom weights.

    Parameters
    ----------
    pred, target :
        Tensors of shape ``(N, C, H, W)``. Expects [-1, 1] if ``normalize=False``,
        or [0, 1] if ``normalize=True`` (default).
    data_range :
        Ignored when normalize=True; used only for conversion.
    net :
        Backbone: ``"alex"``, ``"vgg"``, or ``"squeeze"``.
    model_path :
        Optional path to custom LPIPS weights ``.pth``.

    Returns
    -------
    float :
        Average LPIPS distance. Lower is better.

    References
    ----------
    .. [1] Zhang et al., "The Unreasonable Effectiveness of Deep Features as a
           Perceptual Metric", CVPR 2018.
    """
    import lpips

    if normalize and data_range == 1.0:
        pred = 2.0 * pred - 1.0
        target = 2.0 * target - 1.0
    elif normalize and data_range != 1.0:
        pred = 2.0 * (pred / data_range) - 1.0
        target = 2.0 * (target / data_range) - 1.0

    lpips_kwargs: dict = {"net": net}
    if model_path is not None:
        lpips_kwargs["model_path"] = model_path
    loss_fn = lpips.LPIPS(**lpips_kwargs).to(pred.device)
    with torch.no_grad():
        score = loss_fn(pred, target)
    return score.mean().item()
