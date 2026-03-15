# Credits: LPIPS (Zhang et al., CVPR 2018). Built on open-source implementations.
#
"""LPIPS — Learned Perceptual Image Patch Similarity (Zhang et al., CVPR 2018)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    pass

# Cached LPIPS models to avoid recreating per batch (key: (net, model_path or ""))
_LPIPS_CACHE: dict[tuple[str, str], "torch.nn.Module"] = {}


def compute_lpips(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    net: str = "vgg",
    normalize: bool = True,
    device: torch.device | None = None,
    model_path: str | None = None,
    lpips_net: str | None = None,
    **kwargs,
) -> float:
    """Compute LPIPS perceptual distance.

    Uses the ``lpips`` package. Pass ``model_path`` to load custom weights.
    Default backbone is ``"vgg"``. Use ``net`` or ``lpips_net`` to override.

    Parameters
    ----------
    pred, target :
        Tensors of shape ``(N, C, H, W)``. Expects [-1, 1] if ``normalize=False``,
        or [0, 1] if ``normalize=True`` (default).
    data_range :
        Ignored when normalize=True; used only for conversion.
    net :
        Backbone: ``"alex"``, ``"vgg"`` (default), or ``"squeeze"``.
    lpips_net :
        Alias for ``net`` (evaluator compatibility).
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

    net = lpips_net if lpips_net is not None else net
    if normalize and data_range == 1.0:
        pred = 2.0 * pred - 1.0
        target = 2.0 * target - 1.0
    elif normalize and data_range != 1.0:
        pred = 2.0 * (pred / data_range) - 1.0
        target = 2.0 * (target / data_range) - 1.0

    cache_key = (net, model_path or "")
    if cache_key not in _LPIPS_CACHE:
        lpips_kwargs: dict = {"net": net}
        if model_path is not None:
            lpips_kwargs["model_path"] = model_path
        _LPIPS_CACHE[cache_key] = lpips.LPIPS(**lpips_kwargs)
    loss_fn = _LPIPS_CACHE[cache_key].to(pred.device)
    with torch.no_grad():
        score = loss_fn(pred, target)
    return score.mean().item()
