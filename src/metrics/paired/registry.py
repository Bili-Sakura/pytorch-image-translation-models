"""Registry of paired image metrics."""

from __future__ import annotations

from typing import Callable

from src.metrics.paired.psnr import compute_psnr
from src.metrics.paired.ssim import compute_ssim
from src.metrics.paired.lpips_impl import compute_lpips
from src.metrics.paired.dists_impl import compute_dists
from src.metrics.paired.l1_l2 import compute_l1, compute_l2

METRIC_REGISTRY: dict[str, Callable[..., float]] = {
    "psnr": compute_psnr,
    "ssim": compute_ssim,
    "lpips": compute_lpips,
    "dists": compute_dists,
    "l1": compute_l1,
    "l2": compute_l2,
}


def _register_samscore_if_available() -> None:
    """Lazily register SAMScore when segment_anything or transformers is available."""
    try:
        from src.metrics.paired.samscore_impl import compute_samscore

        METRIC_REGISTRY["samscore"] = compute_samscore
    except ImportError:
        pass


_register_samscore_if_available()


def get_metric_fn(name: str) -> Callable[..., float]:
    """Return the compute function for a registered metric."""
    if name not in METRIC_REGISTRY:
        raise KeyError(
            f"Unknown metric '{name}'. Available: {list(METRIC_REGISTRY.keys())}"
        )
    return METRIC_REGISTRY[name]


def register_metric(name: str, fn: Callable[..., float]) -> None:
    """Register a custom paired metric."""
    METRIC_REGISTRY[name] = fn
