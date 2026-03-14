"""Registry of conditional diversity metrics."""

from __future__ import annotations

from typing import Callable

from src.metrics.conditional.afd import compute_afd
from src.metrics.conditional.vs import compute_vs

METRIC_REGISTRY: dict[str, Callable[..., float]] = {
    "vs": compute_vs,
    "afd": compute_afd,
}


def get_metric_fn(name: str) -> Callable[..., float]:
    """Return the compute function for a registered metric."""
    if name not in METRIC_REGISTRY:
        raise KeyError(
            f"Unknown metric '{name}'. Available: {list(METRIC_REGISTRY.keys())}"
        )
    return METRIC_REGISTRY[name]
