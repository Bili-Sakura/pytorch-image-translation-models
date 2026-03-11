"""Registry of unpaired (distribution-based) image metrics."""

from __future__ import annotations

from typing import Callable

from src.metrics.unpaired.cmmd import compute_cmmd
from src.metrics.unpaired.fid import compute_fid
from src.metrics.unpaired.fwd import compute_fwd
from src.metrics.unpaired.ifid import compute_ifid
from src.metrics.unpaired.is_ import compute_is
from src.metrics.unpaired.kid import compute_kid
from src.metrics.unpaired.sfd import compute_sfd
from src.metrics.unpaired.sfid import compute_sfid

METRIC_REGISTRY: dict[str, Callable[..., float]] = {
    "fid": compute_fid,
    "kid": compute_kid,
    "is": compute_is,
    "sfd": compute_sfd,
    "sfid": compute_sfid,
    "cmmd": compute_cmmd,
    "fwd": compute_fwd,
    "ifid": compute_ifid,
}


def _register_precision_recall_if_available() -> None:
    """Register P&R metrics when torch-fidelity is available."""
    try:
        from src.metrics.unpaired.precision_recall import (
            compute_precision,
            compute_pr_f1,
            compute_recall,
        )

        METRIC_REGISTRY["precision"] = compute_precision
        METRIC_REGISTRY["recall"] = compute_recall
        METRIC_REGISTRY["pr_f1"] = compute_pr_f1
    except ImportError:
        pass


_register_precision_recall_if_available()


def get_metric_fn(name: str) -> Callable[..., float]:
    """Return the compute function for a registered metric."""
    if name not in METRIC_REGISTRY:
        raise KeyError(
            f"Unknown metric '{name}'. Available: {list(METRIC_REGISTRY.keys())}"
        )
    return METRIC_REGISTRY[name]


def register_metric(name: str, fn: Callable[..., float]) -> None:
    """Register a custom unpaired metric."""
    METRIC_REGISTRY[name] = fn
