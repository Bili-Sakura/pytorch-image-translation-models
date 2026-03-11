"""Unpaired image metric evaluator — unified interface for distribution-based metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from src.metrics.unpaired.registry import METRIC_REGISTRY, get_metric_fn

if TYPE_CHECKING:
    from collections.abc import Sequence


class UnpairedImageMetricEvaluator:
    """Evaluator for unpaired (distribution-based) image quality metrics.

    Compares distributions of real vs. generated images. Supports FID, KID, IS,
    and other distribution metrics.

    Parameters
    ----------
    metrics :
        Metric names to compute (e.g. ``["fid", "kid"]``).
        Use :meth:`available_metrics` to list all supported metrics.
    device :
        Device to run metric computation on.
    **kwargs :
        Extra options passed to individual metrics (e.g. ``feature_dim=2048``).
    """

    def __init__(
        self,
        metrics: Sequence[str] | None = None,
        device: str | torch.device = "cpu",
        **kwargs,
    ) -> None:
        self.metrics = list(metrics) if metrics else ["fid"]
        self.device = torch.device(device)
        self._kwargs = kwargs

    @classmethod
    def available_metrics(cls) -> list[str]:
        """Return the list of supported metric names."""
        return list(METRIC_REGISTRY.keys())

    def __call__(
        self,
        real_images: torch.Tensor,
        fake_images: torch.Tensor,
        metrics: Sequence[str] | None = None,
    ) -> dict[str, float]:
        """Compute unpaired metrics between real and fake image distributions.

        Parameters
        ----------
        real_images :
            Real/ground-truth images, shape ``(N, C, H, W)`` in [0, 1].
        fake_images :
            Generated images, shape ``(M, C, H, W)`` in [0, 1].
        metrics :
            Override metrics to compute. If ``None``, use constructor list.

        Returns
        -------
        dict[str, float] :
            Metric name -> scalar value. Lower is better for FID, KID; higher
            for IS.
        """
        which = list(metrics) if metrics is not None else self.metrics
        real_images = real_images.to(self.device)
        fake_images = fake_images.to(self.device)

        result: dict[str, float] = {}
        for name in which:
            if name not in METRIC_REGISTRY:
                raise ValueError(
                    f"Unknown metric '{name}'. Available: {self.available_metrics()}"
                )
            fn = get_metric_fn(name)
            result[name] = fn(
                real_images,
                fake_images,
                device=self.device,
                **self._kwargs,
            )
        return result
