"""Paired image metric evaluator — unified interface for reference-based metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from src.metrics.paired.registry import METRIC_REGISTRY, get_metric_fn

if TYPE_CHECKING:
    from collections.abc import Sequence


class PairedImageMetricEvaluator:
    """Evaluator for paired (reference-based) image quality metrics.

    Compares generated images to reference images using one or more metrics.
    Supports PSNR, SSIM, LPIPS, DISTS, and SAMScore.

    Parameters
    ----------
    metrics :
        Metric names to compute (e.g. ``["psnr", "ssim", "lpips"]``).
        Use :meth:`available_metrics` to list all supported metrics.
    data_range :
        Value range of the images (typically 1.0 for [0, 1] or 2.0 for [-1, 1]).
    device :
        Device to run metric computation on.
    **kwargs :
        Extra options passed to individual metrics (e.g. ``lpips_net="alex"``).
    """

    def __init__(
        self,
        metrics: Sequence[str] | None = None,
        data_range: float = 1.0,
        device: str | torch.device = "cpu",
        **kwargs,
    ) -> None:
        self.metrics = list(metrics) if metrics else ["psnr", "ssim"]
        self.data_range = data_range
        self.device = torch.device(device)
        self._kwargs = kwargs

    @classmethod
    def available_metrics(cls) -> list[str]:
        """Return the list of supported metric names."""
        return list(METRIC_REGISTRY.keys())

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        metrics: Sequence[str] | None = None,
    ) -> dict[str, float]:
        """Compute paired metrics between prediction and target.

        Parameters
        ----------
        pred :
            Generated images, shape ``(N, C, H, W)``.
        target :
            Reference images, shape ``(N, C, H, W)``.
        metrics :
            Override metrics to compute. If ``None``, use constructor list.

        Returns
        -------
        dict[str, float] :
            Metric name -> scalar value. Higher is better for PSNR, SSIM, DISTS,
            SAMScore; lower is better for LPIPS.
        """
        which = list(metrics) if metrics is not None else self.metrics
        pred = pred.to(self.device)
        target = target.to(self.device)

        result: dict[str, float] = {}
        for name in which:
            if name not in METRIC_REGISTRY:
                raise ValueError(
                    f"Unknown metric '{name}'. Available: {self.available_metrics()}"
                )
            fn = get_metric_fn(name)
            result[name] = fn(
                pred,
                target,
                data_range=self.data_range,
                device=self.device,
                **self._kwargs,
            )
        return result
