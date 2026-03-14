"""Conditional diversity metric evaluator."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from src.metrics.conditional.registry import METRIC_REGISTRY, get_metric_fn

if TYPE_CHECKING:
    from collections.abc import Sequence


class ConditionalDiversityMetricEvaluator:
    """Evaluator for conditional diversity metrics (VS, AFD).

    Measures diversity among multiple outputs generated from the same source.
    Use for image translation / conditional generation where each source has
    multiple possible outputs.

    Parameters
    ----------
    metrics :
        Metric names to compute (e.g. ``["vs", "afd"]``).
        Use :meth:`available_metrics` to list all supported metrics.
    device :
        Device to run metric computation on.
    **kwargs :
        Extra options passed to individual metrics
        (e.g. ``feature_dim=2048``, ``feature_extractor_weights_path=...``).
    """

    def __init__(
        self,
        metrics: Sequence[str] | None = None,
        device: str | torch.device = "cpu",
        **kwargs,
    ) -> None:
        self.metrics = list(metrics) if metrics else ["vs", "afd"]
        self.device = torch.device(device)
        self._kwargs = kwargs

    @classmethod
    def available_metrics(cls) -> list[str]:
        """Return the list of supported metric names."""
        return list(METRIC_REGISTRY.keys())

    def __call__(
        self,
        generated_groups: Sequence[torch.Tensor] | torch.Tensor,
        metrics: Sequence[str] | None = None,
    ) -> dict[str, float]:
        """Compute conditional diversity metrics over groups of generated images.

        Parameters
        ----------
        generated_groups :
            Either:
            - List of tensors, each ``(L_i, C, H, W)`` — L_i samples per source i
            - Tensor ``(M, L, C, H, W)`` — M sources, L samples each
        metrics :
            Override metrics to compute. If ``None``, use constructor list.

        Returns
        -------
        dict[str, float] :
            Metric name -> scalar value. Higher is better for both VS and AFD.
        """
        which = list(metrics) if metrics is not None else self.metrics
        result: dict[str, float] = {}
        for name in which:
            if name not in METRIC_REGISTRY:
                raise ValueError(
                    f"Unknown metric '{name}'. Available: {self.available_metrics()}"
                )
            fn = get_metric_fn(name)
            result[name] = fn(
                generated_groups,
                device=self.device,
                **self._kwargs,
            )
        return result
