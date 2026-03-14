"""Conditional diversity metrics (VS, AFD).

Measures diversity among multiple outputs generated from the same source.
Used for conditional image generation / translation evaluation.

Example:
    >>> from src.metrics.conditional import ConditionalDiversityMetricEvaluator
    >>> evaluator = ConditionalDiversityMetricEvaluator(metrics=["vs", "afd"])
    >>> # groups: list of (L, C, H, W) — L samples per source
    >>> scores = evaluator(generated_groups)
"""

from src.metrics.conditional.afd import compute_afd
from src.metrics.conditional.evaluator import ConditionalDiversityMetricEvaluator
from src.metrics.conditional.vs import compute_vs

__all__ = [
    "ConditionalDiversityMetricEvaluator",
    "compute_vs",
    "compute_afd",
]
