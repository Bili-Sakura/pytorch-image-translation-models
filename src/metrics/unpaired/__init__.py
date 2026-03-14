# Credits: Built on open-source libraries and papers acknowledged in README.md citations.
#
"""Unpaired (distribution-based) image quality metrics.

Unpaired metrics compare distributions of real vs. generated samples.
Used for unconditional generation and when reference images are unavailable.

Example:
    >>> from src.metrics.unpaired import UnpairedImageMetricEvaluator
    >>> evaluator = UnpairedImageMetricEvaluator(metrics=["fid", "kid"])
    >>> scores = evaluator(real_images, fake_images)
"""

from src.metrics.unpaired.evaluator import UnpairedImageMetricEvaluator
from src.metrics.unpaired.fid import compute_fid
from src.metrics.unpaired.is_ import compute_is
from src.metrics.unpaired.kid import compute_kid
from src.metrics.unpaired.sfd import compute_sfd

__all__ = [
    "UnpairedImageMetricEvaluator",
    "compute_fid",
    "compute_is",
    "compute_kid",
    "compute_sfd",
]
