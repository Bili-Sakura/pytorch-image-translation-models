"""Evaluation metrics for image translation quality.

Metrics are classified by whether they require paired reference images:

- **Paired** (reference-based): Compare generated output to a reference
  (e.g., image translation, restoration). Use :class:`~src.metrics.paired.PairedImageMetricEvaluator`.

- **Unpaired** (distribution-based): Compare distributions of real vs. generated
  samples. Use :class:`~src.metrics.unpaired.UnpairedImageMetricEvaluator`.

- **Conditional diversity** (VS, AFD): Measure diversity among multiple outputs
  per source. Use :class:`~src.metrics.conditional.ConditionalDiversityMetricEvaluator`.

Example (paired metrics):
    >>> from src.metrics import PairedImageMetricEvaluator
    >>> evaluator = PairedImageMetricEvaluator(metrics=["psnr", "ssim", "lpips"])
    >>> scores = evaluator(generated, reference)
"""

from src.metrics.paired import PairedImageMetricEvaluator
from src.metrics.paired.psnr import compute_psnr
from src.metrics.paired.ssim import compute_ssim
from src.metrics.paired.lpips_impl import compute_lpips
from src.metrics.paired.dists_impl import compute_dists
from src.metrics.paired.l1_l2 import compute_l1, compute_l2

from src.metrics.unpaired import (
    UnpairedImageMetricEvaluator,
    compute_fid,
    compute_is,
    compute_kid,
    compute_sfd,
)

from src.metrics.conditional import (
    ConditionalDiversityMetricEvaluator,
    compute_vs,
    compute_afd,
)

__all__ = [
    "PairedImageMetricEvaluator",
    "compute_psnr",
    "compute_ssim",
    "compute_lpips",
    "compute_dists",
    "compute_l1",
    "compute_l2",
    "UnpairedImageMetricEvaluator",
    "compute_fid",
    "compute_is",
    "compute_kid",
    "compute_sfd",
    "ConditionalDiversityMetricEvaluator",
    "compute_vs",
    "compute_afd",
]
