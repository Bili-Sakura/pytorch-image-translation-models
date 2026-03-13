# Credits: Built on open-source libraries and papers acknowledged in README.md citations.
#
"""Paired (reference-based) image quality metrics.

Paired metrics compare generated output to a reference image. Used for image
translation, restoration, and similar tasks where ground truth is available.

Example:
    >>> from src.metrics.paired import PairedImageMetricEvaluator
    >>> evaluator = PairedImageMetricEvaluator(metrics=["psnr", "ssim", "lpips"])
    >>> scores = evaluator(generated, reference)
    >>> print(scores)  # {"psnr": 28.5, "ssim": 0.92, "lpips": 0.15}
"""

from src.metrics.paired.evaluator import PairedImageMetricEvaluator

__all__ = ["PairedImageMetricEvaluator"]
