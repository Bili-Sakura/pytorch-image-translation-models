# Credits: SAMScore (Li et al., IEEE TAI 2025), https://github.com/Kent0n-Li/SAMScore.
#
"""SAMScore — Content Structural Similarity for Image Translation (Li et al., IEEE TAI 2025)."""

from __future__ import annotations

import torch


def compute_samscore(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    model_type: str = "vit_b",
    model_weight_path: str | None = None,
    **kwargs,
) -> float:
    """Compute SAMScore: content structural similarity via SAM embeddings.

    Uses the ``samscore`` package when available (from
    https://github.com/Kent0n-Li/SAMScore). Loads SAM from HuggingFace when
    using custom model paths. Falls back to segment_anything if samscore
    is not installed.

    Parameters
    ----------
    pred, target :
        Tensors of shape ``(N, C, H, W)`` in [0, 1].
    data_range :
        Value range; images are scaled to [0, 1] for SAM.
    model_type :
        ``"vit_b"``, ``"vit_l"``, or ``"vit_h"``.
    model_weight_path :
        Optional local path to SAM encoder ``.pth`` (e.g. ``sam_vit_b_01ec64.pth``).
        If omitted, the ``samscore`` package downloads from the default URL.

    Returns
    -------
    float :
        Average SAMScore similarity. Higher is better (in [0, 1]).

    References
    ----------
    .. [1] Li et al., "SAMScore: A Content Structural Similarity Metric for Image
           Translation Evaluation", IEEE TAI 2025.
    """
    if data_range != 1.0:
        pred = pred / data_range
        target = target / data_range
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)

    # Expect [0, 255] or [0, 1] for samscore; original expects [0, 255] from cv2
    # samscore.evaluation_from_torch uses preprocess which normalizes
    pred = pred * 255.0
    target = target * 255.0

    try:
        import samscore
    except ImportError as e:
        raise ImportError(
            "SAMScore requires the 'samscore' package. "
            "Install with: pip install git+https://github.com/Kent0n-Li/SAMScore.git"
        ) from e

    sam_kwargs: dict = {"model_type": model_type}
    if model_weight_path is not None:
        sam_kwargs["model_weight_path"] = model_weight_path
    evaluator = samscore.SAMScore(**sam_kwargs)
    with torch.no_grad():
        result = evaluator.evaluation_from_torch(pred, target)

    if isinstance(result, torch.Tensor):
        return result.mean().item()
    return float(result)
