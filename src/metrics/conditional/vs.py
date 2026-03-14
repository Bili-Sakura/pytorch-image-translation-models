"""VS — Vendi Score (Friedman & Dieng, TMLR 2023).

Conditional diversity metric based on the effective number of unique feature
patterns (exponential of Shannon entropy of similarity matrix eigenvalues).

References
----------
.. [1] Friedman & Dieng, "The Vendi Score: A Diversity Evaluation Metric for
       Machine Learning", Transactions on Machine Learning Research 2023.
       https://github.com/vertaix/Vendi-Score
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from src.metrics.conditional._inception import extract_inception_features

if TYPE_CHECKING:
    from collections.abc import Sequence


def _vendi_score_from_features(features: torch.Tensor) -> float:
    """Compute Vendi Score from normalized feature vectors.

    VS = exp(-Σ p_i log p_i) where p_i are normalized eigenvalues of the
    similarity matrix K/n. For normalized features, K = X @ X.T (cosine sim).
    """
    n = features.shape[0]
    if n < 2:
        return 0.0

    # L2 normalize for cosine similarity
    x = features / (features.norm(dim=1, keepdim=True) + 1e-8)
    # Similarity matrix K = X @ X^T (Gram matrix)
    K = x @ x.T
    # Normalize by n per Vendi Score definition
    K = K / n

    # Eigenvalues
    eigvals = torch.linalg.eigvalsh(K)
    # Clamp to avoid log(0)
    eigvals = eigvals.clamp(min=1e-10)

    # Shannon entropy: -Σ p_i log p_i
    entropy = -(eigvals * torch.log(eigvals)).sum().item()
    return float(torch.exp(torch.tensor(entropy)).item())


def compute_vs(
    generated_groups: Sequence[torch.Tensor] | torch.Tensor,
    feature_dim: int = 2048,
    device: torch.device | None = None,
    **kwargs,
) -> float:
    """Compute Vendi Score (conditional diversity) over groups of generated images.

    Each group contains L samples generated from the same source. VS measures
    the effective number of unique feature patterns per group (based on
    eigenvalues of the feature similarity matrix). Higher is better.

    Parameters
    ----------
    generated_groups :
        Either:
        - List of tensors, each ``(L_i, C, H, W)`` — L_i samples for source i
        - Single tensor ``(M, L, C, H, W)`` — M sources, L samples each
    feature_dim :
        Inception feature dimensionality (2048).
    device :
        Device for computation.

    Returns
    -------
    float :
        Mean Vendi Score over groups. Higher indicates better conditional diversity.

    References
    ----------
    .. [1] Friedman & Dieng, "The Vendi Score: A Diversity Evaluation Metric for
           Machine Learning", TMLR 2023.
    """
    dev = device if device is not None else torch.device("cpu")

    # Normalize to list of (L_i, C, H, W)
    if isinstance(generated_groups, torch.Tensor):
        if generated_groups.dim() == 5:
            groups = [g for g in generated_groups]
        else:
            groups = [generated_groups]
    else:
        groups = list(generated_groups)

    if not groups:
        return 0.0

    vs_scores: list[float] = []
    for group in groups:
        group = group.to(dev)
        if group.shape[0] < 2:
            continue
        features = extract_inception_features(
            group,
            feature_dim=feature_dim,
            device=dev,
            feature_extractor_weights_path=kwargs.get("feature_extractor_weights_path"),
        )
        vs_scores.append(_vendi_score_from_features(features))

    if not vs_scores:
        return 0.0
    return sum(vs_scores) / len(vs_scores)
