"""AFD — Average Feature Distance (Zhang et al., NeurIPS 2025).

Conditional diversity metric: average pairwise Euclidean distance between
Inception features within each source's generated samples. Higher is better.

References
----------
.. [1] Zhang et al., "Exploring the Design Space of Diffusion Bridge Models",
       NeurIPS 2025. https://github.com/szhan311/ECSI
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from src.metrics.conditional._inception import extract_inception_features

if TYPE_CHECKING:
    from collections.abc import Sequence


def _afd_from_features_vectorized(features: torch.Tensor) -> float:
    """Vectorized AFD: (1/(L^2-L)) * sum_{k!=l} ||f_k - f_l||."""
    L = features.shape[0]
    if L < 2:
        return 0.0

    # Pairwise L2: ||f_k - f_l|| = sqrt(||f_k||^2 + ||f_l||^2 - 2 f_k·f_l)
    sq_norms = (features**2).sum(dim=1)
    gram = features @ features.T
    sq_dists = sq_norms.unsqueeze(0) + sq_norms.unsqueeze(1) - 2 * gram
    sq_dists = sq_dists.clamp(min=0)
    dists = sq_dists.sqrt()

    # Exclude diagonal, sum all pairs (each counted once)
    mask = ~torch.eye(L, dtype=torch.bool, device=features.device)
    total = dists[mask].sum().item()
    n_pairs = L * (L - 1)
    return total / n_pairs


def compute_afd(
    generated_groups: Sequence[torch.Tensor] | torch.Tensor,
    feature_dim: int = 2048,
    device: torch.device | None = None,
    **kwargs,
) -> float:
    """Compute Average Feature Distance (conditional diversity) over groups.

    AFD = (1/M) * sum_i [ (1/(L^2-L)) * sum_{k!=l} ||F(y_ik) - F(y_il)|| ]

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
        Mean AFD over groups. Higher indicates better conditional diversity.

    References
    ----------
    .. [1] Zhang et al., "Exploring the Design Space of Diffusion Bridge Models",
           NeurIPS 2025.
    """
    dev = device if device is not None else torch.device("cpu")

    if isinstance(generated_groups, torch.Tensor):
        if generated_groups.dim() == 5:
            groups = [g for g in generated_groups]
        else:
            groups = [generated_groups]
    else:
        groups = list(generated_groups)

    if not groups:
        return 0.0

    afd_scores: list[float] = []
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
        afd_scores.append(_afd_from_features_vectorized(features))

    if not afd_scores:
        return 0.0
    return sum(afd_scores) / len(afd_scores)
