"""SFD — Simplified Fréchet Distance (Kim et al., Sensors 2020)."""

from __future__ import annotations

import torch


def compute_sfd(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    feature_dim: int = 2048,
    device: torch.device | None = None,
    **kwargs,
) -> float:
    """Compute Simplified Fréchet Distance (diagonal covariance approx).

    Uses same Inception features as FID but replaces full covariance with
    per-dimension variance: d² = ||μ_r - μ_f||² + Σ_i (σ_r,i - σ_f,i)².

    Parameters
    ----------
    real_images, fake_images :
        Tensors of shape ``(N, C, H, W)`` in [0, 1].
    feature_dim :
        Inception feature dimensionality (2048).

    Returns
    -------
    float :
        SFD score. Lower is better.

    References
    ----------
    .. [1] Kim et al., "Simplified Fréchet Distance for Generative Adversarial
           Nets", Sensors 2020.
    """
    from torchmetrics.image.fid import FrechetInceptionDistance

    dev = device if device is not None else real_images.device
    fid = FrechetInceptionDistance(feature=feature_dim, normalize=True).to(dev)
    fid.update(real_images, real=True)
    fid.update(fake_images, real=False)

    n_r = fid.real_features_num_samples.float()
    n_f = fid.fake_features_num_samples.float()
    if n_r < 2 or n_f < 2:
        return float("inf")

    mu_r = (fid.real_features_sum / n_r).double()
    mu_f = (fid.fake_features_sum / n_f).double()
    cov_r = (fid.real_features_cov_sum - n_r * mu_r.outer(mu_r)) / (n_r - 1)
    cov_f = (fid.fake_features_cov_sum - n_f * mu_f.outer(mu_f)) / (n_f - 1)

    # Simplified: use diagonal stds only
    sigma_r = torch.diagonal(cov_r).clamp(min=1e-6).sqrt()
    sigma_f = torch.diagonal(cov_f).clamp(min=1e-6).sqrt()

    diff_mu = (mu_r - mu_f).norm(2).pow(2)
    diff_sigma = (sigma_r - sigma_f).norm(2).pow(2)
    sfd_sq = (diff_mu + diff_sigma).item()
    return max(0.0, sfd_sq) ** 0.5
