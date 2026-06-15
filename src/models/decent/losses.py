# Copyright (c) 2026 EarthBridge Team.
# Credits: Decent from Xie et al. NeurIPS 2022 — https://github.com/Mid-Push/Decent

"""Density-changing regularization losses for Decent."""

from __future__ import annotations

import math
from typing import List

import torch


def compute_flow_nll_loss(
    log_probs_a: List[torch.Tensor],
    log_probs_b: List[torch.Tensor],
) -> torch.Tensor:
    """Negative log-likelihood objective for flow density estimators."""
    loss_nll_a = sum(-log_prob.mean() for log_prob in log_probs_a)
    loss_nll_b = sum(-log_prob.mean() for log_prob in log_probs_b)
    return (loss_nll_a + loss_nll_b) / len(log_probs_a)


def compute_density_changing_loss(
    log_probs_a: List[torch.Tensor],
    log_probs_b: List[torch.Tensor],
    feat_lens: List[torch.Tensor],
    *,
    batch_size: int,
    num_patches: int,
    var_all: bool = False,
) -> torch.Tensor:
    """Density-changing variance loss from Decent (NeurIPS 2022)."""
    total_var_loss = 0.0
    for log_prob_a, log_prob_b, feat_len in zip(log_probs_a, log_probs_b, feat_lens):
        density_changes = (log_prob_a.detach() - log_prob_b).squeeze()
        density_changes_per_dim = density_changes / (feat_len.mean().item() * math.log(2))

        if var_all:
            loss_layer = torch.var(density_changes_per_dim).mean()
        else:
            density_changes_per_dim = density_changes_per_dim.view(batch_size, num_patches)
            loss_layer = torch.var(density_changes_per_dim, dim=-1).mean()
        total_var_loss += loss_layer

    return total_var_loss / len(log_probs_a)
