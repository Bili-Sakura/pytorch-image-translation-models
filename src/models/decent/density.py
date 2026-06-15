# Copyright (c) 2026 EarthBridge Team.
# Credits: Decent from Xie et al. NeurIPS 2022 — https://github.com/Mid-Push/Decent

"""Patch density estimators for Decent density-changing regularization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.models.decent.flows.bnaf import BNAFModel
from src.models.decent.flows.maf import MAF, MAFMOG


@dataclass
class FlowConfig:
    """Configuration for per-layer normalizing-flow density estimators."""

    flow_type: str = "bnaf"
    flow_blocks: int = 1
    bnaf_layers: int = 0
    bnaf_dim: int = 10
    maf_dim: int = 1024
    maf_layers: int = 2
    maf_comps: int = 10


def create_flow_model(input_dim: int, config: FlowConfig) -> nn.Module:
    """Instantiate a flow density model for a single encoder layer."""
    if config.flow_type == "maf":
        return MAF(
            input_dim,
            n_blocks=config.flow_blocks,
            hidden_size=config.maf_dim,
            n_hidden=config.maf_layers,
        )
    if config.flow_type == "bnaf":
        return BNAFModel(
            input_dim,
            n_flows=config.flow_blocks,
            n_layers=config.bnaf_layers,
            hidden_dim=config.bnaf_dim,
        )
    if config.flow_type == "nsf":
        from src.models.decent.flows.nsf import NSF

        return NSF(input_dim, config.flow_blocks)
    if config.flow_type == "mafmog":
        return MAFMOG(
            config.flow_blocks,
            config.maf_comps,
            input_dim,
            config.maf_dim,
            config.maf_layers,
        )
    raise ValueError(f"Unknown flow_type: {config.flow_type}")


class PatchDensityEstimator(nn.Module):
    """Per-domain patch density estimator using normalizing flows.

    Lazily creates one flow per encoder layer on the first forward pass,
    matching the data-dependent initialisation in the upstream Decent repo.
    """

    def __init__(self, config: FlowConfig) -> None:
        super().__init__()
        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))
        self.config = config

    def create_flows(self, feats: List[torch.Tensor]) -> None:
        for mlp_id, feat in enumerate(feats):
            input_nc = feat.shape[1]
            flow = create_flow_model(input_nc, self.config)
            flow = flow.to(feat.device)
            setattr(self, f"flow_{mlp_id}", flow)

    def forward(
        self,
        feats: List[torch.Tensor],
        num_patches: int = 256,
        patch_ids: Optional[List] = None,
        *,
        detach: bool = False,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List]:
        return_log_probs: list = []
        return_lens: list = []
        return_ids: list = []

        if self.initialized.item() == 0:
            self.create_flows(feats)
            self.initialized.fill_(1)

        for feat_id, feat in enumerate(feats):
            feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)
            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[feat_id]
                else:
                    patch_id = np.random.permutation(feat_reshape.shape[1])
                    patch_id = patch_id[: int(min(num_patches, patch_id.shape[0]))]
                    patch_id = torch.as_tensor(patch_id, dtype=torch.long, device=feat.device)
                x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)
            else:
                x_sample = feat_reshape
                patch_id = []
            return_ids.append(patch_id)

            flow = getattr(self, f"flow_{feat_id}")
            if detach:
                x_sample = x_sample.detach()
            return_log_probs.append(flow.log_probs(x_sample))
            return_lens.append(
                torch.tensor(feat_reshape.size(-1), dtype=torch.float32, device=feat.device)
            )

        return return_log_probs, return_lens, return_ids
