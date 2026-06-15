# Copyright (c) 2026 EarthBridge Team.
# Credits: NEGCUT from Wang et al. "Instance-wise Hard Negative Example Generation
# for Contrastive Learning in Unpaired Image-to-Image Translation" (ICCV 2021).
# Based on https://github.com/WeilunWang/NEGCUT

"""Learned patch contrastive loss for NEGCUT unpaired translation."""

from __future__ import annotations

import torch
import torch.nn as nn


class LearnedPatchNCELoss(nn.Module):
    """PatchNCE with adversarially learned hard negatives instead of batch keys.

    When ``neg_sample`` is provided, negatives come from the negative generator
    rather than cross-patch similarities within the source features.

    Parameters
    ----------
    nce_T : float
        Temperature for the contrastive logits.
    batch_size : int
        Mini-batch size used when reshaping negatives.
    nce_includes_all_negatives_from_minibatch : bool
        Include negatives from the full mini-batch (single-image translation).
    lambda_nce : float
        Scalar weight applied to the loss.
    """

    def __init__(
        self,
        nce_T: float = 0.07,
        batch_size: int = 1,
        nce_includes_all_negatives_from_minibatch: bool = False,
        lambda_nce: float = 1.0,
    ) -> None:
        super().__init__()
        self.nce_T = nce_T
        self.batch_size = batch_size
        self.nce_includes_all_negatives_from_minibatch = nce_includes_all_negatives_from_minibatch
        self.lambda_nce = lambda_nce
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")

    def forward(
        self,
        feat_q: torch.Tensor,
        feat_k: torch.Tensor,
        neg_sample: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        l_pos = torch.bmm(feat_q.view(batch_size, 1, -1), feat_k.view(batch_size, -1, 1))
        l_pos = l_pos.view(batch_size, 1)

        if self.nce_includes_all_negatives_from_minibatch:
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.batch_size

        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        if neg_sample is not None:
            neg_sample = neg_sample.view(batch_dim_for_bmm, -1, dim)
            npatches = neg_sample.size(1)
            l_neg = torch.bmm(feat_q, neg_sample.transpose(2, 1)).view(-1, npatches)
        else:
            feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
            npatches = feat_q.size(1)
            l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))
            diagonal = torch.eye(npatches, device=feat_q.device, dtype=torch.bool)[None, :, :]
            l_neg_curbatch.masked_fill_(diagonal, -10.0)
            l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.nce_T
        loss = self.cross_entropy_loss(
            out,
            torch.zeros(out.size(0), dtype=torch.long, device=feat_q.device),
        )
        return loss * self.lambda_nce
