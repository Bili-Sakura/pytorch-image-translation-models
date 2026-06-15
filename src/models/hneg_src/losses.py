# Copyright (c) 2026 EarthBridge Team.
# Credits: Hneg-SRC from Jung et al. "Exploring Patch-wise Semantic Relation for
# Contrastive Learning in Image-to-Image Translation Tasks" (CVPR 2022).
# Based on https://github.com/jcy132/Hneg_SRC

"""Semantic-relation contrastive losses for Hneg-SRC unpaired translation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureNormalize(nn.Module):
    """L2-normalize features along the channel dimension."""

    def __init__(self, power: float = 2.0) -> None:
        super().__init__()
        self.power = power

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1.0 / self.power)
        return x.div(norm + 1e-7)


class SRCLoss(nn.Module):
    """Patch-wise semantic relation contrastive (SRC) loss.

    Computes JSD between patch relation matrices of source and translated
    features, and returns per-layer weights for :class:`PatchHDCELoss`.

    Parameters
    ----------
    num_patches : int
        Number of sampled patches (used for relation matrix size).
    hdce_gamma : float
        Temperature for semantic-relation weight softmax.
    hdce_gamma_min : float
        Minimum gamma when curriculum scheduling is enabled.
    use_curriculum : bool
        Linearly anneal ``hdce_gamma`` toward ``hdce_gamma_min`` over training.
    step_gamma : bool
        Snap gamma to 1 after ``step_gamma_epoch``.
    step_gamma_epoch : int
        Epoch at which gamma is snapped to 1 when ``step_gamma`` is True.
    n_epochs : int
        Total training epochs (for curriculum).
    n_epochs_decay : int
        Decay epochs appended to ``n_epochs`` for curriculum denominator.
    lambda_src : float
        Scalar weight applied to the SRC loss.
    """

    def __init__(
        self,
        num_patches: int = 256,
        hdce_gamma: float = 1.0,
        hdce_gamma_min: float = 1.0,
        use_curriculum: bool = False,
        step_gamma: bool = False,
        step_gamma_epoch: int = 200,
        n_epochs: int = 200,
        n_epochs_decay: int = 200,
        lambda_src: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_patches = num_patches
        self.hdce_gamma = hdce_gamma
        self.hdce_gamma_min = hdce_gamma_min
        self.use_curriculum = use_curriculum
        self.step_gamma = step_gamma
        self.step_gamma_epoch = step_gamma_epoch
        self.n_epochs = n_epochs
        self.n_epochs_decay = n_epochs_decay
        self.lambda_src = lambda_src
        self._normalize = FeatureNormalize()

    def _current_gamma(self, epoch: int | None) -> float:
        gamma = self.hdce_gamma
        if self.use_curriculum and epoch is not None:
            denom = self.n_epochs + self.n_epochs_decay
            gamma = gamma + (self.hdce_gamma_min - gamma) * epoch / max(denom, 1)
            if self.step_gamma and epoch > self.step_gamma_epoch:
                gamma = 1.0
        return gamma

    def forward(
        self,
        feat_q: torch.Tensor,
        feat_k: torch.Tensor,
        *,
        only_weight: bool = False,
        epoch: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute SRC loss and semantic-relation weights.

        Parameters
        ----------
        feat_q : torch.Tensor
            Query (translated) patch features, shape ``(N, C)``.
        feat_k : torch.Tensor
            Key (source) patch features, shape ``(N, C)``.
        only_weight : bool
            If True, return zero loss and only compute relation weights.
        epoch : int, optional
            Current epoch for curriculum gamma scheduling.

        Returns
        -------
        loss : torch.Tensor
            Scalar SRC loss (0 when ``only_weight`` is True).
        weight : torch.Tensor
            Per-patch semantic relation weights, shape ``(1, num_patches, num_patches)``.
        """
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()
        feat_k = self._normalize(feat_k)
        feat_q = self._normalize(feat_q)

        batch_dim_for_bmm = 1
        feat_q_v = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k_v = feat_k.view(batch_dim_for_bmm, -1, dim)

        spatial_q = torch.bmm(feat_q_v, feat_q_v.transpose(2, 1))
        spatial_k = torch.bmm(feat_k_v, feat_k_v.transpose(2, 1))

        weight_seed = spatial_k.clone().detach()
        diagonal = torch.eye(
            self.num_patches, device=feat_k_v.device, dtype=torch.bool
        )[None, :, :]

        gamma = self._current_gamma(epoch)
        weight_seed.masked_fill_(diagonal, -10.0)
        weight_out = F.softmax(weight_seed / gamma, dim=2).detach()
        wmax_out, _ = torch.max(weight_out, dim=2, keepdim=True)
        weight_out = weight_out / wmax_out

        if only_weight:
            return torch.tensor(0.0, device=feat_q.device), weight_out

        spatial_q = F.softmax(spatial_q, dim=1)
        spatial_k = F.softmax(spatial_k, dim=1).detach()
        loss_src = self._jsd(spatial_q, spatial_k) * self.lambda_src
        return loss_src, weight_out

    @staticmethod
    def _jsd(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        m = 0.5 * (p1 + p2)
        return 0.5 * (
            F.kl_div(torch.log(m), torch.log(p1), reduction="sum", log_target=True)
            + F.kl_div(torch.log(m), torch.log(p2), reduction="sum", log_target=True)
        )


class PatchHDCELoss(nn.Module):
    """Hard-negative DCE (HDCE) contrastive loss weighted by semantic relations.

    Extends patch-wise contrastive learning with relation-derived weights from
    :class:`SRCLoss`, following the Hneg-SRC formulation.

    Parameters
    ----------
    nce_T : float
        Temperature for the contrastive logits.
    batch_size : int
        Mini-batch size used when reshaping negatives.
    nce_includes_all_negatives_from_minibatch : bool
        Include negatives from the full mini-batch (single-image translation).
    lambda_hdce : float
        Scalar weight applied to the HDCE loss.
    """

    def __init__(
        self,
        nce_T: float = 0.07,
        batch_size: int = 1,
        nce_includes_all_negatives_from_minibatch: bool = False,
        lambda_hdce: float = 1.0,
    ) -> None:
        super().__init__()
        self.nce_T = nce_T
        self.batch_size = batch_size
        self.nce_includes_all_negatives_from_minibatch = nce_includes_all_negatives_from_minibatch
        self.lambda_hdce = lambda_hdce
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")

    def forward(
        self,
        feat_q: torch.Tensor,
        feat_k: torch.Tensor,
        weight: torch.Tensor | None = None,
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
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        if weight is not None:
            l_neg_curbatch = l_neg_curbatch * weight

        diagonal = torch.eye(npatches, device=feat_q.device, dtype=torch.bool)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        logits = (l_neg - l_pos) / self.nce_T
        v = torch.logsumexp(logits, dim=1)
        loss_vec = torch.exp(v - v.detach())

        out_dummy = torch.cat((l_pos, l_neg), dim=1) / self.nce_T
        ce_dummy = self.cross_entropy_loss(
            out_dummy,
            torch.zeros(out_dummy.size(0), dtype=torch.long, device=feat_q.device),
        )

        loss = loss_vec.mean() - 1 + ce_dummy.detach()
        return loss * self.lambda_hdce
