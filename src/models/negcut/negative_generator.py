# Copyright (c) 2026 EarthBridge Team.
# Credits: NEGCUT from Wang et al. ICCV 2021 — https://github.com/WeilunWang/NEGCUT

"""Online hard-negative generators for NEGCUT contrastive learning."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from src.models.cut.cut_model import _init_weights


class FeatureNormalize(nn.Module):
    """L2-normalize features along the channel dimension."""

    def __init__(self, power: float = 2.0) -> None:
        super().__init__()
        self.power = power

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1.0 / self.power)
        return x.div(norm + 1e-7)


class NegativePlaceholder(nn.Module):
    """Learnable negative patch embeddings (``neg_param`` mode)."""

    def __init__(
        self,
        nce_layers: List[int],
        num_patches: int = 256,
        nc: int = 256,
    ) -> None:
        super().__init__()
        self.l2norm = FeatureNormalize()
        self.nce_layers = nce_layers
        self.num_patches = num_patches
        self.nc = nc
        self.neg_sample = nn.Parameter(
            torch.empty(len(nce_layers), 1, num_patches, nc),
            requires_grad=True,
        )
        nn.init.xavier_normal_(self.neg_sample, gain=1.0)

    def forward(self, nce_layers: List[int], num_images: int) -> List[torch.Tensor]:
        del nce_layers
        return_feats: list[torch.Tensor] = []
        for layer_id in range(len(self.nce_layers)):
            neg_sample = self.neg_sample[layer_id].repeat(num_images, 1, 1)
            neg_sample = neg_sample.view(-1, self.nc)
            return_feats.append(self.l2norm(neg_sample))
        return return_feats


class NegativeGenerator(nn.Module):
    """Adversarial negative generator conditioned on encoder or projected features."""

    def __init__(
        self,
        *,
        use_conv: bool = False,
        num_patches: int = 256,
        nc: int = 256,
        z_dim: int = 64,
        init_type: str = "xavier",
        init_gain: float = 0.02,
    ) -> None:
        super().__init__()
        self.l2norm = FeatureNormalize()
        self.num_patches = num_patches
        self.nc = nc
        self.z_dim = z_dim
        self.use_conv = use_conv
        self.layer_init = False
        self.init_type = init_type
        self.init_gain = init_gain

    def create_layers(self, feats: List[torch.Tensor]) -> None:
        for feat_id, feat in enumerate(feats):
            input_nc = feat.shape[1]
            if self.use_conv:
                conv = nn.Sequential(
                    nn.Conv2d(input_nc, self.nc, 1, 1),
                    nn.ReLU(),
                    nn.Conv2d(self.nc, self.nc, 1, 1),
                )
                setattr(self, f"conv_{feat_id}", conv)
            mlp = nn.Sequential(
                nn.Linear(self.nc + self.z_dim, self.nc),
                nn.ReLU(),
                nn.Linear(self.nc, self.nc),
            )
            setattr(self, f"mlp_{feat_id}", mlp)
        _init_weights(self, self.init_type, self.init_gain)
        self.layer_init = True

    def forward(self, feats: List[torch.Tensor], num_patches: int) -> List[torch.Tensor]:
        if not self.layer_init:
            self.create_layers(feats)

        return_feats: list[torch.Tensor] = []
        for feat_id, feat in enumerate(feats):
            noise = torch.randn(
                feat.size(0),
                num_patches,
                self.z_dim,
                device=feat.device,
                dtype=feat.dtype,
            )
            if self.use_conv:
                conv = getattr(self, f"conv_{feat_id}")
                feat = conv(feat)
                feat = feat.permute(0, 2, 3, 1).mean(dim=(1, 2))
            else:
                feat = feat.mean(dim=(2, 3))
            feat = feat.unsqueeze(1).repeat(1, num_patches, 1)
            inp = torch.cat([feat, noise], dim=2).flatten(0, 1)
            mlp = getattr(self, f"mlp_{feat_id}")
            neg_sample = self.l2norm(mlp(inp))
            return_feats.append(neg_sample)
        return return_feats


def create_negative_generator(
    nce_layers: List[int],
    netN: str = "neg_gen_momentum",
    *,
    num_patches: int = 256,
    nc: int = 256,
    init_type: str = "xavier",
    init_gain: float = 0.02,
) -> nn.Module:
    """Factory for NEGCUT negative-sample networks.

    Parameters
    ----------
    nce_layers : list of int
        Encoder layer indices used for contrastive learning.
    netN : str
        ``neg_param`` | ``neg_gen`` | ``neg_gen_momentum`` (alias ``neg_gen_al``).
    num_patches : int
        Number of negative patches per layer.
    nc : int
        Feature dimension (``netF_nc``).
    """
    if netN == "neg_param":
        return NegativePlaceholder(nce_layers, num_patches=num_patches, nc=nc)
    if netN == "neg_gen":
        return NegativeGenerator(
            use_conv=True,
            num_patches=num_patches,
            nc=nc,
            init_type=init_type,
            init_gain=init_gain,
        )
    if netN in ("neg_gen_al", "neg_gen_momentum"):
        return NegativeGenerator(
            use_conv=False,
            num_patches=num_patches,
            nc=nc,
            init_type=init_type,
            init_gain=init_gain,
        )
    raise ValueError(f"Unknown negative generator type: {netN}")
