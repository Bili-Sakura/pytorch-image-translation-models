# Copyright (c) 2026 EarthBridge Team. Credits: AlignFlow (ermongroup).

import numpy as np
import torch.nn as nn


class RealNVPLoss(nn.Module):
    """NLL loss for RealNVP (negative log-likelihood)."""

    def __init__(self, k=256):
        super().__init__()
        self.k = k

    def forward(self, z, sldj):
        prior_ll = -0.5 * (z ** 2 + np.log(2 * np.pi))
        prior_ll = prior_ll.view(z.size(0), -1).sum(-1) - np.log(self.k) * np.prod(z.size()[1:])
        ll = prior_ll + sldj
        return -ll.mean()
