# Copyright (c) 2026 EarthBridge Team.
# Credits: AlignFlow (Grover et al., AAAI 2020) https://github.com/ermongroup/alignflow

"""Normalization utilities."""

import functools
import torch
import torch.nn as nn


def get_norm_layer(norm_type="instance"):
    if norm_type == "batch":
        return functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == "instance":
        return functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == "group":
        return functools.partial(nn.GroupNorm, num_groups=16)
    else:
        raise NotImplementedError("Invalid normalization type: {}".format(norm_type))


def get_param_groups(net, weight_decay, norm_suffix="weight_g", verbose=False):
    """Get parameter groups for optimizer with optional weight decay on normalized params."""
    norm_params = []
    unnorm_params = []
    for n, p in net.named_parameters():
        if n.endswith(norm_suffix):
            norm_params.append(p)
        else:
            unnorm_params.append(p)
    return [
        {"name": "normalized", "params": norm_params, "weight_decay": weight_decay},
        {"name": "unnormalized", "params": unnorm_params},
    ]


class WNConv2d(nn.Module):
    """Weight-normalized 2d convolution."""

    def __init__(self, in_channels, out_channels, kernel_size, padding, bias=True):
        super().__init__()
        self.conv = nn.utils.weight_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)
        )

    def forward(self, x):
        return self.conv(x)
