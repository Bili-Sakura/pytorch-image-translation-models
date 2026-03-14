# Copyright (c) 2026 EarthBridge Team.
# Credits: AlignFlow (Grover et al., AAAI 2020) https://github.com/ermongroup/alignflow

"""Initialization utilities."""

import torch.nn as nn


def init_model(model, init_method="normal"):
    """Initialize model parameters."""
    if init_method == "normal":
        model.apply(_normal_init)
    elif init_method == "xavier":
        model.apply(_xavier_init)
    else:
        raise NotImplementedError("Invalid weights initializer: {}".format(init_method))


def _normal_init(model):
    class_name = model.__class__.__name__
    if hasattr(model, "weight") and model.weight is not None:
        if "Conv" in class_name:
            nn.init.normal_(model.weight.data, 0.0, 0.02)
        elif "Linear" in class_name:
            nn.init.normal_(model.weight.data, 0.0, 0.02)
        elif "BatchNorm" in class_name:
            nn.init.normal_(model.weight.data, 1.0, 0.02)
            nn.init.constant_(model.bias.data, 0.0)


def _xavier_init(model):
    class_name = model.__class__.__name__
    if hasattr(model, "weight") and model.weight is not None:
        if "Conv" in class_name:
            nn.init.xavier_normal_(model.weight.data, gain=0.02)
        elif "Linear" in class_name:
            nn.init.xavier_normal_(model.weight.data, gain=0.02)
        elif "BatchNorm" in class_name:
            nn.init.normal_(model.weight.data, 1.0, 0.02)
            nn.init.constant_(model.bias.data, 0.0)
