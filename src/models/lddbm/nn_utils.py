# Copyright (c) 2026 EarthBridge Team.
# Credits: Berman et al., NeurIPS 2025 - Bosch Research MDT.

"""Neural network utilities for MDT."""

import torch as th


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]


def append_zero(x):
    return th.cat([x, x.new_zeros([1])])
