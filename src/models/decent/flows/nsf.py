# Copyright (c) 2026 EarthBridge Team.
# Credits: Decent from Xie et al. NeurIPS 2022 — https://github.com/Mid-Push/Decent

"""Neural Spline Flow (NSF) density estimator via optional ``nflows`` dependency."""

from __future__ import annotations

import torch
import torch.nn as nn


def _require_nflows():
    try:
        import nflows  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "NSF flow type requires the optional 'nflows' package. "
            "Install it with: pip install nflows"
        ) from exc


def _create_linear_transform(linear_transform_type, features):
    from nflows import transforms

    if linear_transform_type == "permutation":
        return transforms.RandomPermutation(features=features)
    if linear_transform_type == "lu":
        return transforms.CompositeTransform(
            [
                transforms.RandomPermutation(features=features),
                transforms.LULinear(features, identity_init=True),
            ]
        )
    if linear_transform_type == "svd":
        return transforms.CompositeTransform(
            [
                transforms.RandomPermutation(features=features),
                transforms.SVDLinear(features, num_householder=10, identity_init=True),
            ]
        )
    raise ValueError(linear_transform_type)


def _create_base_transform(
    base_transform_type,
    features,
    hidden_features,
    num_transform_blocks,
    dropout_probability,
    use_batch_norm,
    num_bins,
    tail_bound,
):
    from nflows import transforms

    if base_transform_type == "affine-autoregressive":
        return transforms.MaskedAffineAutoregressiveTransform(
            features=features,
            hidden_features=hidden_features,
            context_features=None,
            num_blocks=num_transform_blocks,
            use_residual_blocks=True,
            random_mask=False,
            activation=torch.relu,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
        )
    if base_transform_type == "quadratic-autoregressive":
        return transforms.MaskedPiecewiseQuadraticAutoregressiveTransform(
            features=features,
            hidden_features=hidden_features,
            context_features=None,
            num_bins=num_bins,
            tails="linear",
            tail_bound=tail_bound,
            num_blocks=num_transform_blocks,
            use_residual_blocks=True,
            random_mask=False,
            activation=torch.relu,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
        )
    if base_transform_type == "rq-autoregressive":
        return transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
            features=features,
            hidden_features=hidden_features,
            context_features=None,
            num_bins=num_bins,
            tails="linear",
            tail_bound=tail_bound,
            num_blocks=num_transform_blocks,
            use_residual_blocks=True,
            random_mask=False,
            activation=torch.relu,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
        )
    raise ValueError(base_transform_type)


def _create_transform(
    features,
    num_flow_steps,
    linear_transform_type="lu",
    base_transform_type="rq-autoregressive",
    hidden_features=256,
    num_transform_blocks=2,
    dropout_probability=0.25,
    use_batch_norm=0,
    num_bins=8,
    tail_bound=3,
):
    import torch
    from nflows import transforms

    return transforms.CompositeTransform(
        [
            transforms.CompositeTransform(
                [
                    transforms.BatchNorm(features),
                    _create_linear_transform(linear_transform_type, features),
                    _create_base_transform(
                        base_transform_type,
                        features,
                        hidden_features,
                        num_transform_blocks,
                        dropout_probability,
                        use_batch_norm,
                        num_bins,
                        tail_bound,
                    ),
                ]
            )
            for _ in range(num_flow_steps)
        ]
        + [_create_linear_transform(linear_transform_type, features)]
    )


class NSF(nn.Module):
    """Neural spline flow density model."""

    def __init__(self, input_dim: int, num_blocks: int):
        super().__init__()
        _require_nflows()
        from nflows import distributions, flows

        base_dist = distributions.StandardNormal(shape=[input_dim])
        transform = _create_transform(input_dim, num_blocks, base_transform_type="rq-autoregressive")
        self.flow = flows.Flow(transform, base_dist)

    def log_probs(self, x):
        return self.flow.log_prob(x)
