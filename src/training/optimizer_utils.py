# Copyright (c) 2026 EarthBridge Team.
# Credits: Optimizer factory learned from 4th-MAVIC-T (src/utils/training_utils.py).

"""Optimizer factory for image-to-image translation training.

Provides create_optimizer to build Prodigy, Muon, AdamW, or Adam from a string
identifier, mirroring the 4th-MAVIC-T training utilities.
"""

from __future__ import annotations

import logging
from typing import Iterable

import torch

logger = logging.getLogger(__name__)


def _load_muon_optimizers():
    """Load Muon optimizer classes, raising ImportError when unavailable."""
    try:
        import torch.distributed as dist
        import muon as muon_module
    except ImportError as exc:
        raise ImportError(
            "Muon optimizer requires the Muon package (https://github.com/KellerJordan/Muon). "
            "Install it with: pip install git+https://github.com/KellerJordan/Muon.git"
        ) from exc
    if not hasattr(muon_module, "Muon") or not hasattr(muon_module, "SingleDeviceMuon"):
        raise ImportError(
            "The installed 'muon' package does not expose Muon optimizers. "
            "Install the optimizer build with: pip install git+https://github.com/KellerJordan/Muon.git"
        )
    return muon_module.Muon, muon_module.SingleDeviceMuon


def create_optimizer(
    params: Iterable[torch.nn.Parameter],
    optimizer_type: str = "adamw",
    lr: float = 2e-4,
    weight_decay: float = 0.01,
    betas: tuple[float, float] = (0.9, 0.999),
    prodigy_d0: float = 1e-6,
) -> torch.optim.Optimizer:
    """Create an optimizer from a string identifier.

    Parameters
    ----------
    params : iterable of torch.nn.Parameter
        Model parameters to optimise.
    optimizer_type : str
        One of: ``"prodigy"`` | ``"muon"`` | ``"adamw"`` | ``"adam"``.
    lr : float
        Learning rate. For Prodigy the recommended value is ``1.0``; for
        Adam/AdamW typically ``2e-4`` for GAN training.
    weight_decay : float
        Weight-decay coefficient. For AdamW typically 0.01.
    betas : tuple
        Beta coefficients for Adam-family optimizers; also used as momentum
        for Muon (first component).
    prodigy_d0 : float
        Prodigy d0 parameter (initial estimate of D). Default is 1e-6.

    Returns
    -------
    torch.optim.Optimizer
    """
    name = optimizer_type.lower()
    if name == "prodigy":
        try:
            from prodigyopt import Prodigy
        except ImportError:
            raise ImportError(
                "prodigyopt is required for the Prodigy optimizer. "
                "Install it with: pip install prodigyopt"
            )
        return Prodigy(
            params,
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            d0=prodigy_d0,
        )
    elif name == "muon":
        Muon, SingleDeviceMuon = _load_muon_optimizers()
        params_list = list(params)
        try:
            import torch.distributed as dist
            use_distributed = (
                dist.is_available()
                and dist.is_initialized()
                and dist.get_world_size() > 1
            )
        except (RuntimeError, AttributeError) as exc:
            logger.debug("Muon optimizer using single-device fallback: %s", exc)
            use_distributed = False
        momentum = betas[0] if betas else 0.95
        if use_distributed:
            return Muon(
                params_list, lr=lr, weight_decay=weight_decay, momentum=momentum
            )
        return SingleDeviceMuon(
            params_list, lr=lr, weight_decay=weight_decay, momentum=momentum
        )
    elif name == "adamw":
        return torch.optim.AdamW(
            params, lr=lr, weight_decay=weight_decay, betas=betas
        )
    elif name == "adam":
        return torch.optim.Adam(
            params, lr=lr, weight_decay=weight_decay, betas=betas
        )
    raise ValueError(f"Unknown optimizer_type: {optimizer_type!r}")

