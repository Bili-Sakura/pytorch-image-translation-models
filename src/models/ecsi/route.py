# Copyright (c) 2026 EarthBridge Team.
# Adapted from ECSI: https://github.com/szhan311/ECSI
# Credits: Built on DDBM and open-source diffusion bridge models.

"""Route functions for ECSI bridge diffusion (VP, VE, linear)."""

from __future__ import annotations

import itertools

import numpy as np
import torch as th


def get_route(pred_mode: str):
    """Return alpha, alpha_deriv, beta, beta_deriv, gamma, gamma_deriv for the given pred_mode."""
    if pred_mode.startswith("vp"):
        return VP_route()
    if pred_mode.startswith("ve"):
        return VE_route()
    if pred_mode.startswith("linear"):
        parts = ["".join(g) for k, g in itertools.groupby(pred_mode, str.isalpha)]
        gamma_max = float(parts[1])
        return linear_route(gamma_max=gamma_max)
    raise NotImplementedError(f"pred_mode {pred_mode} not supported")


def linear_route(gamma_max: float = 1):
    alpha = lambda t: 1 - t
    alpha_deriv = lambda t: -th.ones_like(t)
    beta = lambda t: t
    beta_deriv = lambda t: th.ones_like(t)
    gamma = lambda t: gamma_max * 2 * (t * (1 - t)) ** 0.5
    gamma_deriv = lambda t: gamma_max * 2 * (1 - 2 * t) / (2 * (t * (1 - t)) ** 0.5)
    return alpha, alpha_deriv, beta, beta_deriv, gamma, gamma_deriv


def VP_route(beta_d: float = 2.0, beta_min: float = 0.1, sigma_max: float = 1.0):
    sigma = lambda t: (np.e ** (0.5 * beta_d * (t**2) + beta_min * t) - 1) ** 0.5
    sigma_deriv = lambda t: 0.5 * (beta_min + beta_d * t) * (sigma(t) + 1 / sigma(t))
    s = lambda t: (1 + sigma(t) ** 2).rsqrt()
    s_deriv = lambda t: -sigma(t) * sigma_deriv(t) * (s(t) ** 3)
    sigma_T = sigma(th.as_tensor(sigma_max))
    s_T = s(th.as_tensor(sigma_max))
    alpha = lambda t: s(t) * (1 - sigma(t) ** 2 / sigma_T**2)
    alpha_deriv = lambda t: (
        s_deriv(t) * (1 - sigma(t) ** 2 / sigma_T**2)
        - s(t) * 2 * sigma(t) * sigma_deriv(t) / sigma_T**2
    )
    beta = lambda t: s(t) * sigma(t) ** 2 / (s_T * sigma_T**2)
    beta_deriv = lambda t: (
        s_deriv(t) * sigma(t) ** 2 + 2 * s(t) * sigma(t) * sigma_deriv(t)
    ) / (s_T * sigma_T**2)
    gamma = lambda t: sigma(t) * s(t) * (1 - sigma(t) ** 2 / sigma_T**2) ** 0.5
    gamma_deriv = lambda t: (
        s(t)
        * (
            (sigma_deriv(t) * (sigma_T**2 - 2 * sigma(t) ** 2))
            / (sigma_T * (sigma_T**2 - sigma(t) ** 2) ** 0.5)
        )
        + s_deriv(t) * sigma(t) * (1 - sigma(t) ** 2 / sigma_T**2) ** 0.5
    )
    return alpha, alpha_deriv, beta, beta_deriv, gamma, gamma_deriv


def VE_route(sigma_max: float = 1.0):
    alpha = lambda t: 1 - (t**2) / (sigma_max**2)
    alpha_deriv = lambda t: -2 * t / (sigma_max**2)
    beta = lambda t: (t**2) / (sigma_max**2)
    beta_deriv = lambda t: 2 * t / (sigma_max**2)
    gamma = lambda t: ((t**2) * alpha(t)).sqrt()
    gamma_deriv = lambda t: alpha(t).sqrt() + (
        alpha_deriv(t) * t / (2 * alpha(t).sqrt())
    )
    return alpha, alpha_deriv, beta, beta_deriv, gamma, gamma_deriv
