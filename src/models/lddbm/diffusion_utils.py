# Copyright (c) 2026 EarthBridge Team.
# Credits: Berman et al., NeurIPS 2025 - Bosch Research MDT.

"""Diffusion utility functions for MDT bridge."""

import torch as th


def vp_logsnr(t, beta_d, beta_min):
    t = th.as_tensor(t)
    return -th.log((0.5 * beta_d * (t**2) + beta_min * t).exp() - 1)


def vp_logs(t, beta_d, beta_min):
    t = th.as_tensor(t)
    return -0.25 * t**2 * (beta_d) - 0.5 * t * beta_min
