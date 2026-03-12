# Copyright (c) 2026 EarthBridge Team.
# Credits: ECSI (https://github.com/szhan311/ECSI) - diffusion bridge denoiser.

"""KarrasDenoiser for ECSI / DDBM bridge diffusion."""

from __future__ import annotations

import torch as th

from src.models.ecsi._utils.nn import append_dims, mean_flat
from src.models.ecsi.route import get_route


class KarrasDenoiser:
    """Bridge diffusion denoiser with Karras preconditioning."""

    def __init__(
        self,
        sigma_data: float = 0.5,
        sigma_min: float = 1e-4,
        sigma_max: float = 1.0,
        cov_xy: float = 0.0,
        rho: float = 7.0,
        image_size: int = 64,
        weight_schedule: str = "",
        pred_mode: str = "vp",
        loss_norm: str = "lpips",
        smooth: float = 0.0,
    ) -> None:
        self.sigma_0 = sigma_data
        self.sigma_1 = sigma_data + smooth
        self.sigma_01 = cov_xy
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max - 1e-4
        self.rho = rho
        self.smooth = smooth
        self.route = get_route(pred_mode)

        self.weight_schedule = weight_schedule
        self.pred_mode = pred_mode
        self.loss_norm = loss_norm
        self.lpips_loss = None
        if loss_norm == "lpips":
            try:
                from piq import LPIPS

                self.lpips_loss = LPIPS(replace_pooling=True, reduction="none")
            except ImportError:
                pass

        self.num_timesteps = 40
        self.image_size = image_size

    def _get_bridge_scalings(self, t: th.Tensor) -> tuple:
        alpha, _, beta, _, gamma, _ = self.route
        A = (
            alpha(t) ** 2 * self.sigma_0**2
            + beta(t) ** 2 * self.sigma_1**2
            + 2 * alpha(t) * beta(t) * self.sigma_01
            + gamma(t) ** 2
        )
        c_in = 1 / A**0.5
        c_skip = (alpha(t) * self.sigma_0**2 + beta(t) * self.sigma_01**2) / A
        c_out = (
            (
                beta(t) ** 2 * self.sigma_0**2 * self.sigma_1**2
                - beta(t) ** 2 * self.sigma_01**2
                + gamma(t) ** 2 * self.sigma_0**2
            )
            ** 0.5
            * c_in
        )
        return c_in, c_skip, c_out

    def _get_weightings(self, t: th.Tensor) -> th.Tensor:
        alpha, _, beta, _, gamma, _ = self.route
        if self.weight_schedule == "":
            return th.ones_like(t)
        if self.weight_schedule == "snr":
            return gamma(t) ** -2
        if self.weight_schedule == "karras":
            return gamma(t) ** -2 + self.sigma_0**-2
        if self.weight_schedule == "sibm":
            A = (
                alpha(t) ** 2 * self.sigma_0**2
                + beta(t) ** 2 * self.sigma_1**2
                + 2 * alpha(t) * beta(t) * self.sigma_01
                + gamma(t) ** 2
            )
            return A / (
                beta(t) ** 2 * self.sigma_0**2 * self.sigma_1**2
                - beta(t) ** 2 * self.sigma_01**2
                + gamma(t) ** 2 * self.sigma_0**2
            )
        return th.ones_like(t)

    def bridge_sample(
        self, x0: th.Tensor, x1: th.Tensor, t: th.Tensor
    ) -> th.Tensor:
        x1 = x1 + self.smooth * th.randn_like(x1)
        t = append_dims(t, x0.ndim)
        z = th.randn_like(x0)
        alpha, _, beta, _, gamma, _ = self.route
        return alpha(t) * x0 + beta(t) * x1 + gamma(t) * z

    def denoise(
        self, model: th.nn.Module, x_t: th.Tensor, t: th.Tensor, **model_kwargs
    ) -> th.Tensor:
        c_in, c_skip, c_out = [
            append_dims(x, x_t.ndim) for x in self._get_bridge_scalings(t)
        ]
        rescaled_t = 1000 * 0.25 * th.log(t + 1e-44)
        model_output = model(c_in * x_t, rescaled_t, **model_kwargs)
        return c_out * model_output + c_skip * x_t
