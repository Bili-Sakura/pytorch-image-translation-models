# Copyright (c) 2026 EarthBridge Team.
# Credits: ECSI (https://github.com/szhan311/ECSI) - Heun/stochastic sampling.

"""ECSI sampling: Heun and stochastic samplers for diffusion bridge models."""

from __future__ import annotations

import torch as th

from src.models.ecsi.route import get_route


def get_sigmas_karras(
    n: int,
    sigma_min: float,
    sigma_max: float,
    rho: float = 7.0,
    device: str | th.device = "cpu",
) -> th.Tensor:
    """Karras sigma schedule. Matches k_diffusion.sampling.get_sigmas_karras."""
    ramp = th.linspace(1, 0, n, device=device)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return th.cat([sigmas, sigmas.new_zeros(1)])


def get_scaling(k: float = 0.0):
    s = lambda t: th.where(t < 0.5, k * t + 1, k * (1 - t) + 1)
    s_deriv = lambda t: th.where(t < 0.5, k, -k)
    return s, s_deriv


def update_route(alpha, alpha_deriv, beta, beta_deriv, gamma, gamma_deriv, s, s_deriv):
    alpha_new = s * alpha
    beta_new = s * beta
    gamma_new = s * gamma
    alpha_deriv_new = s_deriv * alpha + s * alpha_deriv
    beta_deriv_new = s_deriv * beta + s * beta_deriv
    gamma_deriv_new = s_deriv * gamma + s * gamma_deriv
    return alpha_new, alpha_deriv_new, beta_new, beta_deriv_new, gamma_new, gamma_deriv_new


def to_d_sky(x, denoised, x_T, alpha, alpha_deriv, beta, beta_deriv, gamma, gamma_deriv, s, s_deriv, stochastic=False):
    """Converts a denoiser output to a Karras ODE derivative."""
    score = (alpha * denoised + beta * x_T - x / s) / gamma**2
    score = score / s
    alpha, alpha_deriv, beta, beta_deriv, gamma, gamma_deriv = update_route(
        alpha, alpha_deriv, beta, beta_deriv, gamma, gamma_deriv, s, s_deriv
    )
    if gamma == 0:
        return alpha_deriv * denoised + beta_deriv * x_T, 0
    if stochastic is False:
        diffusion_term = (
            alpha_deriv / alpha * x
            + (beta_deriv - alpha_deriv / alpha * beta) * x_T
            - (gamma * gamma_deriv - alpha_deriv / alpha * gamma**2) * score
        )
        return diffusion_term, 0
    diffusion_term = (
        alpha_deriv / alpha * x
        + (beta_deriv - alpha_deriv / alpha * beta) * x_T
        - 2 * (gamma * gamma_deriv - alpha_deriv / alpha * gamma**2) * score
    )
    drift_term = (2 * (gamma * gamma_deriv - alpha_deriv / alpha * gamma**2)) ** 0.5
    return diffusion_term, drift_term


def to_d_stoch(x, x0_hat, x_T, alpha, alpha_deriv, beta, beta_deriv, gamma, gamma_deriv, epsilon):
    """Converts a denoiser output to a stochastic derivative."""
    z_hat = (x - alpha * x0_hat - beta * x_T) / gamma
    if gamma == 0:
        return alpha_deriv * x0_hat + beta_deriv * x_T, 0
    diffusion_term = (
        alpha_deriv * x0_hat + beta_deriv * x_T + (gamma_deriv + epsilon / gamma) * z_hat
    )
    drift_term = (2 * epsilon) ** 0.5
    return diffusion_term, drift_term


@th.no_grad()
def sample_heun(denoiser, x, sigmas, route, progress=False, callback=None, churn_step_ratio=0.0, route_scaling=0, smooth=0):
    """Heun steps (Algorithm 2) from Karras et al. (2022)."""
    x_T = x
    path = [x.detach().cpu()]
    x0_est = [x.detach().cpu()]
    s_in = x.new_ones([x.shape[0]])
    indices = list(range(len(sigmas) - 1))
    if progress:
        from tqdm.auto import tqdm
        indices = tqdm(indices)

    s = lambda t: 1
    s_deriv = lambda t: 0

    alpha, alpha_deriv, beta, beta_deriv, gamma, gamma_deriv = route
    for i in indices:
        if churn_step_ratio > 0:
            sigma_hat = (sigmas[i + 1] - sigmas[i]) * churn_step_ratio + sigmas[i]
            denoised = denoiser(x / s(sigmas[i]), sigmas[i] * s_in, x_T)
            x0_est.append(denoised.detach().cpu())
            diffusion_term, drift_term = to_d_sky(
                x, denoised, x_T,
                alpha(sigmas[i]), alpha_deriv(sigmas[i]),
                beta(sigmas[i]), beta_deriv(sigmas[i]),
                gamma(sigmas[i]), gamma_deriv(sigmas[i]),
                s(sigmas[i]), s_deriv(sigmas[i]),
                stochastic=True,
            )
            dt = sigma_hat - sigmas[i]
            x = x + diffusion_term * dt + th.randn_like(x) * (dt.abs() ** 0.5) * drift_term
            path.append(x.detach().cpu())
        else:
            sigma_hat = sigmas[i]

        if churn_step_ratio < 1:
            denoised = denoiser(x / s(sigma_hat), sigma_hat * s_in, x_T)
            x0_est.append(denoised.detach().cpu())
            d, _ = to_d_sky(
                x, denoised, x_T,
                alpha(sigma_hat), alpha_deriv(sigma_hat),
                beta(sigma_hat), beta_deriv(sigma_hat),
                gamma(sigma_hat), gamma_deriv(sigma_hat),
                s(sigma_hat), s_deriv(sigma_hat),
            )
            if callback is not None:
                callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "denoised": denoised})
            dt = sigmas[i + 1] - sigma_hat
            if sigmas[i + 1] == 0:
                x = x + d * dt
            else:
                x_2 = x + d * dt
                denoised_2 = denoiser(x_2 / s(sigmas[i + 1]), sigmas[i + 1] * s_in, x_T)
                x0_est.append(denoised_2.detach().cpu())
                d_2, _ = to_d_sky(
                    x_2, denoised_2, x_T,
                    alpha(sigmas[i + 1]), alpha_deriv(sigmas[i + 1]),
                    beta(sigmas[i + 1]), beta_deriv(sigmas[i + 1]),
                    gamma(sigmas[i + 1]), gamma_deriv(sigmas[i + 1]),
                    s(sigmas[i + 1]), s_deriv(sigmas[i + 1]),
                )
                d_prime = (d + d_2) / 2
                x = x + d_prime * dt
            path.append(x.detach().cpu())
    return x, path, x0_est


@th.no_grad()
def sample_stoch(denoiser, x, sigmas, route, progress=False, callback=None, churn_step_ratio=0.0, route_scaling=0, smooth=0):
    """Stochastic sampler for ECSI bridge diffusion."""
    x_T = x
    x = x + smooth * th.randn_like(x)
    x_T_s = x

    s_in = x.new_ones([x.shape[0]])
    indices = list(range(len(sigmas) - 1))
    if progress:
        from tqdm.auto import tqdm
        indices = tqdm(indices)

    alpha, alpha_deriv, beta, beta_deriv, gamma, gamma_deriv = route
    epsilon = lambda t: churn_step_ratio * (gamma(t) * gamma_deriv(t) - alpha_deriv(t) / alpha(t) * gamma(t) ** 2)
    path = [x.detach().cpu()]
    x0_est = [x.detach().cpu()]

    for i in indices:
        x0_hat = denoiser(x, sigmas[i] * s_in, x_T)
        x0_est.append(x0_hat.detach().cpu())
        dt = sigmas[i + 1] - sigmas[i]
        if i >= len(indices) - 2:
            x = (
                alpha(sigmas[i + 1]) * x0_hat
                + beta(sigmas[i + 1]) * x_T_s
                + (gamma(sigmas[i + 1]) / gamma(sigmas[i])) * (x - alpha(sigmas[i]) * x0_hat - beta(sigmas[i]) * x_T_s)
            )
        else:
            diffusion_term, drift_term = to_d_stoch(
                x, x0_hat, x_T_s,
                alpha(sigmas[i]), alpha_deriv(sigmas[i]),
                beta(sigmas[i]), beta_deriv(sigmas[i]),
                gamma(sigmas[i]), gamma_deriv(sigmas[i]),
                epsilon(sigmas[i]),
            )
            x = x + diffusion_term * dt + th.randn_like(x) * (dt.abs() ** 0.5) * drift_term
        path.append(x.detach().cpu())

    return x, path, x0_est


def karras_sample(
    diffusion,
    model,
    x_T,
    x_0,
    route,
    steps,
    clip_denoised=True,
    progress=False,
    callback=None,
    model_kwargs=None,
    device=None,
    sigma_min=0.002,
    sigma_max=1.0,
    rho=7.0,
    sampler="heun",
    churn_step_ratio=0.0,
    route_scaling=0,
    guidance=1,
    smooth=0,
):
    """Run ECSI bridge diffusion sampling (Heun or stochastic)."""
    if model_kwargs is None:
        model_kwargs = {}
    model_kwargs.setdefault("xT", x_T)

    assert sampler in ["heun", "stoch"]
    sigmas = get_sigmas_karras(steps, sigma_min + 1e-4, sigma_max - 1e-4, rho, device=x_T.device)

    def denoiser(x_t, sigma, x_T_opt=None):
        denoised = diffusion.denoise(model, x_t, sigma, **model_kwargs)
        if clip_denoised:
            denoised = denoised.clamp(-1, 1)
        return denoised

    sampler_args = dict(
        churn_step_ratio=churn_step_ratio,
        route_scaling=route_scaling,
        smooth=smooth,
    )
    if sampler == "heun":
        x_0, path, x0_est = sample_heun(
            denoiser, x_T, sigmas, route, progress=progress, callback=callback, **sampler_args
        )
    else:
        x_0, path, x0_est = sample_stoch(
            denoiser, x_T, sigmas, route, progress=progress, callback=callback, **sampler_args
        )

    return x_0.clamp(-1, 1), [p.clamp(-1, 1) for p in path], [p.clamp(-1, 1) for p in x0_est]
