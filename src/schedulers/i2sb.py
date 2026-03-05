# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""I2SB Scheduler – noise schedule and sampling for Image-to-Image Schrödinger Bridge."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class I2SBSchedulerOutput:
    """Output of a single scheduler step."""

    prev_sample: torch.Tensor


class I2SBScheduler:
    """Noise scheduler for Image-to-Image Schrödinger Bridge (I2SB).

    Implements a *symmetric* beta schedule and the forward / reverse
    transition kernels of the diffusion bridge.

    Parameters
    ----------
    interval : int
        Number of discrete time-steps.
    beta_max : float
        Peak value of the beta schedule (reached at the midpoint).
    """

    def __init__(self, interval: int = 1000, beta_max: float = 0.3) -> None:
        self.interval = interval

        # Symmetric linear beta schedule
        half = interval // 2
        betas_inc = torch.linspace(1e-4, beta_max, half, dtype=torch.float64)
        self.betas = torch.cat([betas_inc, betas_inc.flip(0)]).float()

        # Cumulative variance
        alphas = self.betas.double().cumsum(dim=0)
        alpha_T = alphas[-1]

        # Forward standard deviation: σ_fwd(t) = √α_t
        self.std_fwd = alphas.sqrt().float()

        # Coefficient for x_0 in the bridge mean:
        #   μ_x0(t) = (α_T − α_t) / α_T
        self.mu_x0 = ((alpha_T - alphas) / alpha_T).float()

        # Bridge standard deviation:
        #   σ_bridge(t) = √(α_t · (α_T − α_t) / α_T)
        self.std_bridge = (alphas * (alpha_T - alphas) / alpha_T).sqrt().float()

        self._alpha = alphas.float()
        self._alpha_T = alpha_T.float()

        self.timesteps: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Forward (training) helpers
    # ------------------------------------------------------------------

    def q_sample(
        self,
        step: torch.Tensor,
        x0: torch.Tensor,
        x1: torch.Tensor,
        ot_ode: bool = False,
    ) -> torch.Tensor:
        """Sample from the bridge process ``q(x_t | x_0, x_1)``.

        Parameters
        ----------
        step : Tensor ``[B]``
            Timestep indices.
        x0 : Tensor ``[B, C, H, W]``
            Source images.
        x1 : Tensor ``[B, C, H, W]``
            Target images.
        ot_ode : bool
            If ``True``, return the deterministic (ODE) interpolant
            without noise.
        """
        dims = [step.shape[0]] + [1] * (x0.dim() - 1)
        mu_x0_t = self.mu_x0[step].view(*dims).to(x0.device)

        mu = mu_x0_t * x0 + (1 - mu_x0_t) * x1

        if ot_ode:
            return mu

        std_t = self.std_bridge[step].view(*dims).to(x0.device)
        return mu + std_t * torch.randn_like(x0)

    def compute_label(
        self,
        step: torch.Tensor,
        x0: torch.Tensor,
        xt: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the training target (noise-prediction label).

        Returns ``(x_t − x_0) / σ_fwd(t)``.
        """
        dims = [step.shape[0]] + [1] * (x0.dim() - 1)
        std_t = self.std_fwd[step].view(*dims).to(x0.device)
        return (xt - x0) / std_t

    def compute_pred_x0(
        self,
        step: torch.Tensor | int,
        xt: torch.Tensor,
        net_out: torch.Tensor,
    ) -> torch.Tensor:
        """Convert the network noise prediction back to an ``x_0`` estimate.

        Returns ``x_t − σ_fwd(t) · net_out``.
        """
        if isinstance(step, int):
            step = torch.tensor([step])
        dims = [step.shape[0]] + [1] * (xt.dim() - 1)
        std_t = self.std_fwd[step].view(*dims).to(xt.device)
        return xt - std_t * net_out

    # ------------------------------------------------------------------
    # Reverse (inference) helpers
    # ------------------------------------------------------------------

    def p_posterior(
        self,
        prev_step: int,
        step: int,
        x_n: torch.Tensor,
        x0: torch.Tensor,
        ot_ode: bool = False,
    ) -> torch.Tensor:
        """Compute the reverse posterior ``q(x_s | x_t, x_0)``.

        Parameters
        ----------
        prev_step : int
            Target (earlier) timestep *s*.
        step : int
            Current timestep *t*  (``s < t``).
        x_n : Tensor ``[B, C, H, W]``
            Current sample ``x_t``.
        x0 : Tensor ``[B, C, H, W]``
            Predicted clean image.
        """
        alpha_t = self._alpha[step].to(x_n.device)
        alpha_s = self._alpha[prev_step].to(x_n.device)

        coef_xt = alpha_s / alpha_t
        mu = coef_xt * x_n + (1 - coef_xt) * x0

        if ot_ode:
            return mu

        var = alpha_s * (alpha_t - alpha_s) / alpha_t
        return mu + var.sqrt() * torch.randn_like(x_n)

    def set_timesteps(self, nfe: int) -> None:
        """Set the inference timestep schedule.

        Parameters
        ----------
        nfe : int
            Number of function evaluations (denoising steps).
            The resulting ``timesteps`` tensor has ``nfe + 1`` entries.
        """
        self.timesteps = torch.linspace(self.interval - 1, 0, nfe + 1).long()

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        prev_timestep: int,
        sample: torch.Tensor,
    ) -> I2SBSchedulerOutput:
        """Perform one denoising step.

        Parameters
        ----------
        model_output : Tensor
            Network prediction at the current timestep.
        timestep : int
            Current timestep index.
        prev_timestep : int
            Target (earlier) timestep index.
        sample : Tensor
            Current noisy sample ``x_t``.
        """
        pred_x0 = self.compute_pred_x0(torch.tensor([timestep]), sample, model_output)
        prev_sample = self.p_posterior(prev_timestep, timestep, sample, pred_x0)
        return I2SBSchedulerOutput(prev_sample=prev_sample)
