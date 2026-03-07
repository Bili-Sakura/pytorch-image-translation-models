# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""UNSB scheduler for multi-step Schrödinger Bridge sampling.

Provides :class:`UNSBScheduler` which computes the non-uniform time schedule
and stochastic interpolation dynamics used by the UNSB method.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from diffusers.utils import BaseOutput


@dataclass
class UNSBSchedulerOutput(BaseOutput):
    """Output of a single UNSB scheduler step.

    Attributes
    ----------
    prev_sample : torch.Tensor
        Interpolated noisy sample ready for the next generator call.
    """

    prev_sample: torch.Tensor


class UNSBScheduler:
    """Scheduler for the UNSB (Unpaired Neural Schrödinger Bridge) method.

    Computes the non-uniform time schedule with harmonic spacing and
    implements the stochastic interpolation dynamics used to propagate
    samples through the bridge.

    Parameters
    ----------
    num_timesteps : int
        Number of refinement steps (NFE).
    tau : float
        Entropy parameter controlling stochasticity of the bridge dynamics.
        Smaller values produce more deterministic trajectories.
    """

    def __init__(
        self,
        num_timesteps: int = 5,
        tau: float = 0.01,
    ) -> None:
        self.num_timesteps = num_timesteps
        self.tau = tau
        self._times: torch.Tensor | None = None

    @property
    def times(self) -> torch.Tensor:
        """Return the precomputed time schedule as a 1-D float tensor."""
        if self._times is None:
            self._times = self._build_time_schedule(self.num_timesteps)
        return self._times

    @staticmethod
    def _build_time_schedule(num_timesteps: int) -> torch.Tensor:
        """Build the non-uniform (harmonic) time schedule.

        The schedule uses increments ``[0, 1, 1/2, 1/3, …]``, accumulates
        them, normalises to ``[0, 1]``, then shifts to ``[0.5*max, max]``
        and prepends a zero.  This places more temporal resolution near the
        end of the trajectory where refinement matters most.
        """
        incs = np.array([0.0] + [1.0 / (i + 1) for i in range(num_timesteps - 1)])
        times = np.cumsum(incs)
        if times[-1] > 0:
            times = times / times[-1]
        times = 0.5 * times[-1] + 0.5 * times
        times = np.concatenate([np.zeros(1), times])
        return torch.tensor(times, dtype=torch.float32)

    def step(
        self,
        model_output: torch.Tensor,
        timestep_idx: int,
        sample: torch.Tensor,
    ) -> UNSBSchedulerOutput:
        """Perform one stochastic interpolation step.

        Given the generator output ``model_output`` at step ``t``, compute
        the interpolated noisy sample for step ``t + 1``.

        Parameters
        ----------
        model_output : Tensor (B, C, H, W)
            Generator prediction at the current timestep (``X_{t+1}``).
        timestep_idx : int
            Current timestep index (0-based).
        sample : Tensor (B, C, H, W)
            Current noisy sample (``X_t``).

        Returns
        -------
        UNSBSchedulerOutput
            Contains the interpolated sample for the next step.
        """
        times = self.times.to(sample.device)
        t_next = timestep_idx + 1

        if t_next >= len(times) - 1:
            # Last step — return the model output directly
            return UNSBSchedulerOutput(prev_sample=model_output)

        delta = times[t_next] - times[timestep_idx]
        denom = times[-1] - times[timestep_idx]
        inter = (delta / denom).reshape(-1, 1, 1, 1)
        scale = (delta * (1 - delta / denom)).reshape(-1, 1, 1, 1)

        noise = torch.randn_like(sample)
        prev_sample = (
            (1 - inter) * sample
            + inter * model_output.detach()
            + (scale * self.tau).sqrt() * noise
        )
        return UNSBSchedulerOutput(prev_sample=prev_sample)
