# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.
#
# SiD2 (Simpler Diffusion v2) credits:
#   Hoogeboom et al. (2025). Simpler Diffusion (SiD2).
#   CVPR 2025. https://arxiv.org/abs/2503.xxxxx

"""SiD2 Scheduler – exact continuous scheduler (t ∈ [0,1]) for pixel-space diffusion.

Implements the cosine-interpolated log-SNR schedule from the SiD2 paper (Appendix B).
Drop-in replacement for DDPMScheduler in training loops.
"""

from __future__ import annotations

import math

import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin

__all__ = ["SiD2Scheduler"]


class SiD2Scheduler(SchedulerMixin, ConfigMixin):
    """
    Exact SiD2 continuous scheduler (t ∈ [0,1]).
    Drop-in replacement for DDPMScheduler in training loops.
    """

    _compatibles = ["DDPMScheduler"]
    config_name = "sid2_scheduler_config.json"

    @register_to_config
    def __init__(
        self,
        image_size: int = 512,
        num_train_timesteps: int = 1000,  # only used for discrete sampling mode
        logsnr_min: float = -10.0,
        logsnr_max: float = 10.0,
    ):
        super().__init__()
        self.image_size = image_size

    def cosine_interpolated_logsnr(self, t: torch.Tensor) -> torch.Tensor:
        """Exact formula from SiD2 paper Appendix B."""
        log_change_high = math.log(self.image_size) - math.log(512)
        log_change_low = math.log(self.image_size) - math.log(32)
        b = math.atan(math.exp(-0.5 * self.config.logsnr_max))
        a = math.atan(math.exp(-0.5 * self.config.logsnr_min)) - b
        logsnr_cosine = -2.0 * torch.log(torch.tan(a * t + b))
        logsnr_high = logsnr_cosine + log_change_high
        logsnr_low = logsnr_cosine + log_change_low
        return (1 - t) * logsnr_high + t * logsnr_low

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,  # float t ∈ [0,1] or integer (auto-scaled)
    ) -> torch.Tensor:
        t = timesteps.to(original_samples.device).float()
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(original_samples.shape[0])
        if t.max() > 1.0:  # user passed integer timesteps
            t = t / self.config.num_train_timesteps

        logsnr = self.cosine_interpolated_logsnr(t).view(-1, 1, 1, 1)
        alpha_bar = 1.0 / (1.0 + torch.exp(-logsnr))
        sqrt_alpha = alpha_bar.sqrt()
        sqrt_one_minus_alpha = (1.0 - alpha_bar).sqrt()
        return sqrt_alpha * original_samples + sqrt_one_minus_alpha * noise

    def get_velocity(
        self,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """v = α√ * ε - √(1-α) * x (same as Diffusers v-prediction)."""
        t = timesteps.to(sample.device).float()
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(sample.shape[0])
        if t.max() > 1.0:
            t = t / self.config.num_train_timesteps

        logsnr = self.cosine_interpolated_logsnr(t).view(-1, 1, 1, 1)
        alpha_bar = 1.0 / (1.0 + torch.exp(-logsnr))
        return alpha_bar.sqrt() * noise - (1.0 - alpha_bar).sqrt() * sample
