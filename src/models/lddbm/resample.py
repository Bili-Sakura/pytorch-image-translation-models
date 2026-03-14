# Copyright (c) 2025 Copyright holder of the paper "Towards General Modality Translation with Contrastive and Predictive Latent Diffusion Bridge" submitted to NeurIPS 2025 for review.
# All rights reserved.

"""Schedule samplers for MDT bridge training."""

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.distributed as dist
from scipy.stats import norm


def create_named_schedule_sampler(name, diffusion):
    if name == "uniform":
        return UniformSampler(diffusion)
    elif name == "real-uniform":
        return RealUniformSampler(diffusion)
    elif name == "loss-second-moment":
        return LossSecondMomentResampler(diffusion)
    elif name == "lognormal":
        return LogNormalSampler(diffusion)
    else:
        raise NotImplementedError(f"unknown schedule sampler: {name}")


class ScheduleSampler(ABC):
    @abstractmethod
    def weights(self):
        pass

    def sample(self, batch_size, device):
        w = self.weights()
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = torch.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = torch.from_numpy(weights_np).float().to(device)
        return indices, weights


class UniformSampler(ScheduleSampler):
    def __init__(self, diffusion):
        self.diffusion = diffusion
        self._weights = np.ones([diffusion.num_timesteps])

    def weights(self):
        return self._weights


class RealUniformSampler:
    """Sample sigma uniformly in [sigma_min, sigma_max]."""

    def __init__(self, diffusion):
        self.diffusion = diffusion
        self.sigma_max = diffusion.sigma_max
        self.sigma_min = diffusion.sigma_min

    def sample(self, batch_size, device):
        ts = (
            torch.rand(batch_size, device=device)
            * (self.sigma_max - self.sigma_min)
            + self.sigma_min
        )
        return ts, torch.ones_like(ts)


class LossAwareSampler(ScheduleSampler):
    def update_with_local_losses(self, local_ts, local_losses):
        batch_sizes = [
            torch.tensor([0], dtype=torch.int32, device=local_ts.device)
            for _ in range(dist.get_world_size())
        ]
        dist.all_gather(
            batch_sizes,
            torch.tensor([len(local_ts)], dtype=torch.int32, device=local_ts.device),
        )
        batch_sizes = [x.item() for x in batch_sizes]
        max_bs = max(batch_sizes)
        timestep_batches = [torch.zeros(max_bs).to(local_ts) for bs in batch_sizes]
        loss_batches = [torch.zeros(max_bs).to(local_losses) for bs in batch_sizes]
        dist.all_gather(timestep_batches, local_ts)
        dist.all_gather(loss_batches, local_losses)
        timesteps = [
            x.item() for y, bs in zip(timestep_batches, batch_sizes) for x in y[:bs]
        ]
        losses = [x.item() for y, bs in zip(loss_batches, batch_sizes) for x in y[:bs]]
        self.update_with_all_losses(timesteps, losses)

    @abstractmethod
    def update_with_all_losses(self, ts, losses):
        pass


class LossSecondMomentResampler(LossAwareSampler):
    def __init__(self, diffusion, history_per_term=10, uniform_prob=0.001):
        self.diffusion = diffusion
        self.history_per_term = history_per_term
        self.uniform_prob = uniform_prob
        self._loss_history = np.zeros(
            [diffusion.num_timesteps, history_per_term], dtype=np.float64
        )
        self._loss_counts = np.zeros([diffusion.num_timesteps], dtype=np.int_)

    def weights(self):
        if not self._warmed_up():
            return np.ones([self.diffusion.num_timesteps], dtype=np.float64)
        weights = np.sqrt(np.mean(self._loss_history**2, axis=-1))
        weights /= np.sum(weights)
        weights *= 1 - self.uniform_prob
        weights += self.uniform_prob / len(weights)
        return weights

    def update_with_all_losses(self, ts, losses):
        for t, loss in zip(ts, losses):
            if self._loss_counts[t] == self.history_per_term:
                self._loss_history[t, :-1] = self._loss_history[t, 1:]
                self._loss_history[t, -1] = loss
            else:
                self._loss_history[t, self._loss_counts[t]] = loss
                self._loss_counts[t] += 1

    def _warmed_up(self):
        return (self._loss_counts == self.history_per_term).all()


class LogNormalSampler:
    def __init__(self, diffusion, p_mean=-1.2, p_std=1.2, even=False):
        self.p_mean = p_mean
        self.p_std = p_std
        self.even = even
        if self.even:
            self.inv_cdf = lambda x: norm.ppf(x, loc=p_mean, scale=p_std)
            self.rank, self.size = dist.get_rank(), dist.get_world_size()

    def sample(self, bs, device):
        if self.even:
            start_i, end_i = self.rank * bs, (self.rank + 1) * bs
            global_batch_size = self.size * bs
            locs = (np.arange(start_i, end_i) + np.random.rand(bs)) / global_batch_size
            log_sigmas = torch.tensor(
                self.inv_cdf(locs), dtype=torch.float32, device=device
            )
        else:
            log_sigmas = self.p_mean + self.p_std * torch.randn(bs, device=device)
        sigmas = torch.exp(log_sigmas)
        weights = torch.ones_like(sigmas)
        return sigmas, weights
