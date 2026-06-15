# Copyright (c) 2026 EarthBridge Team.
# Credits: Decent from Xie et al. NeurIPS 2022 — https://github.com/Mid-Push/Decent

"""Masked Autoregressive Flow (MAF) density estimators."""

from __future__ import annotations

import math

import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F


def _create_masks(input_size, hidden_size, n_hidden, input_order="sequential", input_degrees=None):
    degrees = []
    if input_order == "sequential":
        degrees += [torch.arange(input_size)] if input_degrees is None else [input_degrees]
        for _ in range(n_hidden + 1):
            degrees += [torch.arange(hidden_size) % (input_size - 1)]
        degrees += (
            [torch.arange(input_size) % input_size - 1]
            if input_degrees is None
            else [input_degrees % input_size - 1]
        )
    elif input_order == "random":
        degrees += [torch.randperm(input_size)] if input_degrees is None else [input_degrees]
        for _ in range(n_hidden + 1):
            min_prev_degree = min(degrees[-1].min().item(), input_size - 1)
            degrees += [torch.randint(min_prev_degree, input_size, (hidden_size,))]
        min_prev_degree = min(degrees[-1].min().item(), input_size - 1)
        degrees += (
            [torch.randint(min_prev_degree, input_size, (input_size,)) - 1]
            if input_degrees is None
            else [input_degrees - 1]
        )
    else:
        raise ValueError(f"Unknown input_order: {input_order}")

    masks = []
    for d0, d1 in zip(degrees[:-1], degrees[1:]):
        masks += [(d1.unsqueeze(-1) >= d0.unsqueeze(0)).float()]
    return masks, degrees[0]


class MaskedLinear(nn.Linear):
    def __init__(self, input_size, n_outputs, mask, cond_label_size=None):
        super().__init__(input_size, n_outputs)
        self.register_buffer("mask", mask)
        self.cond_label_size = cond_label_size
        if cond_label_size is not None:
            self.cond_weight = nn.Parameter(torch.rand(n_outputs, cond_label_size) / math.sqrt(cond_label_size))

    def forward(self, x, y=None):
        out = F.linear(x, self.weight * self.mask, self.bias)
        if y is not None:
            out = out + F.linear(y, self.cond_weight)
        return out


class BatchNorm(nn.Module):
    def __init__(self, input_size, momentum=0.9, eps=1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.log_gamma = nn.Parameter(torch.zeros(input_size))
        self.beta = nn.Parameter(torch.zeros(input_size))
        self.register_buffer("running_mean", torch.zeros(input_size))
        self.register_buffer("running_var", torch.ones(input_size))

    def forward(self, x, cond_y=None):
        if self.training:
            self.batch_mean = x.mean(0)
            self.batch_var = x.var(0)
            self.running_mean.mul_(self.momentum).add_(self.batch_mean.data * (1 - self.momentum))
            self.running_var.mul_(self.momentum).add_(self.batch_var.data * (1 - self.momentum))
            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        y = self.log_gamma.exp() * x_hat + self.beta
        log_abs_det_jacobian = self.log_gamma - 0.5 * torch.log(var + self.eps)
        return y, log_abs_det_jacobian.expand_as(x)

    def inverse(self, u, cond_y=None):
        if self.training:
            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var
        x_hat = (u - self.beta) * torch.exp(-self.log_gamma)
        x = x_hat * torch.sqrt(var + self.eps) + mean
        log_abs_det_jacobian = 0.5 * torch.log(var + self.eps) - self.log_gamma
        return x, log_abs_det_jacobian.expand_as(u)


class FlowSequential(nn.Sequential):
    def forward(self, x, y=None):
        sum_log_abs_det_jacobians = 0
        for module in self:
            x, log_abs_det_jacobian = module(x, y)
            sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + log_abs_det_jacobian
        return x, sum_log_abs_det_jacobians

    def inverse(self, u, y=None):
        sum_log_abs_det_jacobians = 0
        for module in reversed(self):
            u, log_abs_det_jacobian = module.inverse(u, y)
            sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + log_abs_det_jacobian
        return u, sum_log_abs_det_jacobians


class MADE(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        n_hidden,
        cond_label_size=None,
        activation="relu",
        input_order="sequential",
        input_degrees=None,
    ):
        super().__init__()
        self.register_buffer("base_dist_mean", torch.zeros(input_size))
        self.register_buffer("base_dist_var", torch.ones(input_size))
        masks, self.input_degrees = _create_masks(
            input_size, hidden_size, n_hidden, input_order, input_degrees
        )
        activation_fn = nn.ReLU() if activation == "relu" else nn.Tanh()
        self.net_input = MaskedLinear(input_size, hidden_size, masks[0], cond_label_size)
        self.net = []
        for m in masks[1:-1]:
            self.net += [activation_fn, MaskedLinear(hidden_size, hidden_size, m)]
        self.net += [activation_fn, MaskedLinear(hidden_size, 2 * input_size, masks[-1].repeat(2, 1))]
        self.net = nn.Sequential(*self.net)

    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def forward(self, x, y=None):
        m, loga = self.net(self.net_input(x, y)).chunk(chunks=2, dim=1)
        u = (x - m) * torch.exp(-loga)
        return u, -loga

    def inverse(self, u, y=None, sum_log_abs_det_jacobians=None):
        x = torch.zeros_like(u)
        for i in self.input_degrees:
            m, loga = self.net(self.net_input(x, y)).chunk(chunks=2, dim=1)
            x[:, i] = u[:, i] * torch.exp(loga[:, i]) + m[:, i]
        return x, loga


class MADEMOG(nn.Module):
    def __init__(
        self,
        n_components,
        input_size,
        hidden_size,
        n_hidden,
        cond_label_size=None,
        activation="relu",
        input_order="sequential",
        input_degrees=None,
    ):
        super().__init__()
        self.n_components = n_components
        self.register_buffer("base_dist_mean", torch.zeros(input_size))
        self.register_buffer("base_dist_var", torch.ones(input_size))
        masks, self.input_degrees = _create_masks(
            input_size, hidden_size, n_hidden, input_order, input_degrees
        )
        activation_fn = nn.ReLU() if activation == "relu" else nn.Tanh()
        self.net_input = MaskedLinear(input_size, hidden_size, masks[0], cond_label_size)
        self.net = []
        for m in masks[1:-1]:
            self.net += [activation_fn, MaskedLinear(hidden_size, hidden_size, m)]
        self.net += [
            activation_fn,
            MaskedLinear(hidden_size, n_components * 3 * input_size, masks[-1].repeat(n_components * 3, 1)),
        ]
        self.net = nn.Sequential(*self.net)

    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def forward(self, x, y=None):
        n, l = x.shape
        c = self.n_components
        m, loga, logr = self.net(self.net_input(x, y)).view(n, c, 3 * l).chunk(chunks=3, dim=-1)
        x = x.repeat(1, c).view(n, c, l)
        u = (x - m) * torch.exp(-loga)
        log_abs_det_jacobian = -loga
        self.logr = logr - logr.logsumexp(1, keepdim=True)
        return u, log_abs_det_jacobian


class MAF(nn.Module):
    def __init__(
        self,
        input_size,
        n_blocks=5,
        hidden_size=1024,
        n_hidden=2,
        cond_label_size=None,
        activation="relu",
        input_order="sequential",
        batch_norm=True,
    ):
        super().__init__()
        self.register_buffer("base_dist_mean", torch.zeros(input_size))
        self.register_buffer("base_dist_var", torch.ones(input_size))
        modules = []
        self.input_degrees = None
        for _ in range(n_blocks):
            modules += [
                MADE(
                    input_size,
                    hidden_size,
                    n_hidden,
                    cond_label_size,
                    activation,
                    input_order,
                    self.input_degrees,
                )
            ]
            self.input_degrees = modules[-1].input_degrees.flip(0)
            modules += batch_norm * [BatchNorm(input_size)]
        self.net = FlowSequential(*modules)

    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def forward(self, x, y=None):
        return self.net(x, y)

    def log_probs(self, x, y=None):
        u, sum_log_abs_det_jacobians = self.forward(x, y)
        return torch.sum(self.base_dist.log_prob(u) + sum_log_abs_det_jacobians, dim=1)


class MAFMOG(nn.Module):
    def __init__(
        self,
        n_blocks,
        n_components,
        input_size,
        hidden_size,
        n_hidden,
        cond_label_size=None,
        activation="relu",
        input_order="sequential",
        batch_norm=True,
    ):
        super().__init__()
        self.register_buffer("base_dist_mean", torch.zeros(input_size))
        self.register_buffer("base_dist_var", torch.ones(input_size))
        self.maf = MAF(
            input_size,
            n_blocks,
            hidden_size,
            n_hidden,
            cond_label_size,
            activation,
            input_order,
            batch_norm,
        )
        input_degrees = self.maf.input_degrees
        self.mademog = MADEMOG(
            n_components,
            input_size,
            hidden_size,
            n_hidden,
            cond_label_size,
            activation,
            input_order,
            input_degrees,
        )

    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def forward(self, x, y=None):
        u, maf_log_abs_dets = self.maf(x, y)
        u, made_log_abs_dets = self.mademog(u, y)
        return u, maf_log_abs_dets.unsqueeze(1) + made_log_abs_dets

    def log_probs(self, x, y=None):
        u, log_abs_det_jacobian = self.forward(x, y)
        log_probs = torch.logsumexp(
            self.mademog.logr + self.base_dist.log_prob(u) + log_abs_det_jacobian,
            dim=1,
        )
        return log_probs.sum(1)
