# Copyright (c) 2026 EarthBridge Team.
# Credits: Decent from Xie et al. NeurIPS 2022 — https://github.com/Mid-Push/Decent

"""Block Neural Autoregressive Flow (BNAF) density estimator."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn


class _FlowSequential(torch.nn.Sequential):
    """Sequential container that accumulates log-det-Jacobian terms."""

    def forward(self, inputs: torch.Tensor):
        log_det_jacobian = 0.0
        for module in self._modules.values():
            inputs, log_det_jacobian_ = module(inputs)
            log_det_jacobian = log_det_jacobian + log_det_jacobian_
        return inputs, log_det_jacobian


class BNAF(torch.nn.Sequential):
    """Block Neural Normalizing Flow stack."""

    def __init__(self, *args, res: str | None = None):
        super().__init__(*args)
        self.res = res
        if res == "gated":
            self.gate = torch.nn.Parameter(torch.nn.init.normal_(torch.Tensor(1)))

    def forward(self, inputs: torch.Tensor):
        outputs = inputs
        grad = None
        for module in self._modules.values():
            outputs, grad = module(outputs, grad)

        grad = grad if len(grad.shape) == 4 else grad.view(*grad.shape, 1, 1)
        assert inputs.shape[-1] == outputs.shape[-1]

        if self.res == "normal":
            return inputs + outputs, torch.nn.functional.softplus(grad.squeeze()).sum(-1)
        if self.res == "gated":
            gate = self.gate.sigmoid()
            return (
                gate * outputs + (1 - gate) * inputs,
                (
                    torch.nn.functional.softplus(grad.squeeze() + self.gate)
                    - torch.nn.functional.softplus(self.gate)
                ).sum(-1),
            )
        return outputs, grad.squeeze().sum(-1)


class Permutation(torch.nn.Module):
    def __init__(self, in_features: int, p: list | str | None = None):
        super().__init__()
        self.in_features = in_features
        if p is None:
            self.p = np.random.permutation(in_features)
        elif p == "flip":
            self.p = list(reversed(range(in_features)))
        else:
            self.p = p

    def forward(self, inputs: torch.Tensor):
        return inputs[:, self.p], 0


class MaskedWeight(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, dim: int, bias: bool = True):
        super().__init__()
        self.in_features, self.out_features, self.dim = in_features, out_features, dim

        weight = torch.zeros(out_features, in_features)
        for i in range(dim):
            weight[
                i * out_features // dim : (i + 1) * out_features // dim,
                0 : (i + 1) * in_features // dim,
            ] = torch.nn.init.xavier_uniform_(
                torch.Tensor(out_features // dim, (i + 1) * in_features // dim)
            )

        self._weight = torch.nn.Parameter(weight)
        self._diag_weight = torch.nn.Parameter(
            torch.nn.init.uniform_(torch.Tensor(out_features, 1)).log()
        )
        self.bias = (
            torch.nn.Parameter(
                torch.nn.init.uniform_(
                    torch.Tensor(out_features),
                    -1 / math.sqrt(out_features),
                    1 / math.sqrt(out_features),
                )
            )
            if bias
            else 0
        )

        mask_d = torch.zeros_like(weight)
        for i in range(dim):
            mask_d[
                i * (out_features // dim) : (i + 1) * (out_features // dim),
                i * (in_features // dim) : (i + 1) * (in_features // dim),
            ] = 1
        self.register_buffer("mask_d", mask_d)

        mask_o = torch.ones_like(weight)
        for i in range(dim):
            mask_o[
                i * (out_features // dim) : (i + 1) * (out_features // dim),
                i * (in_features // dim) :,
            ] = 0
        self.register_buffer("mask_o", mask_o)

    def get_weights(self):
        w = torch.exp(self._weight) * self.mask_d + self._weight * self.mask_o
        w_squared_norm = (w**2).sum(-1, keepdim=True)
        w = self._diag_weight.exp() * w / w_squared_norm.sqrt()
        wpl = self._diag_weight + self._weight - 0.5 * torch.log(w_squared_norm)
        return w.t(), wpl.t()[self.mask_d.bool().t()].view(
            self.dim, self.in_features // self.dim, self.out_features // self.dim
        )

    def forward(self, inputs, grad: torch.Tensor | None = None):
        w, wpl = self.get_weights()
        g = wpl.transpose(-2, -1).unsqueeze(0).repeat(inputs.shape[0], 1, 1, 1)
        return (
            inputs.matmul(w) + self.bias,
            torch.logsumexp(g.unsqueeze(-2) + grad.transpose(-2, -1).unsqueeze(-3), -1)
            if grad is not None
            else g,
        )


class Tanh(torch.nn.Tanh):
    def forward(self, inputs, grad: torch.Tensor | None = None):
        g = -2 * (inputs - math.log(2) + torch.nn.functional.softplus(-2 * inputs))
        return torch.tanh(inputs), (g.view(grad.shape) + grad) if grad is not None else g


class BNAFModel(nn.Module):
    """BNAF density model used by :class:`~src.models.decent.PatchDensityEstimator`."""

    def __init__(
        self,
        num_inputs: int,
        n_flows: int = 5,
        n_layers: int = 0,
        hidden_dim: int = 10,
        residual: str = "gated",
    ):
        super().__init__()
        flows = []
        for f in range(n_flows):
            layers = []
            for _ in range(n_layers - 1):
                layers.append(
                    MaskedWeight(num_inputs * hidden_dim, num_inputs * hidden_dim, dim=num_inputs)
                )
                layers.append(Tanh())

            flows.append(
                BNAF(
                    *(
                        [
                            MaskedWeight(num_inputs, num_inputs * hidden_dim, dim=num_inputs),
                            Tanh(),
                        ]
                        + layers
                        + [MaskedWeight(num_inputs * hidden_dim, num_inputs, dim=num_inputs)]
                    ),
                    res=residual if f < n_flows - 1 else None,
                )
            )
            if f < n_flows - 1:
                flows.append(Permutation(num_inputs, "flip"))
        self.model = _FlowSequential(*flows)

    def log_probs(self, x_mb: torch.Tensor) -> torch.Tensor:
        y_mb, log_diag_j_mb = self.model(x_mb)
        log_p_y_mb = (
            torch.distributions.Normal(torch.zeros_like(y_mb), torch.ones_like(y_mb))
            .log_prob(y_mb)
            .sum(-1)
        )
        return log_p_y_mb + log_diag_j_mb


class FlowAdam(torch.optim.Optimizer):
    """Adam optimizer with Polyak averaging for flow density estimators."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False,
        polyak: float = 0.998,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= polyak <= 1.0:
            raise ValueError(f"Invalid polyak decay term: {polyak}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            polyak=polyak,
        )
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("FlowAdam does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                    state["exp_avg_param"] = torch.zeros_like(p.data)
                    if group["amsgrad"]:
                        state["max_exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                if group["amsgrad"]:
                    max_exp_avg_sq = state["max_exp_avg_sq"]

                state["step"] += 1

                if group["weight_decay"] != 0:
                    grad = grad.add(p.data, alpha=group["weight_decay"])

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if group["amsgrad"]:
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().add_(group["eps"])
                else:
                    denom = exp_avg_sq.sqrt().add_(group["eps"])

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1
                p.data.addcdiv_(-step_size, exp_avg, denom)

                polyak = self.defaults["polyak"]
                state["exp_avg_param"] = (
                    polyak * state["exp_avg_param"] + (1 - polyak) * p.data
                )

        return loss

    def swap(self):
        """Swap current parameters with Polyak-averaged copies."""
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                new = p.data
                p.data = state["exp_avg_param"]
                state["exp_avg_param"] = new
