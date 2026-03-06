# Copyright (c) 2026 EarthBridge Team.
# Credits: Adapted from StegoGAN (Wu et al., CVPR 2024).
# Original: https://github.com/sian-wusidi/StegoGAN

"""Helper network building blocks for StegoGAN."""

from __future__ import annotations

import functools

import torch
import torch.nn as nn


class SoftClamp(nn.Module):
    """Differentiable soft-clamping to ``[0, 1]``.

    Values outside the range are linearly attenuated by *alpha* rather
    than being hard-clipped, which keeps gradients flowing.

    Reference: https://github.com/monniert/dti-sprites
    """

    def __init__(self, alpha: float = 0.001) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = torch.min(x, torch.zeros_like(x))
        x1 = torch.max(x - 1, torch.zeros_like(x))
        return torch.clamp(x, 0, 1) + self.alpha * x0 + self.alpha * x1


class ResnetBlock(nn.Module):
    """ResNet block with configurable padding, normalisation, and dropout.

    Parameters
    ----------
    dim : int
        Number of channels.
    padding_type : str
        ``"reflect"`` | ``"replicate"`` | ``"zero"``.
    norm_layer
        Normalisation layer constructor (e.g. ``nn.BatchNorm2d``).
    use_dropout : bool
        Whether to include a dropout layer.
    use_bias : bool
        Whether convolution layers use a bias term.
    """

    def __init__(
        self,
        dim: int,
        padding_type: str = "reflect",
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
        use_dropout: bool = False,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        self.conv_block = self._build(dim, padding_type, norm_layer, use_dropout, use_bias)

    @staticmethod
    def _build(
        dim: int,
        padding_type: str,
        norm_layer: type[nn.Module],
        use_dropout: bool,
        use_bias: bool,
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        p = 0
        if padding_type == "reflect":
            layers.append(nn.ReflectionPad2d(1))
        elif padding_type == "replicate":
            layers.append(nn.ReplicationPad2d(1))
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError(f"padding [{padding_type}] is not implemented")

        layers += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            layers.append(nn.Dropout(0.5))

        p = 0
        if padding_type == "reflect":
            layers.append(nn.ReflectionPad2d(1))
        elif padding_type == "replicate":
            layers.append(nn.ReplicationPad2d(1))
        elif padding_type == "zero":
            p = 1

        layers += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv_block(x)


class NetMatchability(nn.Module):
    """Per-pixel matchability mask predictor.

    Produces a sigmoid mask that identifies which spatial features
    correspond to domain-matched content versus domain-mismatched content.

    Parameters
    ----------
    input_dim : int
        Number of input feature channels.
    out_dim : int
        Number of output mask channels.
    """

    def __init__(self, input_dim: int = 256, out_dim: int = 256) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.InstanceNorm2d(input_dim, eps=1e-05)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.InstanceNorm2d(input_dim, eps=1e-05)
        self.conv3 = nn.Conv2d(input_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.norm1(self.conv1(feat)))
        x = self.relu(self.norm2(self.conv2(x)))
        return self.sigmoid(self.conv3(x))


def mask_generate(
    latent_feature: torch.Tensor,
    mask_net: NetMatchability,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the matchability mask to latent features.

    Returns
    -------
    output : Tensor
        Masked latent features (matched content only).
    features_discarded : Tensor
        Discarded latent features (mismatched content).
    reverse_mask_sum : Tensor
        Single-channel summary of the reverse mask (max over channels).
    """
    latent_mask = mask_net(latent_feature)
    output = latent_feature * latent_mask
    reverse_mask = 1 - latent_mask
    reverse_mask_sum = torch.unsqueeze(torch.max(reverse_mask, dim=1)[0], dim=1)
    features_discarded = latent_feature * reverse_mask
    return output, features_discarded, reverse_mask_sum


def get_norm_layer(norm_type: str = "instance"):
    """Return a normalisation layer constructor.

    Parameters
    ----------
    norm_type : str
        ``"batch"`` | ``"instance"`` | ``"none"``.
    """
    if norm_type == "batch":
        return functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    if norm_type == "instance":
        return functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    if norm_type == "none":
        return nn.Identity
    raise NotImplementedError(f"normalization layer [{norm_type}] is not found")
