# Copyright (c) 2026 EarthBridge Team.
# Credits: Adapted from Parallel-GAN (Wang et al., TGRS 2022).
# Original: https://github.com/ZZG-Z/Parallel-GAN

"""Parallel-GAN network architectures and loss modules.

This module contains all network definitions and loss helpers for the
Parallel-GAN SAR-to-Optical image translation pipeline.

Models
------
* ``Resrecon`` – Stage 1 reconstruction network (ResNet-50 encoder + decoder).
* ``ParaGAN`` – Stage 2 translation generator with skip connections.
* ``_NLayerDiscriminator`` – PatchGAN discriminator.

Losses
------
* ``VGGLoss`` – Multi-layer VGG-19 perceptual loss.
* ``_GANLoss`` – GAN loss wrapper (vanilla / LSGAN).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models


# ---------------------------------------------------------------------------
# Initialisation helpers
# ---------------------------------------------------------------------------


def _init_weights(net: nn.Module, init_type: str = "normal", init_gain: float = 0.02) -> None:
    """Apply weight initialisation to *net* in-place."""

    def _init_func(m: nn.Module) -> None:
        cls = m.__class__.__name__
        if hasattr(m, "weight") and ("Conv" in cls or "Linear" in cls):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(f"init method [{init_type}] not implemented")
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif "BatchNorm2d" in cls:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(_init_func)


# ---------------------------------------------------------------------------
# ResNet block
# ---------------------------------------------------------------------------


class _ResnetBlock(nn.Module):
    """ResNet block with configurable padding, normalisation, and dropout."""

    def __init__(
        self,
        dim: int,
        padding_type: str = "reflect",
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
        use_dropout: bool = False,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        p = 0
        if padding_type == "reflect":
            layers.append(nn.ReflectionPad2d(1))
        elif padding_type == "replicate":
            layers.append(nn.ReplicationPad2d(1))
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError(f"padding [{padding_type}] not implemented")

        layers += [nn.Conv2d(dim, dim, 3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            layers.append(nn.Dropout(0.5))

        p = 0
        if padding_type == "reflect":
            layers.append(nn.ReflectionPad2d(1))
        elif padding_type == "replicate":
            layers.append(nn.ReplicationPad2d(1))
        elif padding_type == "zero":
            p = 1

        layers += [nn.Conv2d(dim, dim, 3, padding=p, bias=use_bias), norm_layer(dim)]
        self.conv_block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv_block(x)


# ---------------------------------------------------------------------------
# Reconstruction network  (Stage 1)
# ---------------------------------------------------------------------------


class Resrecon(nn.Module):
    """Reconstruction network: ResNet-50 encoder → transposed-conv decoder.

    Trained in Stage 1 to reconstruct optical images from optical images,
    learning a rich feature hierarchy.  At inference / Stage 2, the
    intermediate features serve as supervision targets.

    The forward pass returns a list of six feature tensors (five
    intermediate decoder features + the final RGB output).
    """

    def __init__(self) -> None:
        super().__init__()
        norm_layer = nn.BatchNorm2d
        use_bias = False

        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, 4, stride=2, padding=1, bias=use_bias),
            norm_layer(1024),
            nn.ReLU(False),
            nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1, bias=use_bias),
            norm_layer(512),
            nn.ReLU(False),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=use_bias),
            norm_layer(256),
            nn.ReLU(False),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=use_bias),
            norm_layer(128),
            nn.ReLU(False),
            nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1, bias=use_bias),
            norm_layer(64),
            nn.ReLU(False),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1, bias=use_bias),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        out_features: list[torch.Tensor] = []
        feat = self.encoder(x)
        for i in range(5):
            feat = self.decoder[3 * i : 3 * (i + 1)](feat)
            out_features.append(feat)
        out_features.append(self.decoder[15:17](feat))
        return out_features


# ---------------------------------------------------------------------------
# Translation network  (Stage 2)
# ---------------------------------------------------------------------------


class ParaGAN(nn.Module):
    """Parallel-GAN translation generator.

    Uses a ResNet-50 encoder, six residual bottleneck blocks, and a decoder
    with encoder skip connections (concatenation).  The forward pass returns
    a list of six feature tensors for hierarchical feature supervision.

    Parameters
    ----------
    input_nc : int
        Number of input channels (e.g. 3 for RGB SAR, 4 for SpaceNet).
    output_nc : int
        Number of output channels (typically 3 for RGB optical).
    channel : int
        Channel width inside the residual blocks (default 2048).
    norm_layer
        Normalisation layer constructor.
    use_dropout : bool
        Use dropout inside residual blocks.
    n_blocks : int
        Number of residual blocks between encoder and decoder.
    """

    def __init__(
        self,
        input_nc: int = 3,
        output_nc: int = 3,
        channel: int = 2048,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
        use_dropout: bool = False,
        n_blocks: int = 6,
        padding_type: str = "reflect",
    ) -> None:
        super().__init__()
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])

        # Handle 4-channel SAR input
        if input_nc == 4:
            with torch.no_grad():
                pretrained_w = self.encoder[0].weight.clone()
                self.encoder[0] = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
                self.encoder[0].weight[:, :3] = pretrained_w
                self.encoder[0].weight[:, 3:] = pretrained_w[:, :1]

        self.resnet_part = nn.Sequential(
            *[
                _ResnetBlock(channel, padding_type=padding_type, norm_layer=norm_layer,
                             use_dropout=use_dropout, use_bias=use_bias)
                for _ in range(n_blocks)
            ]
        )

        self.decoder = nn.Sequential(
            # Stage 0: 2048 → 1024
            nn.ConvTranspose2d(2048, 1024, 4, stride=2, padding=1, bias=use_bias),
            norm_layer(1024),
            nn.ReLU(False),
            # Stage 1: 2048 (cat) → 512
            nn.ConvTranspose2d(2048, 512, 4, stride=2, padding=1, bias=use_bias),
            norm_layer(512),
            nn.ReLU(False),
            # Stage 2: 1024 (cat) → 256
            nn.ConvTranspose2d(1024, 256, 4, stride=2, padding=1, bias=use_bias),
            norm_layer(256),
            nn.ReLU(False),
            # Stage 3: 512 (cat) → 128
            nn.ConvTranspose2d(512, 128, 4, stride=2, padding=1, bias=use_bias),
            norm_layer(128),
            nn.ReLU(False),
            # Stage 4: 128 → 64
            nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1, bias=use_bias),
            norm_layer(64),
            nn.ReLU(False),
            # Final: 128 (cat) → output_nc
            nn.ConvTranspose2d(128, output_nc, 4, stride=2, padding=1, bias=use_bias),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        encoder_features: list[torch.Tensor] = []
        out_features: list[torch.Tensor] = []

        feat = x
        for block in self.encoder:
            feat = block(feat)
            encoder_features.append(feat)

        decoder_feat = self.resnet_part(encoder_features[-1])

        for i in range(5):
            decoder_feat = self.decoder[3 * i : 3 * (i + 1)](decoder_feat)
            out_features.append(decoder_feat)
            if i < 3:
                decoder_feat = torch.cat([decoder_feat, encoder_features[6 - i]], dim=1)

        decoder_feat = torch.cat([decoder_feat, encoder_features[0]], dim=1)
        out_features.append(self.decoder[15:17](decoder_feat))
        return out_features


# ---------------------------------------------------------------------------
# PatchGAN discriminator
# ---------------------------------------------------------------------------


class _NLayerDiscriminator(nn.Module):
    """PatchGAN discriminator."""

    def __init__(
        self,
        input_nc: int,
        ndf: int = 64,
        n_layers: int = 3,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
    ) -> None:
        super().__init__()
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw, padw = 4, 1
        sequence: list[nn.Module] = [
            nn.Conv2d(input_nc, ndf, kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * nf_mult, 1, kw, stride=1, padding=padw),
        ]
        self.model = nn.Sequential(*sequence)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ---------------------------------------------------------------------------
# VGG perceptual loss
# ---------------------------------------------------------------------------


class _VGGFeatures(nn.Module):
    """VGG-19 multi-layer feature extractor (frozen)."""

    def __init__(self) -> None:
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        slices = [2, 7, 12, 21, 30]
        self.slices = nn.ModuleList()
        prev = 0
        for s in slices:
            self.slices.append(nn.Sequential(*[vgg[i] for i in range(prev, s)]))
            prev = s
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats: list[torch.Tensor] = []
        h = x
        for s in self.slices:
            h = s(h)
            feats.append(h)
        return feats


class VGGLoss(nn.Module):
    """Multi-layer VGG perceptual loss used by the reconstruction network."""

    def __init__(self) -> None:
        super().__init__()
        self.vgg = _VGGFeatures()
        self.criterion = nn.L1Loss()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_feats = self.vgg(x)
        y_feats = self.vgg(y)
        loss = torch.tensor(0.0, device=x.device)
        for xf, yf in zip(x_feats, y_feats):
            loss = loss + self.criterion(xf, yf.detach())
        return loss


# ---------------------------------------------------------------------------
# GAN loss
# ---------------------------------------------------------------------------


class _GANLoss(nn.Module):
    """GAN loss (vanilla / lsgan)."""

    def __init__(self, gan_mode: str = "vanilla") -> None:
        super().__init__()
        self.register_buffer("real_label", torch.tensor(1.0))
        self.register_buffer("fake_label", torch.tensor(0.0))
        self.gan_mode = gan_mode
        if gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        elif gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError(f"GAN mode [{gan_mode}] not implemented")

    def _target(self, pred: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        return (self.real_label if target_is_real else self.fake_label).expand_as(pred)

    def forward(self, pred: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        return self.loss(pred, self._target(pred, target_is_real))
