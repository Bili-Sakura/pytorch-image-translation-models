# Copyright (c) 2026 EarthBridge Team.
# Credits: Adapted from StegoGAN (Wu et al., CVPR 2024).
# Original: https://github.com/sian-wusidi/StegoGAN

"""StegoGAN generator architectures with steganographic masking.

The two generators form a CycleGAN-like pair:

* **ResnetMaskV1Generator** (``G_A``): translates domain A → B and
  optionally injects steganographic feature information via additive
  skip connections.
* **ResnetMaskV3Generator** (``G_B``): translates domain B → A while
  producing a per-pixel matchability mask that identifies
  domain-mismatched content.
"""

from __future__ import annotations

import functools
from typing import Optional

import torch
import torch.nn as nn

from src.models.stegogan.networks import (
    NetMatchability,
    ResnetBlock,
    mask_generate,
)


class ResnetMaskV1Generator(nn.Module):
    """ResNet generator with optional steganographic feature injection (``G_A``).

    Parameters
    ----------
    input_nc : int
        Number of input channels.
    output_nc : int
        Number of output channels.
    ngf : int
        Base number of generator filters.
    norm_layer
        Normalisation layer constructor.
    use_dropout : bool
        Whether to use dropout inside ResNet blocks.
    n_blocks : int
        Total number of ResNet blocks.
    resnet_layer : int
        After which ResNet block the extra feature is injected.
        ``-1`` means before any ResNet block, ``n_blocks - 1`` (or ``8``)
        means after all blocks.
    fusionblock : bool
        If ``True``, apply an extra ResNet block to the injected features.
    """

    def __init__(
        self,
        input_nc: int = 3,
        output_nc: int = 3,
        ngf: int = 64,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
        use_dropout: bool = False,
        n_blocks: int = 9,
        padding_type: str = "reflect",
        resnet_layer: int = 1,
        fusionblock: bool = False,
    ) -> None:
        super().__init__()
        self.resnet_layer = resnet_layer
        self.n_blocks = n_blocks
        assert n_blocks >= 0

        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # Encoder
        encoder: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True),
        ]
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            encoder += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True),
            ]
        self.conv1 = nn.Sequential(*encoder)

        # ResNet blocks (split around the injection point)
        mult = 2 ** n_downsampling
        before: list[nn.Module] = [
            ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                        use_dropout=use_dropout, use_bias=use_bias)
            for _ in range(self.resnet_layer + 1)
        ]
        self.conv2 = nn.Sequential(*before)

        after: list[nn.Module] = [
            ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                        use_dropout=use_dropout, use_bias=use_bias)
            for _ in range(self.resnet_layer + 1, self.n_blocks)
        ]
        self.conv3 = nn.Sequential(*after)

        # Decoder
        decoder: list[nn.Module] = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            decoder += [
                nn.ConvTranspose2d(
                    ngf * mult, ngf * mult // 2,
                    kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias,
                ),
                norm_layer(ngf * mult // 2),
                nn.ReLU(True),
            ]
        decoder += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.decoder = nn.Sequential(*decoder)

        # Optional fusion block for injected features
        self.fusion_block = fusionblock
        if self.fusion_block:
            fb = [
                ResnetBlock(ngf * (2 ** n_downsampling), padding_type=padding_type,
                            norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
            ]
            self.fusion = nn.Sequential(*fb)

    def forward(
        self,
        x: torch.Tensor,
        extra_feature: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor ``[B, C, H, W]``
            Input image.
        extra_feature : Tensor, optional
            Steganographic feature to inject (same spatial dims as
            the intermediate representation).
        """
        output = self.conv1(x)

        if extra_feature is not None and self.fusion_block:
            extra_feature = self.fusion(extra_feature)

        if self.resnet_layer == -1:
            if extra_feature is not None:
                output = output + extra_feature
            output = self.conv3(output)
        elif self.resnet_layer >= self.n_blocks - 1:
            output = self.conv2(output)
            if extra_feature is not None:
                output = output + extra_feature
        else:
            output = self.conv2(output)
            if extra_feature is not None:
                output = output + extra_feature
            output = self.conv3(output)

        return self.decoder(output)


class ResnetMaskV3Generator(nn.Module):
    """ResNet generator with matchability masking (``G_B``).

    In addition to translating the image, this generator produces a
    per-pixel mask via :class:`NetMatchability` that identifies
    domain-mismatched content.

    Parameters
    ----------
    input_nc : int
        Number of input channels.
    output_nc : int
        Number of output channels.
    ngf : int
        Base number of generator filters.
    norm_layer
        Normalisation layer constructor.
    use_dropout : bool
        Whether to use dropout.
    n_blocks : int
        Number of ResNet blocks.
    out_dim : int
        Number of mask output channels (passed to :class:`NetMatchability`).
    resnet_layer : int
        Injection point for the matchability mask.
    """

    def __init__(
        self,
        input_nc: int = 3,
        output_nc: int = 3,
        ngf: int = 64,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
        use_dropout: bool = False,
        n_blocks: int = 9,
        padding_type: str = "reflect",
        input_dim: int = 256,
        out_dim: int = 256,
        resnet_layer: int = 1,
    ) -> None:
        super().__init__()
        self.resnet_layer = resnet_layer
        self.n_blocks = n_blocks
        assert n_blocks >= 0

        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # Encoder
        encoder: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True),
        ]
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            encoder += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True),
            ]
        self.conv1 = nn.Sequential(*encoder)

        mult = 2 ** n_downsampling
        before: list[nn.Module] = [
            ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                        use_dropout=use_dropout, use_bias=use_bias)
            for _ in range(self.resnet_layer + 1)
        ]
        self.conv2 = nn.Sequential(*before)

        after: list[nn.Module] = [
            ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                        use_dropout=use_dropout, use_bias=use_bias)
            for _ in range(self.resnet_layer + 1, self.n_blocks)
        ]
        self.conv3 = nn.Sequential(*after)

        self.mask = NetMatchability(input_dim=input_dim, out_dim=out_dim)

        # Decoder
        decoder: list[nn.Module] = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            decoder += [
                nn.ConvTranspose2d(
                    ngf * mult, ngf * mult // 2,
                    kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias,
                ),
                norm_layer(ngf * mult // 2),
                nn.ReLU(True),
            ]
        decoder += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.decoder = nn.Sequential(*decoder)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns
        -------
        output : Tensor ``[B, output_nc, H, W]``
            Translated image.
        features_discarded : Tensor
            Latent features identified as mismatched.
        reverse_mask_sum : Tensor ``[B, 1, h, w]``
            Single-channel mismatch mask (max across channels).
        """
        output = self.conv1(x)

        if self.resnet_layer == -1:
            output, features_discarded, reverse_mask_sum = mask_generate(output, self.mask)
            output = self.conv3(output)
        elif self.resnet_layer >= self.n_blocks - 1:
            output = self.conv2(output)
            output, features_discarded, reverse_mask_sum = mask_generate(output, self.mask)
        else:
            output = self.conv2(output)
            output, features_discarded, reverse_mask_sum = mask_generate(output, self.mask)
            output = self.conv3(output)

        output = self.decoder(output)
        return output, features_discarded, reverse_mask_sum
