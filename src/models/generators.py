"""Generator architectures for image-to-image translation."""

from __future__ import annotations

import torch
import torch.nn as nn


class UNetSkipConnection(nn.Module):
    """Single U-Net skip-connection block.

    Recursively builds the encoder-decoder path.  The outermost block
    has no skip connection on the encoder side; the innermost block uses
    no sub-module.
    """

    def __init__(
        self,
        outer_channels: int,
        inner_channels: int,
        input_channels: int | None = None,
        sub_module: nn.Module | None = None,
        outermost: bool = False,
        innermost: bool = False,
        use_dropout: bool = False,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
    ) -> None:
        super().__init__()
        self.outermost = outermost
        if input_channels is None:
            input_channels = outer_channels

        down_conv = nn.Conv2d(
            input_channels, inner_channels, kernel_size=4, stride=2, padding=1, bias=False
        )
        down_relu = nn.LeakyReLU(0.2, inplace=True)
        down_norm = norm_layer(inner_channels)

        up_relu = nn.ReLU(inplace=True)
        up_norm = norm_layer(outer_channels)

        if outermost:
            up_conv = nn.ConvTranspose2d(
                inner_channels * 2, outer_channels, kernel_size=4, stride=2, padding=1
            )
            down = [down_conv]
            up = [up_relu, up_conv, nn.Tanh()]
        elif innermost:
            up_conv = nn.ConvTranspose2d(
                inner_channels, outer_channels, kernel_size=4, stride=2, padding=1, bias=False
            )
            down = [down_relu, down_conv]
            up = [up_relu, up_conv, up_norm]
        else:
            up_conv = nn.ConvTranspose2d(
                inner_channels * 2, outer_channels, kernel_size=4, stride=2, padding=1, bias=False
            )
            down = [down_relu, down_conv, down_norm]
            up = [up_relu, up_conv, up_norm]
            if use_dropout:
                up = up + [nn.Dropout(0.5)]

        self.model: nn.Module
        if sub_module is not None:
            self.model = nn.Sequential(*down, sub_module, *up)
        else:
            self.model = nn.Sequential(*down, *up)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.outermost:
            return self.model(x)
        return torch.cat([x, self.model(x)], dim=1)


class UNetGenerator(nn.Module):
    """U-Net based generator with skip connections.

    Suitable for tasks where the input and output share structural
    information (e.g. paired image translation, segmentation masks
    to photos).

    Parameters
    ----------
    in_channels:
        Number of input image channels.
    out_channels:
        Number of output image channels.
    num_downs:
        Number of down-sampling layers (depth of the U-Net).
    base_filters:
        Number of filters in the last convolution layer.
    use_dropout:
        Whether to use dropout in the middle layers.
    norm_layer:
        Normalisation layer constructor.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_downs: int = 8,
        base_filters: int = 64,
        use_dropout: bool = True,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
    ) -> None:
        super().__init__()

        # Innermost block
        block = UNetSkipConnection(
            base_filters * 8,
            base_filters * 8,
            innermost=True,
            norm_layer=norm_layer,
        )

        # Middle blocks with skip connections
        for _ in range(num_downs - 5):
            block = UNetSkipConnection(
                base_filters * 8,
                base_filters * 8,
                sub_module=block,
                use_dropout=use_dropout,
                norm_layer=norm_layer,
            )

        # Progressively reduce filters
        block = UNetSkipConnection(
            base_filters * 4, base_filters * 8, sub_module=block, norm_layer=norm_layer
        )
        block = UNetSkipConnection(
            base_filters * 2, base_filters * 4, sub_module=block, norm_layer=norm_layer
        )
        block = UNetSkipConnection(
            base_filters, base_filters * 2, sub_module=block, norm_layer=norm_layer
        )

        # Outermost block
        self.model = UNetSkipConnection(
            out_channels,
            base_filters,
            input_channels=in_channels,
            sub_module=block,
            outermost=True,
            norm_layer=norm_layer,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ResidualBlock(nn.Module):
    """Residual block with two convolutions and a skip connection."""

    def __init__(self, channels: int, norm_layer: type[nn.Module] = nn.InstanceNorm2d) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=False),
            norm_layer(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=False),
            norm_layer(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResNetGenerator(nn.Module):
    """ResNet-based generator for image-to-image translation.

    Uses an encoder, a sequence of residual blocks, and a decoder.
    Well suited to style-transfer and unpaired translation tasks.

    Parameters
    ----------
    in_channels:
        Number of input image channels.
    out_channels:
        Number of output image channels.
    base_filters:
        Number of filters in the first convolution layer.
    n_residual_blocks:
        Number of residual blocks in the middle of the network.
    norm_layer:
        Normalisation layer constructor.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_filters: int = 64,
        n_residual_blocks: int = 9,
        norm_layer: type[nn.Module] = nn.InstanceNorm2d,
    ) -> None:
        super().__init__()

        # Initial convolution
        encoder: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, base_filters, kernel_size=7, bias=False),
            norm_layer(base_filters),
            nn.ReLU(inplace=True),
        ]

        # Down-sampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            encoder += [
                nn.Conv2d(
                    base_filters * mult,
                    base_filters * mult * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                norm_layer(base_filters * mult * 2),
                nn.ReLU(inplace=True),
            ]

        # Residual blocks
        mult = 2**n_downsampling
        residual = [
            ResidualBlock(base_filters * mult, norm_layer=norm_layer)
            for _ in range(n_residual_blocks)
        ]

        # Up-sampling
        decoder: list[nn.Module] = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            decoder += [
                nn.ConvTranspose2d(
                    base_filters * mult,
                    base_filters * mult // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=False,
                ),
                norm_layer(base_filters * mult // 2),
                nn.ReLU(inplace=True),
            ]

        # Final convolution
        decoder += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(base_filters, out_channels, kernel_size=7),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*encoder, *residual, *decoder)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
