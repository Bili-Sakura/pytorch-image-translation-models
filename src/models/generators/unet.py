# Credits: Built on pix2pix (Isola et al., CVPR 2017) and open-source libraries.
#
"""U-Net generator architecture."""

from __future__ import annotations

import torch
import torch.nn as nn


class UNetSkipConnection(nn.Module):
    """Single U-Net skip-connection block."""

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
    """U-Net based generator with skip connections."""

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
        block = UNetSkipConnection(
            base_filters * 8,
            base_filters * 8,
            innermost=True,
            norm_layer=norm_layer,
        )
        for _ in range(num_downs - 5):
            block = UNetSkipConnection(
                base_filters * 8,
                base_filters * 8,
                sub_module=block,
                use_dropout=use_dropout,
                norm_layer=norm_layer,
            )
        block = UNetSkipConnection(
            base_filters * 4, base_filters * 8, sub_module=block, norm_layer=norm_layer
        )
        block = UNetSkipConnection(
            base_filters * 2, base_filters * 4, sub_module=block, norm_layer=norm_layer
        )
        block = UNetSkipConnection(
            base_filters, base_filters * 2, sub_module=block, norm_layer=norm_layer
        )
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

