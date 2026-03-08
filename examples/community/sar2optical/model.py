"""SAR2Optical model components (pix2pix-style cGAN).

Adapted from:
https://github.com/yuuIind/SAR2Optical
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _DownsamplingBlock(nn.Module):
    """Convolution + optional BatchNorm + LeakyReLU."""

    def __init__(
        self,
        c_in: int,
        c_out: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        negative_slope: float = 0.2,
        use_norm: bool = True,
    ) -> None:
        super().__init__()
        block: list[nn.Module] = [
            nn.Conv2d(
                in_channels=c_in,
                out_channels=c_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=not use_norm,
            )
        ]
        if use_norm:
            block.append(nn.BatchNorm2d(c_out))
        block.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        self.conv_block = nn.Sequential(*block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_block(x)


class _UpsamplingBlock(nn.Module):
    """Transpose-conv or upsample-conv + BatchNorm + optional dropout + ReLU."""

    def __init__(
        self,
        c_in: int,
        c_out: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        use_dropout: bool = False,
        use_upsampling: bool = False,
        mode: str = "nearest",
    ) -> None:
        super().__init__()
        block: list[nn.Module] = []
        if use_upsampling:
            up_mode = mode if mode in ("nearest", "bilinear", "bicubic") else "nearest"
            block.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode=up_mode),
                    nn.Conv2d(
                        in_channels=c_in,
                        out_channels=c_out,
                        kernel_size=3,
                        stride=1,
                        padding=padding,
                        bias=False,
                    ),
                )
            )
        else:
            block.append(
                nn.ConvTranspose2d(
                    in_channels=c_in,
                    out_channels=c_out,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False,
                )
            )
        block.append(nn.BatchNorm2d(c_out))
        if use_dropout:
            block.append(nn.Dropout(0.5))
        block.append(nn.ReLU(inplace=True))
        self.conv_block = nn.Sequential(*block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_block(x)


class _UnetEncoder(nn.Module):
    """U-Net encoder branch (8 downsampling stages)."""

    def __init__(self, c_in: int = 3, c_out: int = 512) -> None:
        super().__init__()
        self.enc1 = _DownsamplingBlock(c_in, 64, use_norm=False)
        self.enc2 = _DownsamplingBlock(64, 128)
        self.enc3 = _DownsamplingBlock(128, 256)
        self.enc4 = _DownsamplingBlock(256, 512)
        self.enc5 = _DownsamplingBlock(512, 512)
        self.enc6 = _DownsamplingBlock(512, 512)
        self.enc7 = _DownsamplingBlock(512, 512)
        self.enc8 = _DownsamplingBlock(512, c_out)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)
        x6 = self.enc6(x5)
        x7 = self.enc7(x6)
        x8 = self.enc8(x7)
        return [x8, x7, x6, x5, x4, x3, x2, x1]


class _UnetDecoder(nn.Module):
    """U-Net decoder branch with skip-connections."""

    def __init__(
        self,
        c_in: int = 512,
        c_out: int = 64,
        use_upsampling: bool = False,
        mode: str = "nearest",
    ) -> None:
        super().__init__()
        self.dec1 = _UpsamplingBlock(c_in, 512, use_dropout=True, use_upsampling=use_upsampling, mode=mode)
        self.dec2 = _UpsamplingBlock(1024, 512, use_dropout=True, use_upsampling=use_upsampling, mode=mode)
        self.dec3 = _UpsamplingBlock(1024, 512, use_dropout=True, use_upsampling=use_upsampling, mode=mode)
        self.dec4 = _UpsamplingBlock(1024, 512, use_upsampling=use_upsampling, mode=mode)
        self.dec5 = _UpsamplingBlock(1024, 256, use_upsampling=use_upsampling, mode=mode)
        self.dec6 = _UpsamplingBlock(512, 128, use_upsampling=use_upsampling, mode=mode)
        self.dec7 = _UpsamplingBlock(256, 64, use_upsampling=use_upsampling, mode=mode)
        self.dec8 = _UpsamplingBlock(128, c_out, use_upsampling=use_upsampling, mode=mode)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        x9 = torch.cat([x[1], self.dec1(x[0])], dim=1)
        x10 = torch.cat([x[2], self.dec2(x9)], dim=1)
        x11 = torch.cat([x[3], self.dec3(x10)], dim=1)
        x12 = torch.cat([x[4], self.dec4(x11)], dim=1)
        x13 = torch.cat([x[5], self.dec5(x12)], dim=1)
        x14 = torch.cat([x[6], self.dec6(x13)], dim=1)
        x15 = torch.cat([x[7], self.dec7(x14)], dim=1)
        return self.dec8(x15)


class SAR2OpticalGenerator(nn.Module):
    """U-Net generator used by SAR2Optical pix2pix training."""

    def __init__(
        self,
        c_in: int = 3,
        c_out: int = 3,
        use_upsampling: bool = False,
        mode: str = "nearest",
    ) -> None:
        super().__init__()
        self.encoder = _UnetEncoder(c_in=c_in)
        self.decoder = _UnetDecoder(use_upsampling=use_upsampling, mode=mode)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=c_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_e = self.encoder(x)
        out_d = self.decoder(out_e)
        return self.head(out_d)


class _PatchDiscriminator(nn.Module):
    """PatchGAN discriminator."""

    def __init__(self, c_in: int = 3, c_hid: int = 64, n_layers: int = 3) -> None:
        super().__init__()
        model: list[nn.Module] = [_DownsamplingBlock(c_in, c_hid, use_norm=False)]
        n_prev = 1
        n_curr = 1
        for n in range(1, n_layers):
            n_prev = n_curr
            n_curr = min(2**n, 8)
            model.append(_DownsamplingBlock(c_hid * n_prev, c_hid * n_curr))
        n_prev = n_curr
        n_curr = min(2**n_layers, 8)
        model.append(_DownsamplingBlock(c_hid * n_prev, c_hid * n_curr, stride=1))
        model.append(nn.Conv2d(in_channels=c_hid * n_curr, out_channels=1, kernel_size=4, stride=1, padding=1))
        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class _PixelDiscriminator(nn.Module):
    """PixelGAN (1x1 PatchGAN) discriminator."""

    def __init__(self, c_in: int = 3, c_hid: int = 64) -> None:
        super().__init__()
        self.model = nn.Sequential(
            _DownsamplingBlock(c_in, c_hid, kernel_size=1, stride=1, padding=0, use_norm=False),
            _DownsamplingBlock(c_hid, c_hid * 2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=c_hid * 2, out_channels=1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class SAR2OpticalDiscriminator(nn.Module):
    """Conditional PatchGAN/PixelGAN discriminator for SAR2Optical."""

    def __init__(self, c_in: int = 6, c_hid: int = 64, mode: str = "patch", n_layers: int = 3) -> None:
        super().__init__()
        if mode == "pixel":
            self.model = _PixelDiscriminator(c_in=c_in, c_hid=c_hid)
        else:
            self.model = _PatchDiscriminator(c_in=c_in, c_hid=c_hid, n_layers=n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def init_weights(module: nn.Module) -> None:
    """Weight initialisation from the upstream SAR2Optical implementation."""

    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    if isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.constant_(module.bias, 0.0)
