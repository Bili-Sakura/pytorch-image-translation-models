# Credits: SPADE (Park et al., CVPR 2019) adapted for this project style.
#
"""SPADE generator."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.spade.blocks import SPADEResnetBlock


class SPADEGenerator(nn.Module):
    """SPADE semantic-to-image generator."""

    _UPSAMPLE_FACTORS = {
        "normal": 5,
        "more": 6,
        "most": 7,
    }

    def __init__(
        self,
        label_nc: int = 35,
        output_nc: int = 3,
        ngf: int = 64,
        image_size: int = 256,
        num_upsampling_layers: str = "normal",
    ) -> None:
        super().__init__()
        if num_upsampling_layers not in self._UPSAMPLE_FACTORS:
            raise ValueError(f"Unsupported num_upsampling_layers: {num_upsampling_layers}")

        self.label_nc = label_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.image_size = image_size
        self.num_upsampling_layers = num_upsampling_layers

        self.fc = nn.Conv2d(label_nc, 16 * ngf, kernel_size=3, padding=1)
        self.head_0 = SPADEResnetBlock(16 * ngf, 16 * ngf, label_nc)
        self.g_middle_0 = SPADEResnetBlock(16 * ngf, 16 * ngf, label_nc)
        self.g_middle_1 = SPADEResnetBlock(16 * ngf, 16 * ngf, label_nc)
        self.up_0 = SPADEResnetBlock(16 * ngf, 8 * ngf, label_nc)
        self.up_1 = SPADEResnetBlock(8 * ngf, 4 * ngf, label_nc)
        self.up_2 = SPADEResnetBlock(4 * ngf, 2 * ngf, label_nc)
        self.up_3 = SPADEResnetBlock(2 * ngf, 1 * ngf, label_nc)
        if num_upsampling_layers == "most":
            self.up_4 = SPADEResnetBlock(1 * ngf, ngf // 2, label_nc)
            conv_in = ngf // 2
        else:
            self.up_4 = None
            conv_in = ngf
        self.conv_img = nn.Conv2d(conv_in, output_nc, kernel_size=3, padding=1)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

    @property
    def latent_hw(self) -> int:
        factor = 2 ** self._UPSAMPLE_FACTORS[self.num_upsampling_layers]
        return max(1, self.image_size // factor)

    def forward(self, segmap: torch.Tensor) -> torch.Tensor:
        if segmap.ndim != 4:
            raise ValueError("segmap must be a 4D tensor (B, C, H, W)")
        if segmap.shape[1] != self.label_nc:
            raise ValueError(f"Expected segmap channels={self.label_nc}, got {segmap.shape[1]}")

        x = F.interpolate(segmap, size=(self.latent_hw, self.latent_hw), mode="nearest")
        x = self.fc(x)

        x = self.head_0(x, segmap)
        x = self.up(x)
        x = self.g_middle_0(x, segmap)

        if self.num_upsampling_layers in {"more", "most"}:
            x = self.up(x)
        x = self.g_middle_1(x, segmap)

        x = self.up(x)
        x = self.up_0(x, segmap)
        x = self.up(x)
        x = self.up_1(x, segmap)
        x = self.up(x)
        x = self.up_2(x, segmap)
        x = self.up(x)
        x = self.up_3(x, segmap)

        if self.up_4 is not None:
            x = self.up(x)
            x = self.up_4(x, segmap)

        return torch.tanh(self.conv_img(F.leaky_relu(x, 2e-1)))
