# Credits: StarGAN (Choi et al., CVPR 2018) - https://github.com/yunjey/stargan

"""StarGAN discriminator architecture."""

from __future__ import annotations

import torch
import torch.nn as nn


class StarGANDiscriminator(nn.Module):
    """PatchGAN discriminator with source and domain classification heads."""

    def __init__(self, image_size: int = 128, conv_dim: int = 64, c_dim: int = 5, repeat_num: int = 6) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.01),
        ]

        curr_dim = conv_dim
        for _ in range(1, repeat_num):
            layers += [
                nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.01),
            ]
            curr_dim *= 2

        kernel_size = int(image_size / (2**repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv_src = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_cls = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.main(x)
        out_src = self.conv_src(h)
        out_cls = self.conv_cls(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))

