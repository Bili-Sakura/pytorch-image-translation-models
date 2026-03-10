"""StarGAN generator architecture."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.stargan.blocks import StarGANResidualBlock


class StarGANGenerator(nn.Module):
    """StarGAN generator for multi-domain image translation."""

    def __init__(self, conv_dim: int = 64, c_dim: int = 5, repeat_num: int = 6) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(3 + c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
        ]

        curr_dim = conv_dim
        for _ in range(2):
            layers += [
                nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(curr_dim * 2, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            ]
            curr_dim *= 2

        for _ in range(repeat_num):
            layers.append(StarGANResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        for _ in range(2):
            layers += [
                nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(curr_dim // 2, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            ]
            curr_dim = curr_dim // 2

        layers += [
            nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Tanh(),
        ]
        self.main = nn.Sequential(*layers)
        self.c_dim = c_dim

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # Replicate domain labels spatially and concatenate with input image.
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        return self.main(x)

