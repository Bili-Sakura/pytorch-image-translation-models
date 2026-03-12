# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.
#
# SiD2 (Simpler Diffusion v2) credits:
#   Hoogeboom, E., Mensink, T., Heek, J., Lamerigts, K., Gao, R., & Salimans, T. (2025).
#   Simpler Diffusion (SiD2): 1.5 FID on ImageNet512 with pixel-space diffusion.
#   In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2025).

"""SiD2 Residual U-ViT – pixel-space diffusion backbone using diffusers blocks.

Uses ResnetBlock2D and BasicTransformerBlock with a single level-wise residual skip
(per SiD2 paper). No per-block timestep embedding.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from diffusers import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.resnet import ResnetBlock2D
from diffusers.utils import BaseOutput

__all__ = ["SiD2UNet"]


# ---------------------------------------------------------------------------
# Building blocks (diffusers-native)
# ---------------------------------------------------------------------------


class SiD2ResBlock2D(nn.Module):
    """ResnetBlock2D wrapper – no timestep conditioning (SiD2 uses continuous t)."""

    def __init__(self, dim: int, groups: int = 32) -> None:
        super().__init__()
        self.block = ResnetBlock2D(
            in_channels=dim,
            out_channels=dim,
            temb_channels=None,
            groups=groups,
            non_linearity="silu",
            eps=1e-6,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SiD2TransformerBlock(nn.Module):
    """BasicTransformerBlock wrapper for spatial feature maps."""

    def __init__(self, dim: int, num_heads: int = 8) -> None:
        super().__init__()
        self.block = BasicTransformerBlock(
            dim=dim,
            num_attention_heads=num_heads,
            attention_head_dim=dim // num_heads,
            dropout=0.0,
            cross_attention_dim=None,
            activation_fn="geglu",
            attention_bias=False,
            norm_type="layer_norm",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        x = self.block(x)
        return x.transpose(1, 2).reshape(B, C, H, W)


# ---------------------------------------------------------------------------
# SiD2 Residual U-ViT
# ---------------------------------------------------------------------------


class SiD2UNet(ModelMixin, ConfigMixin):
    """SiD2 Residual U-ViT with level-wise residual skip.

    Parameters
    ----------
    image_size : int
        Input spatial resolution.
    in_channels : int
        Input channels (typically 3).
    out_channels : int
        Output channels (typically equals in_channels).
    patch_size : int
        Patch size for embedding (2 recommended for SiD2).
    channels : tuple[int, ...]
        Channel widths per level.
    num_updown_blocks : tuple[int, ...]
        Number of blocks per encoder/decoder level.
    num_mid_blocks : int
        Number of transformer blocks in the bottleneck.
    block_types : tuple[str, ...]
        Per-level block type: ``"ResBlock"`` or ``"Transformer"``.
    num_heads : int
        Attention heads in transformer blocks.
    """

    @register_to_config
    def __init__(
        self,
        image_size: int = 512,
        in_channels: int = 3,
        out_channels: Optional[int] = None,
        patch_size: int = 2,
        channels: Tuple[int, ...] = (128, 256, 512, 1024),
        num_updown_blocks: Tuple[int, ...] = (3, 3, 3),
        num_mid_blocks: int = 16,
        block_types: Tuple[str, ...] = ("ResBlock", "ResBlock", "Transformer", "Transformer"),
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        out_channels = out_channels or in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.patch_embed = nn.Conv2d(in_channels, channels[0], kernel_size=patch_size, stride=patch_size)

        # Encoder
        self.down_blocks = nn.ModuleList()
        cur_ch = channels[0]
        for i, n in enumerate(num_updown_blocks):
            self.down_blocks.append(
                nn.ModuleList([
                    self._make_block(block_types[i % len(block_types)], cur_ch, num_heads)
                    for _ in range(n)
                ])
            )
            if i < len(num_updown_blocks) - 1:
                self.down_blocks.append(
                    nn.Sequential(
                        nn.AvgPool2d(2),
                        nn.Conv2d(cur_ch, channels[i + 1], kernel_size=1),
                    )
                )
                cur_ch = channels[i + 1]

        # Bottleneck
        self.mid_blocks = nn.ModuleList([
            self._make_block("Transformer", channels[-1], num_heads)
            for _ in range(num_mid_blocks)
        ])

        # Decoder (symmetric, with level-wise residual skip)
        self.up_blocks = nn.ModuleList()
        for i in reversed(range(len(num_updown_blocks))):
            self.up_blocks.append(
                nn.ModuleList([
                    self._make_block(block_types[i % len(block_types)], channels[i], num_heads)
                    for _ in range(num_updown_blocks[i])
                ])
            )
            if i > 0:
                self.up_blocks.append(
                    nn.Sequential(
                        nn.Conv2d(channels[i], channels[i - 1], kernel_size=1),
                        nn.Upsample(scale_factor=2, mode="nearest"),
                    )
                )

        self.final_conv = nn.Conv2d(channels[0], out_channels, kernel_size=1)

    def _make_block(self, block_type: str, dim: int, num_heads: int) -> nn.Module:
        if block_type == "ResBlock":
            return SiD2ResBlock2D(dim)
        return SiD2TransformerBlock(dim, num_heads)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
    ) -> BaseOutput:
        """Forward pass. *timestep* is accepted for API compatibility but unused."""
        h = self.patch_embed(sample)
        down_features: list[torch.Tensor] = []

        # Encoder
        for level in self.down_blocks:
            if isinstance(level, nn.ModuleList):
                for blk in level:
                    h = blk(h)
            else:
                h = level(h)
            down_features.append(h)

        # Bottleneck
        for blk in self.mid_blocks:
            h = blk(h)

        # Decoder + level-wise residual skip
        for level in self.up_blocks:
            if isinstance(level, nn.ModuleList):
                for blk in level:
                    h = blk(h)
            else:
                d_h = down_features.pop()
                h = level(h - d_h) + d_h

        return BaseOutput(sample=self.final_conv(h))
