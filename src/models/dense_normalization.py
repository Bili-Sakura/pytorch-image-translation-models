# Copyright (c) 2024 Ming-Yang Ho, Che-Ming Wu, and Min-Sheng Wu
# All rights reserved.
#
# Adapted from https://github.com/Kaminyou/Dense-Normalization.
# Original source code is licensed under the AGPL License.
#
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.
"""Dense Normalization layers for ultra-high-resolution translation."""

from __future__ import annotations

import typing as t

import torch
import torch.nn as nn


class Interpolation3D:
    """3D interpolation helper used by Dense Normalization."""

    def __init__(self, channel: int) -> None:
        self.channel = channel
        self.is_init = False
        self.device: torch.device | None = None
        self.size: int | None = None
        self.half_size: int | None = None
        self.eps = 1e-7

    def init(self, size: int, device: torch.device) -> None:
        if self.is_init and self.size == size and self.device == device:
            return
        self.size = size
        self.half_size = size // 2
        self.device = device
        self._init_matrix()
        self.is_init = True

    def _init_matrix(self) -> None:
        if self.size is None or self.device is None or self.half_size is None:
            raise RuntimeError("Interpolation3D.init() must be called before use.")
        self.small_to_large = torch.arange(0.5, self.size + 0.5, 1, device=self.device)
        self.large_to_small = torch.arange(self.size - 0.5, 0, -1, device=self.device)

        denom = self.size * self.size
        self.top_left = (self.large_to_small * self.large_to_small.unsqueeze(0).T) / denom
        self.down_left = (self.large_to_small * self.small_to_large.unsqueeze(0).T) / denom
        self.top_right = (self.small_to_large * self.large_to_small.unsqueeze(0).T) / denom
        self.down_right = (self.small_to_large * self.small_to_large.unsqueeze(0).T) / denom

        self.top_left = self.top_left.contiguous()
        self.down_left = self.down_left.contiguous()
        self.top_right = self.top_right.contiguous()
        self.down_right = self.down_right.contiguous()

    def _top_left_corner(
        self,
        top_left_value: torch.Tensor,
        top_right_value: torch.Tensor,
        down_left_value: torch.Tensor,
        down_right_value: torch.Tensor,
    ) -> torch.Tensor:
        if self.half_size is None:
            raise RuntimeError("Interpolation3D.init() must be called before use.")
        return (
            top_left_value * self.top_left[-self.half_size :, -self.half_size :]
            + top_right_value * self.top_right[-self.half_size :, -self.half_size :]
            + down_left_value * self.down_left[-self.half_size :, -self.half_size :]
            + down_right_value * self.down_right[-self.half_size :, -self.half_size :]
        )

    def _top_right_corner(
        self,
        top_left_value: torch.Tensor,
        top_right_value: torch.Tensor,
        down_left_value: torch.Tensor,
        down_right_value: torch.Tensor,
    ) -> torch.Tensor:
        if self.half_size is None:
            raise RuntimeError("Interpolation3D.init() must be called before use.")
        return (
            top_left_value * self.top_left[-self.half_size :, : self.half_size]
            + top_right_value * self.top_right[-self.half_size :, : self.half_size]
            + down_left_value * self.down_left[-self.half_size :, : self.half_size]
            + down_right_value * self.down_right[-self.half_size :, : self.half_size]
        )

    def _down_left_corner(
        self,
        top_left_value: torch.Tensor,
        top_right_value: torch.Tensor,
        down_left_value: torch.Tensor,
        down_right_value: torch.Tensor,
    ) -> torch.Tensor:
        if self.half_size is None:
            raise RuntimeError("Interpolation3D.init() must be called before use.")
        return (
            top_left_value * self.top_left[: self.half_size, -self.half_size :]
            + top_right_value * self.top_right[: self.half_size, -self.half_size :]
            + down_left_value * self.down_left[: self.half_size, -self.half_size :]
            + down_right_value * self.down_right[: self.half_size, -self.half_size :]
        )

    def _down_right_corner(
        self,
        top_left_value: torch.Tensor,
        top_right_value: torch.Tensor,
        down_left_value: torch.Tensor,
        down_right_value: torch.Tensor,
    ) -> torch.Tensor:
        if self.half_size is None:
            raise RuntimeError("Interpolation3D.init() must be called before use.")
        return (
            top_left_value * self.top_left[: self.half_size, : self.half_size]
            + top_right_value * self.top_right[: self.half_size, : self.half_size]
            + down_left_value * self.down_left[: self.half_size, : self.half_size]
            + down_right_value * self.down_right[: self.half_size, : self.half_size]
        )

    def _interpolation_mean_table(
        self,
        y0x0: torch.Tensor,
        y0x1: torch.Tensor,
        y0x2: torch.Tensor,
        y1x0: torch.Tensor,
        y1x1: torch.Tensor,
        y1x2: torch.Tensor,
        y2x0: torch.Tensor,
        y2x1: torch.Tensor,
        y2x2: torch.Tensor,
    ) -> torch.Tensor:
        if self.size is None or self.half_size is None or self.device is None:
            raise RuntimeError("Interpolation3D.init() must be called before use.")
        table = torch.zeros((self.channel, self.size, self.size), device=self.device)
        table[:, : self.half_size, : self.half_size] = self._top_left_corner(
            y0x0,
            y0x1,
            y1x0,
            y1x1,
        )
        table[:, : self.half_size, self.half_size :] = self._top_right_corner(
            y0x1,
            y0x2,
            y1x1,
            y1x2,
        )
        table[:, self.half_size :, : self.half_size] = self._down_left_corner(y1x0, y1x1, y2x0, y2x1)
        table[:, self.half_size :, self.half_size :] = self._down_right_corner(
            y1x1,
            y1x2,
            y2x1,
            y2x2,
        )
        return table

    def _replace_inf_nan(self, matrix_3x3: torch.Tensor) -> torch.Tensor:
        return torch.where(
            torch.logical_or(torch.isinf(matrix_3x3), torch.isnan(matrix_3x3)),
            matrix_3x3[:, 1:2, 1:2],
            matrix_3x3,
        )

    def interpolation_mean_table(self, matrix_3x3: torch.Tensor) -> torch.Tensor:
        matrix_3x3 = self._replace_inf_nan(matrix_3x3)
        matrix_3x3 = matrix_3x3.unsqueeze(-1).unsqueeze(-1)
        return self._interpolation_mean_table(
            matrix_3x3[:, 0, 0, :, :],
            matrix_3x3[:, 0, 1, :, :],
            matrix_3x3[:, 0, 2, :, :],
            matrix_3x3[:, 1, 0, :, :],
            matrix_3x3[:, 1, 1, :, :],
            matrix_3x3[:, 1, 2, :, :],
            matrix_3x3[:, 2, 0, :, :],
            matrix_3x3[:, 2, 1, :, :],
            matrix_3x3[:, 2, 2, :, :],
        )

    def interpolation_std_table_inverse(self, matrix_3x3: torch.Tensor) -> torch.Tensor:
        matrix_3x3 = self._replace_inf_nan(matrix_3x3)
        matrix_3x3 = matrix_3x3.unsqueeze(-1).unsqueeze(-1)
        matrix_3x3 = 1 / (matrix_3x3 + self.eps)
        return self._interpolation_mean_table(
            matrix_3x3[:, 0, 0, :, :],
            matrix_3x3[:, 0, 1, :, :],
            matrix_3x3[:, 0, 2, :, :],
            matrix_3x3[:, 1, 0, :, :],
            matrix_3x3[:, 1, 1, :, :],
            matrix_3x3[:, 1, 2, :, :],
            matrix_3x3[:, 2, 0, :, :],
            matrix_3x3[:, 2, 1, :, :],
            matrix_3x3[:, 2, 2, :, :],
        )


class DenseInstanceNorm(nn.Module):
    """Dense Normalization layer that can fall back to instance normalization."""

    def __init__(self, out_channels: int, affine: bool = True) -> None:
        super().__init__()
        self.normal_instance_normalization = False
        self.collection_mode = False
        self.out_channels = out_channels
        self.interpolation3d = Interpolation3D(channel=out_channels)

        if affine:
            self.weight = nn.Parameter(
                torch.ones(size=(1, out_channels, 1, 1), requires_grad=True)
            )
            self.bias = nn.Parameter(
                torch.zeros(size=(1, out_channels, 1, 1), requires_grad=True)
            )
        else:
            self.weight = None
            self.bias = None

        self.mean_table: torch.Tensor | None = None
        self.std_table: torch.Tensor | None = None
        self.padded_mean_table: torch.Tensor | None = None
        self.padded_std_table: torch.Tensor | None = None

    def _device(self) -> torch.device:
        if self.weight is not None:
            return self.weight.device
        if self.bias is not None:
            return self.bias.device
        return torch.device("cpu")

    def _apply_affine(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight is None or self.bias is None:
            return x
        return x * self.weight + self.bias

    def init_collection(self, y_anchor_num: int, x_anchor_num: int) -> None:
        device = self._device()
        self.mean_table = torch.zeros(y_anchor_num, x_anchor_num, self.out_channels, device=device)
        self.std_table = torch.zeros(y_anchor_num, x_anchor_num, self.out_channels, device=device)

    def pad_table(self, padding: int = 1) -> None:
        if self.mean_table is None or self.std_table is None:
            raise RuntimeError("DenseInstanceNorm.init_collection() must be called before pad_table().")
        pad_func = nn.ReplicationPad2d((padding, padding, padding, padding))
        self.padded_mean_table = pad_func(self.mean_table.permute(2, 0, 1).unsqueeze(0))
        self.padded_std_table = pad_func(self.std_table.permute(2, 0, 1).unsqueeze(0))

    def forward_normal(self, x: torch.Tensor) -> torch.Tensor:
        x_std, x_mean = torch.std_mean(x, dim=(2, 3), keepdim=True)
        x = (x - x_mean) / x_std
        return self._apply_affine(x)

    def forward(
        self,
        x: torch.Tensor,
        y_anchor: t.Optional[int] = None,
        x_anchor: t.Optional[int] = None,
        padding: int = 1,
    ) -> torch.Tensor:
        if self.collection_mode:
            if y_anchor is None or x_anchor is None:
                raise ValueError("y_anchor and x_anchor are required in collection_mode.")
            _, _, h, w = x.shape
            self.interpolation3d.init(size=h, device=x.device)
            x_std, x_mean = torch.std_mean(x, dim=(2, 3))
            if self.mean_table is None or self.std_table is None:
                raise RuntimeError("DenseInstanceNorm.init_collection() must be called before collection_mode.")
            self.mean_table[y_anchor, x_anchor] = x_mean
            self.std_table[y_anchor, x_anchor] = x_std
            x_mean = x_mean.unsqueeze(-1).unsqueeze(-1)
            x_std = x_std.unsqueeze(-1).unsqueeze(-1)
            x = (x - x_mean) / x_std
            return self._apply_affine(x)

        if self.training or self.normal_instance_normalization or y_anchor is None or x_anchor is None:
            return self.forward_normal(x)

        if x.shape[0] != 1:
            raise ValueError("DenseInstanceNorm only supports batch size = 1 during inference.")
        if self.padded_mean_table is None or self.padded_std_table is None:
            raise RuntimeError("DenseInstanceNorm.pad_table() must be called before inference mode.")

        top = y_anchor
        down = y_anchor + 2 * padding + 1
        left = x_anchor
        right = x_anchor + 2 * padding + 1

        x_mean = self.padded_mean_table[:, :, top:down, left:right]
        x_std = self.padded_std_table[:, :, top:down, left:right]

        self.interpolation3d.init(size=x.shape[2], device=x.device)
        x_mean = self.interpolation3d.interpolation_mean_table(x_mean[0]).unsqueeze(0)
        x_std = self.interpolation3d.interpolation_std_table_inverse(x_std[0]).unsqueeze(0)

        x = (x - x_mean) * x_std
        return self._apply_affine(x)


class PrefetchDenseInstanceNorm(nn.Module):
    """Prefetching variant of DenseInstanceNorm for streaming inference."""

    def __init__(self, out_channels: int, affine: bool = True) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.interpolation3d = Interpolation3D(channel=out_channels)
        if affine:
            self.weight = nn.Parameter(
                torch.ones(size=(1, out_channels, 1, 1), requires_grad=True)
            )
            self.bias = nn.Parameter(
                torch.zeros(size=(1, out_channels, 1, 1), requires_grad=True)
            )
        else:
            self.weight = None
            self.bias = None
        self.pad_func = nn.ReplicationPad2d((1, 1, 1, 1))
        self.mean_table: torch.Tensor | None = None
        self.std_table: torch.Tensor | None = None
        self.padded_mean_table: torch.Tensor | None = None
        self.padded_std_table: torch.Tensor | None = None

    def _apply_affine(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight is None or self.bias is None:
            return x
        return x * self.weight + self.bias

    def init_collection(self, y_anchor_num: int, x_anchor_num: int) -> None:
        device = self.weight.device if self.weight is not None else torch.device("cpu")
        self.mean_table = torch.zeros(y_anchor_num, x_anchor_num, self.out_channels, device=device)
        self.std_table = torch.zeros(y_anchor_num, x_anchor_num, self.out_channels, device=device)
        self.pad_table()

    def pad_table(self, padding: int = 1) -> None:
        if self.mean_table is None or self.std_table is None:
            raise RuntimeError("PrefetchDenseInstanceNorm.init_collection() must be called before pad_table().")
        pad_func = nn.ReplicationPad2d((padding, padding, padding, padding))
        self.padded_mean_table = pad_func(self.mean_table.permute(2, 0, 1).unsqueeze(0))
        self.padded_std_table = pad_func(self.std_table.permute(2, 0, 1).unsqueeze(0))

    def forward_normal(self, x: torch.Tensor) -> torch.Tensor:
        x_std, x_mean = torch.std_mean(x, dim=(2, 3), keepdim=True)
        x = (x - x_mean) / x_std
        return self._apply_affine(x)

    def forward(
        self,
        x: torch.Tensor,
        y_anchor: int | None = None,
        x_anchor: int | None = None,
        padding: int = 1,
        pre_y_anchor: t.Optional[t.List[int]] = None,
        pre_x_anchor: t.Optional[t.List[int]] = None,
    ) -> torch.Tensor:
        if x.shape[0] <= 1:
            return self.forward_normal(x)

        n, _, h, _ = x.shape
        real_x, pre_x = torch.split(x, (1, n - 1), dim=0)

        self.interpolation3d.init(size=h, device=x.device)

        if pre_y_anchor is not None and pre_x_anchor is not None:
            pre_x_std, pre_x_mean = torch.std_mean(pre_x, dim=(2, 3))
            if self.mean_table is None or self.std_table is None:
                raise RuntimeError("PrefetchDenseInstanceNorm.init_collection() must be called before use.")
            for i, (sub_pre_y_anchor, sub_pre_x_anchor) in enumerate(zip(pre_y_anchor, pre_x_anchor)):
                if sub_pre_y_anchor == -1:
                    continue
                self.mean_table[sub_pre_y_anchor, sub_pre_x_anchor] = pre_x_mean[i]
                self.std_table[sub_pre_y_anchor, sub_pre_x_anchor] = pre_x_std[i]

            pre_x_mean = pre_x_mean.unsqueeze(-1).unsqueeze(-1)
            pre_x_std = pre_x_std.unsqueeze(-1).unsqueeze(-1)
            pre_x = (pre_x - pre_x_mean) / pre_x_std
            pre_x = self._apply_affine(pre_x)

        if y_anchor is not None and x_anchor is not None and y_anchor != -1 and x_anchor != -1:
            if self.padded_mean_table is None or self.padded_std_table is None:
                raise RuntimeError("PrefetchDenseInstanceNorm.pad_table() must be called before use.")
            top = y_anchor
            left = x_anchor
            down = y_anchor + 2 * padding + 1
            right = x_anchor + 2 * padding + 1

            x_mean = self.padded_mean_table[:, :, top:down, left:right].squeeze(0)
            x_std = self.padded_std_table[:, :, top:down, left:right].squeeze(0)
            x_mean_expand = (
                x_mean[:, 1, 1].unsqueeze(-1).unsqueeze(-1).expand(-1, 3, 3)
            )
            x_std_expand = (
                x_std[:, 1, 1].unsqueeze(-1).unsqueeze(-1).expand(-1, 3, 3)
            )
            x_mean = torch.where(x_mean == 0, x_mean_expand, x_mean)
            x_std = torch.where(x_std == 0, x_std_expand, x_std)
            x_mean = self.interpolation3d.interpolation_mean_table(x_mean).unsqueeze(0)
            x_std = self.interpolation3d.interpolation_std_table_inverse(x_std).unsqueeze(0)

            real_x = (real_x - x_mean) * x_std
            real_x = self._apply_affine(real_x)

        return torch.cat((real_x, pre_x), dim=0)


def not_use_dense_instance_norm(model: nn.Module) -> None:
    for _, layer in model.named_modules():
        if isinstance(layer, DenseInstanceNorm):
            layer.collection_mode = False
            layer.normal_instance_normalization = True


def init_dense_instance_norm(model: nn.Module, y_anchor_num: int, x_anchor_num: int) -> None:
    for _, layer in model.named_modules():
        if isinstance(layer, DenseInstanceNorm):
            layer.collection_mode = True
            layer.normal_instance_normalization = False
            layer.init_collection(y_anchor_num=y_anchor_num, x_anchor_num=x_anchor_num)


def use_dense_instance_norm(model: nn.Module, padding: int = 1) -> None:
    for _, layer in model.named_modules():
        if isinstance(layer, DenseInstanceNorm):
            layer.pad_table(padding=padding)
            layer.collection_mode = False
            layer.normal_instance_normalization = False


def init_prefetch_dense_instance_norm(model: nn.Module, y_anchor_num: int, x_anchor_num: int) -> None:
    for _, layer in model.named_modules():
        if isinstance(layer, PrefetchDenseInstanceNorm):
            layer.init_collection(y_anchor_num=y_anchor_num, x_anchor_num=x_anchor_num)
