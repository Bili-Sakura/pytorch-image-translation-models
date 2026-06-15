# Copyright (c) 2026 EarthBridge Team.
# Credits: F-LSeSim from Zheng et al. CVPR 2021 —
# https://github.com/lyndonzheng/F-LSeSim

"""Spatially-correlative loss for unpaired image-to-image translation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn import init


class ImageNetNormalization(nn.Module):
    """Normalize images to ImageNet statistics."""

    def __init__(self) -> None:
        super().__init__()
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        self.register_buffer("mean", mean.view(1, 3, 1, 1))
        self.register_buffer("std", std.view(1, 3, 1, 1))

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return (img - self.mean) / self.std


class VGG16FeatureExtractor(nn.Module):
    """VGG16 feature extractor for spatially-correlative loss."""

    _LAYER_KEYS = (
        "relu1_1",
        "relu1_2",
        "relu2_1",
        "relu2_2",
        "relu3_1",
        "relu3_2",
        "relu3_3",
        "relu4_1",
        "relu4_2",
        "relu4_3",
        "relu5_1",
        "relu5_2",
        "relu5_3",
    )

    def __init__(self) -> None:
        super().__init__()
        try:
            weights = models.VGG16_Weights.IMAGENET1K_V1
            features = models.vgg16(weights=weights).features
        except AttributeError:
            features = models.vgg16(pretrained=True).features

        blocks: dict[str, nn.Sequential] = {}
        ranges = [
            ("relu1_1", 0, 2),
            ("relu1_2", 2, 4),
            ("relu2_1", 4, 7),
            ("relu2_2", 7, 9),
            ("relu3_1", 9, 12),
            ("relu3_2", 12, 14),
            ("relu3_3", 14, 16),
            ("relu4_1", 16, 18),
            ("relu4_2", 18, 21),
            ("relu4_3", 21, 23),
            ("relu5_1", 23, 26),
            ("relu5_2", 26, 28),
            ("relu5_3", 28, 30),
        ]
        for name, start, end in ranges:
            block = nn.Sequential()
            for idx in range(start, end):
                block.add_module(str(idx), features[idx])
            blocks[name] = block
            self.add_module(name, block)

    def forward(
        self,
        x: torch.Tensor,
        layers: list[int] | None = None,
        *,
        encode_only: bool = False,
    ) -> dict[str, torch.Tensor] | list[torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        h = x
        for key in self._LAYER_KEYS:
            h = getattr(self, key)(h)
            out[key] = h

        if encode_only and layers:
            feats: list[torch.Tensor] = []
            for layer_idx, key in enumerate(self._LAYER_KEYS):
                if layer_idx in layers:
                    feats.append(out[key])
            return feats
        return out


class PatchSim(nn.Module):
    """Patch similarity map used by spatially-correlative loss."""

    def __init__(
        self,
        patch_nums: int = 256,
        patch_size: int | None = None,
        *,
        use_norm: bool = True,
    ) -> None:
        super().__init__()
        self.patch_nums = patch_nums
        self.patch_size = patch_size
        self.use_norm = use_norm

    def forward(
        self,
        feat: torch.Tensor,
        patch_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, C, H, W = feat.size()
        feat = feat - feat.mean(dim=(-2, -1), keepdim=True)
        if self.use_norm:
            feat = F.normalize(feat, dim=1)
        else:
            feat = feat / (C ** 0.5)

        query, key, patch_ids = self._select_patch(feat, patch_ids=patch_ids)
        patch_sim = query.bmm(key) if self.use_norm else torch.tanh(query.bmm(key) / 10)
        if patch_ids is not None:
            patch_sim = patch_sim.view(B, len(patch_ids), -1)
        return patch_sim, patch_ids

    def _select_patch(
        self,
        feat: torch.Tensor,
        patch_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        B, C, H, W = feat.size()
        pw = ph = self.patch_size
        feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)

        if self.patch_nums > 0:
            if patch_ids is None:
                patch_ids = torch.randperm(feat_reshape.size(1), device=feat.device)
                patch_ids = patch_ids[: int(min(self.patch_nums, patch_ids.size(0)))]
            feat_query = feat_reshape[:, patch_ids, :]
            feat_key: torch.Tensor
            num = feat_query.size(1)
            if pw is not None and ph is not None and pw < W and ph < H:
                pos_x = patch_ids // W
                pos_y = patch_ids % W
                left = pos_x - int(pw / 2)
                top = pos_y - int(ph / 2)
                left = torch.where(left > 0, left, torch.zeros_like(left))
                top = torch.where(top > 0, top, torch.zeros_like(top))
                start_x = torch.where(left > (W - pw), (W - pw) * torch.ones_like(left), left)
                start_y = torch.where(top > (H - ph), (H - ph) * torch.ones_like(top), top)
                keys = []
                for i in range(num):
                    keys.append(
                        feat[:, :, start_x[i] : start_x[i] + pw, start_y[i] : start_y[i] + ph]
                    )
                feat_key = torch.stack(keys, dim=0).permute(1, 0, 2, 3, 4)
                feat_key = feat_key.reshape(B * num, C, pw * ph)
                feat_query = feat_query.reshape(B * num, 1, C)
            else:
                feat_key = feat.reshape(B, C, H * W)
        else:
            feat_query = feat.reshape(B, C, H * W).permute(0, 2, 1)
            feat_key = feat.reshape(B, C, H * W)

        return feat_query, feat_key, patch_ids


class SpatialCorrelativeLoss(nn.Module):
    """Learnable patch-based spatially-correlative loss (F-LSeSim)."""

    def __init__(
        self,
        loss_mode: str = "cos",
        patch_nums: int = 256,
        patch_size: int = 32,
        *,
        use_norm: bool = True,
        use_conv: bool = True,
        init_type: str = "normal",
        init_gain: float = 0.02,
        temperature: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_sim = PatchSim(patch_nums=patch_nums, patch_size=patch_size, use_norm=use_norm)
        self.patch_size = patch_size
        self.patch_nums = patch_nums
        self.use_norm = use_norm
        self.use_conv = use_conv
        self.conv_init = False
        self.init_type = init_type
        self.init_gain = init_gain
        self.loss_mode = loss_mode
        self.temperature = temperature
        self.criterion = nn.L1Loss() if use_norm else nn.SmoothL1Loss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def update_init_(self) -> None:
        self.conv_init = True

    def _init_conv_weights(self, conv: nn.Module) -> None:
        for m in conv.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                if self.init_type == "normal":
                    init.normal_(m.weight.data, 0.0, self.init_gain)
                elif self.init_type == "xavier":
                    init.xavier_normal_(m.weight.data, gain=self.init_gain)
                if getattr(m, "bias", None) is not None:
                    init.constant_(m.bias.data, 0.0)

    def create_conv(self, feat: torch.Tensor, layer: int) -> None:
        input_nc = feat.size(1)
        output_nc = max(32, input_nc // 4)
        conv = nn.Sequential(
            nn.Conv2d(input_nc, output_nc, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(output_nc, output_nc, kernel_size=1),
        )
        conv.to(feat.device)
        self._init_conv_weights(conv)
        setattr(self, f"conv_{layer}", conv)

    def cal_sim(
        self,
        f_src: torch.Tensor,
        f_tgt: torch.Tensor,
        f_other: torch.Tensor | None = None,
        *,
        layer: int = 0,
        patch_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if self.use_conv:
            if not self.conv_init:
                self.create_conv(f_src, layer)
            conv = getattr(self, f"conv_{layer}")
            f_src, f_tgt = conv(f_src), conv(f_tgt)
            f_other = conv(f_other) if f_other is not None else None

        sim_src, patch_ids = self.patch_sim(f_src, patch_ids)
        sim_tgt, _ = self.patch_sim(f_tgt, patch_ids)
        sim_other = None
        if f_other is not None:
            sim_other, _ = self.patch_sim(f_other, patch_ids)
        return sim_src, sim_tgt, sim_other

    def compare_sim(
        self,
        sim_src: torch.Tensor,
        sim_tgt: torch.Tensor,
        sim_other: torch.Tensor | None,
    ) -> torch.Tensor:
        _, num, n = sim_src.size()
        if self.loss_mode == "info" or sim_other is not None:
            sim_src = F.normalize(sim_src, dim=-1)
            sim_tgt = F.normalize(sim_tgt, dim=-1)
            sim_other = F.normalize(sim_other, dim=-1)
            sam_neg1 = (sim_src.bmm(sim_other.permute(0, 2, 1))).view(-1, num) / self.temperature
            sam_neg2 = (sim_tgt.bmm(sim_other.permute(0, 2, 1))).view(-1, num) / self.temperature
            sam_self = (sim_src.bmm(sim_tgt.permute(0, 2, 1))).view(-1, num) / self.temperature
            sam_self = torch.cat([sam_self, sam_neg1, sam_neg2], dim=-1)
            labels = torch.arange(0, sam_self.size(0), dtype=torch.long, device=sim_src.device) % num
            return self.cross_entropy_loss(sam_self, labels)

        tgt_sorted, _ = sim_tgt.sort(dim=-1, descending=True)
        cutoff = int(n / 4)
        src = torch.where(sim_tgt < tgt_sorted[:, :, cutoff : cutoff + 1], 0 * sim_src, sim_src)
        tgt = torch.where(sim_tgt < tgt_sorted[:, :, cutoff : cutoff + 1], 0 * sim_tgt, sim_tgt)
        if self.loss_mode == "l1":
            return self.criterion((n / cutoff) * src, (n / cutoff) * tgt)
        if self.loss_mode == "cos":
            sim_pos = F.cosine_similarity(src, tgt, dim=-1)
            return self.criterion(torch.ones_like(sim_pos), sim_pos)
        raise NotImplementedError(f"loss_mode [{self.loss_mode}] is not implemented")

    def loss(
        self,
        f_src: torch.Tensor,
        f_tgt: torch.Tensor,
        f_other: torch.Tensor | None = None,
        *,
        layer: int = 0,
    ) -> torch.Tensor:
        sim_src, sim_tgt, sim_other = self.cal_sim(f_src, f_tgt, f_other, layer=layer)
        return self.compare_sim(sim_src, sim_tgt, sim_other)


def compute_spatial_correlative_loss(
    feature_net: VGG16FeatureExtractor,
    criterion: SpatialCorrelativeLoss,
    src: torch.Tensor,
    tgt: torch.Tensor,
    other: torch.Tensor | None,
    attn_layers: list[int],
) -> torch.Tensor:
    """Compute averaged spatially-correlative loss over VGG layers."""
    feats_src = feature_net(src, attn_layers, encode_only=True)
    feats_tgt = feature_net(tgt, attn_layers, encode_only=True)
    if other is not None:
        feats_other = feature_net(torch.flip(other, [2, 3]), attn_layers, encode_only=True)
    else:
        feats_other = [None for _ in attn_layers]

    total = torch.tensor(0.0, device=src.device)
    for layer_idx, (feat_src, feat_tgt, feat_oth) in enumerate(
        zip(feats_src, feats_tgt, feats_other)
    ):
        total = total + criterion.loss(feat_src, feat_tgt, feat_oth, layer=layer_idx).mean()

    if not criterion.conv_init:
        criterion.update_init_()
    return total / len(attn_layers)


class ImagePool:
    """Buffer that stores previously generated images for discriminator training."""

    def __init__(self, pool_size: int) -> None:
        self.pool_size = pool_size
        self.images: list[torch.Tensor] = []

    def query(self, images: torch.Tensor) -> torch.Tensor:
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images.data:
            image = torch.unsqueeze(image, 0)
            if len(self.images) < self.pool_size:
                self.images.append(image)
                return_images.append(image)
            else:
                if torch.rand(1).item() > 0.5:
                    idx = int(torch.randint(0, self.pool_size, (1,)).item())
                    tmp = self.images[idx].clone()
                    self.images[idx] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return torch.cat(return_images, dim=0)
