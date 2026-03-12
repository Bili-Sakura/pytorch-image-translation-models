# Copyright (c) 2026 EarthBridge Team.
# Credits: FCDM (Kwon et al., CVPR 2026) - https://github.com/star-kwon/FCDM
# "Reviving ConvNeXt for Efficient Convolutional Diffusion Models"

"""FCDM model architectures: ConvNeXt-based diffusion backbone."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm over channels (NCHW -> NHWC for norm)."""

    def __init__(self, num_channels: int, eps: float = 1e-6, affine: bool = True) -> None:
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(-1).unsqueeze(-1)


# ---------------------------------------------------------------------------
# Embedding Layers
# ---------------------------------------------------------------------------


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class LabelEmbedder(nn.Module):
    """Embeds class labels; supports classifier-free guidance via dropout."""

    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float) -> None:
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + int(use_cfg_embedding), hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(
        self, labels: torch.Tensor, force_drop_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        return torch.where(drop_ids, torch.full_like(labels, self.num_classes), labels)

    def forward(
        self,
        labels: torch.Tensor,
        train: bool,
        force_drop_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        return self.embedding_table(labels)


# ---------------------------------------------------------------------------
# ConvNeXt Blocks
# ---------------------------------------------------------------------------


class GRN(nn.Module):
    """Global Response Normalization layer."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class ConvNeXtBlock(nn.Module):
    """ConvNeXt-style block with adaLN-Zero conditioning."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim, affine=False, eps=1e-6)
        self.pwconv1 = nn.Conv2d(dim, int(dim * mlp_ratio), 1)
        self.act = nn.GELU()
        self.grn = GRN(int(dim * mlp_ratio))
        self.pwconv2 = nn.Conv2d(int(dim * mlp_ratio), dim, 1)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 3 * dim, bias=True))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        h = self.dwconv(x)
        shift, scale, gate = self.adaLN_modulation(c).unsqueeze(2).unsqueeze(3).chunk(3, dim=1)
        h = self.norm(h)
        h = torch.addcmul(shift, h, scale + 1)
        h = self.pwconv1(h)
        h = self.act(h)
        h = self.grn(h)
        h = self.pwconv2(h)
        h = h * gate
        return x + h


class ConvFinalLayer(nn.Module):
    """Conv-style final output layer with adaLN."""

    def __init__(self, hidden_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm = LayerNorm2d(hidden_size, affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )
        self.conv = nn.Conv2d(hidden_size, out_channels, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        return self.conv(x)


class Downsample(nn.Module):
    """Spatial downsampling via convolution + pixel unshuffle."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch // 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Spatial upsampling via pixel shuffle + convolution."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ---------------------------------------------------------------------------
# FCDM (class-conditional)
# ---------------------------------------------------------------------------


class FCDM(nn.Module):
    """Fully Convolutional Diffusion Model: ConvNeXt backbone with class conditioning.

    Designed for latent-space diffusion (e.g. 4ch VAE latents). With learn_sigma=True,
    outputs [epsilon, log_var] for learned variance during sampling.
    """

    def __init__(
        self,
        in_channels: int = 4,
        hidden_size: int = 1152,
        depth: tuple[int, ...] = (2, 5, 8, 5, 2),
        mlp_ratio: float = 3.0,
        class_dropout_prob: float = 0.1,
        num_classes: int = 1000,
        learn_sigma: bool = True,
        **kwargs: object,
    ) -> None:
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels

        self.t_embedder_1 = TimestepEmbedder(hidden_size)
        self.y_embedder_1 = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        self.t_embedder_2 = TimestepEmbedder(hidden_size * 2)
        self.y_embedder_2 = LabelEmbedder(num_classes, hidden_size * 2, class_dropout_prob)
        self.t_embedder_3 = TimestepEmbedder(hidden_size * 4)
        self.y_embedder_3 = LabelEmbedder(num_classes, hidden_size * 4, class_dropout_prob)

        self.x_embedder = nn.Conv2d(in_channels, hidden_size, kernel_size=3, stride=1, padding=1)

        # Encoder
        self.encoder_level_1 = nn.ModuleList(
            [ConvNeXtBlock(hidden_size, mlp_ratio=mlp_ratio) for _ in range(depth[0])]
        )
        self.down1_2 = Downsample(hidden_size, hidden_size * 2)
        self.encoder_level_2 = nn.ModuleList(
            [ConvNeXtBlock(hidden_size * 2, mlp_ratio=mlp_ratio) for _ in range(depth[1])]
        )
        self.down2_3 = Downsample(hidden_size * 2, hidden_size * 4)
        self.latent = nn.ModuleList(
            [ConvNeXtBlock(hidden_size * 4, mlp_ratio=mlp_ratio) for _ in range(depth[2])]
        )
        self.up3_2 = Upsample(hidden_size * 4, hidden_size * 2)
        self.reduce_chans_2 = nn.Conv2d(hidden_size * 4, hidden_size * 2, kernel_size=1)
        self.decoder_level_2 = nn.ModuleList(
            [ConvNeXtBlock(hidden_size * 2, mlp_ratio=mlp_ratio) for _ in range(depth[3])]
        )
        self.up2_1 = Upsample(hidden_size * 2, hidden_size)
        self.reduce_chans_1 = nn.Conv2d(hidden_size * 2, hidden_size, kernel_size=1)
        self.decoder_level_1 = nn.ModuleList(
            [ConvNeXtBlock(hidden_size, mlp_ratio=mlp_ratio) for _ in range(depth[4])]
        )
        self.output_layer = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1)
        self.final_layer = ConvFinalLayer(hidden_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.constant_(self.x_embedder.bias, 0)

        for embedder in [self.y_embedder_1, self.y_embedder_2, self.y_embedder_3]:
            nn.init.normal_(embedder.embedding_table.weight, std=0.02)

        for embedder in [self.t_embedder_1, self.t_embedder_2, self.t_embedder_3]:
            nn.init.normal_(embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(embedder.mlp[2].weight, std=0.02)

        blocks = (
            list(self.encoder_level_1)
            + list(self.encoder_level_2)
            + list(self.latent)
            + list(self.decoder_level_2)
            + list(self.decoder_level_1)
        )
        for block in blocks:
            if hasattr(block, "adaLN_modulation"):
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.conv.weight, 0)
        nn.init.constant_(self.final_layer.conv.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        x_emb = self.x_embedder(x)
        c1 = self.t_embedder_1(t) + self.y_embedder_1(y, self.training)
        c2 = self.t_embedder_2(t) + self.y_embedder_2(y, self.training)
        c3 = self.t_embedder_3(t) + self.y_embedder_3(y, self.training)

        out_enc_level1 = x_emb
        for block in self.encoder_level_1:
            out_enc_level1 = block(out_enc_level1, c1)
        inp_enc_level2 = self.down1_2(out_enc_level1)

        out_enc_level2 = inp_enc_level2
        for block in self.encoder_level_2:
            out_enc_level2 = block(out_enc_level2, c2)
        inp_enc_level3 = self.down2_3(out_enc_level2)

        latent = inp_enc_level3
        for block in self.latent:
            latent = block(latent, c3)

        inp_dec_level2 = self.up3_2(latent)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], dim=1)
        out_dec_level2 = self.reduce_chans_2(inp_dec_level2)
        for block in self.decoder_level_2:
            out_dec_level2 = block(out_dec_level2, c2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], dim=1)
        out_dec_level1 = self.reduce_chans_1(inp_dec_level1)
        for block in self.decoder_level_1:
            out_dec_level1 = block(out_dec_level1, c1)

        x_out = self.output_layer(out_dec_level1)
        return self.final_layer(x_out, c1)

    def forward_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        cfg_scale: float,
    ) -> torch.Tensor:
        """Classifier-free guidance: batch conditional + unconditional."""
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        eps, rest = model_out[:, : self.in_channels], model_out[:, self.in_channels :]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


# ---------------------------------------------------------------------------
# FCDM Image-Conditioned (for translation)
# ---------------------------------------------------------------------------


class FCDMImageCond(nn.Module):
    """FCDM backbone with image conditioning (concat) for image-to-image translation.

    Uses concatenation of condition latent with noisy input instead of class labels.
    """

    def __init__(
        self,
        in_channels: int = 4,
        condition_channels: int = 4,
        hidden_size: int = 512,
        depth: tuple[int, ...] = (2, 4, 8, 4, 2),
        mlp_ratio: float = 3.0,
        learn_sigma: bool = False,
        **kwargs: object,
    ) -> None:
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.condition_channels = condition_channels
        total_in = in_channels + condition_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels

        self.t_embedder_1 = TimestepEmbedder(hidden_size)
        self.t_embedder_2 = TimestepEmbedder(hidden_size * 2)
        self.t_embedder_3 = TimestepEmbedder(hidden_size * 4)
        self.x_embedder = nn.Conv2d(total_in, hidden_size, kernel_size=3, stride=1, padding=1)

        self.encoder_level_1 = nn.ModuleList(
            [ConvNeXtBlock(hidden_size, mlp_ratio=mlp_ratio) for _ in range(depth[0])]
        )
        self.down1_2 = Downsample(hidden_size, hidden_size * 2)
        self.encoder_level_2 = nn.ModuleList(
            [ConvNeXtBlock(hidden_size * 2, mlp_ratio=mlp_ratio) for _ in range(depth[1])]
        )
        self.down2_3 = Downsample(hidden_size * 2, hidden_size * 4)
        self.latent = nn.ModuleList(
            [ConvNeXtBlock(hidden_size * 4, mlp_ratio=mlp_ratio) for _ in range(depth[2])]
        )
        self.up3_2 = Upsample(hidden_size * 4, hidden_size * 2)
        self.reduce_chans_2 = nn.Conv2d(hidden_size * 4, hidden_size * 2, kernel_size=1)
        self.decoder_level_2 = nn.ModuleList(
            [ConvNeXtBlock(hidden_size * 2, mlp_ratio=mlp_ratio) for _ in range(depth[3])]
        )
        self.up2_1 = Upsample(hidden_size * 2, hidden_size)
        self.reduce_chans_1 = nn.Conv2d(hidden_size * 2, hidden_size, kernel_size=1)
        self.decoder_level_1 = nn.ModuleList(
            [ConvNeXtBlock(hidden_size, mlp_ratio=mlp_ratio) for _ in range(depth[4])]
        )
        self.output_layer = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1)
        self.final_layer = ConvFinalLayer(hidden_size, self.out_channels)

        self._init_weights()

    def _init_weights(self) -> None:
        def _basic_init(m: nn.Module) -> None:
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(_basic_init)
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.constant_(self.x_embedder.bias, 0)

        for embedder in [self.t_embedder_1, self.t_embedder_2, self.t_embedder_3]:
            nn.init.normal_(embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(embedder.mlp[2].weight, std=0.02)

        blocks = (
            list(self.encoder_level_1)
            + list(self.encoder_level_2)
            + list(self.latent)
            + list(self.decoder_level_2)
            + list(self.decoder_level_1)
        )
        for block in blocks:
            if hasattr(block, "adaLN_modulation"):
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.conv.weight, 0)
        nn.init.constant_(self.final_layer.conv.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        x_in = torch.cat([x, cond], dim=1)
        x_emb = self.x_embedder(x_in)
        c1 = self.t_embedder_1(t)
        c2 = self.t_embedder_2(t)
        c3 = self.t_embedder_3(t)

        out_enc_level1 = x_emb
        for block in self.encoder_level_1:
            out_enc_level1 = block(out_enc_level1, c1)
        inp_enc_level2 = self.down1_2(out_enc_level1)

        out_enc_level2 = inp_enc_level2
        for block in self.encoder_level_2:
            out_enc_level2 = block(out_enc_level2, c2)
        inp_enc_level3 = self.down2_3(out_enc_level2)

        latent = inp_enc_level3
        for block in self.latent:
            latent = block(latent, c3)

        inp_dec_level2 = self.up3_2(latent)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], dim=1)
        out_dec_level2 = self.reduce_chans_2(inp_dec_level2)
        for block in self.decoder_level_2:
            out_dec_level2 = block(out_dec_level2, c2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], dim=1)
        out_dec_level1 = self.reduce_chans_1(inp_dec_level1)
        for block in self.decoder_level_1:
            out_dec_level1 = block(out_dec_level1, c1)

        x_out = self.output_layer(out_dec_level1)
        return self.final_layer(x_out, c1)


# ---------------------------------------------------------------------------
# Preset configs
# ---------------------------------------------------------------------------


def FCDM_S(**kwargs: object) -> FCDM:
    return FCDM(hidden_size=128, depth=(2, 4, 8, 4, 2), **kwargs)


def FCDM_B(**kwargs: object) -> FCDM:
    return FCDM(hidden_size=256, depth=(2, 4, 8, 4, 2), **kwargs)


def FCDM_L(**kwargs: object) -> FCDM:
    return FCDM(hidden_size=512, depth=(2, 4, 8, 4, 2), **kwargs)


def FCDM_XL(**kwargs: object) -> FCDM:
    return FCDM(hidden_size=512, depth=(3, 6, 12, 6, 3), **kwargs)


FCDM_MODELS = {
    "FCDM-S": FCDM_S,
    "FCDM-B": FCDM_B,
    "FCDM-L": FCDM_L,
    "FCDM-XL": FCDM_XL,
}
