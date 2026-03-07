# Copyright (c) 2026 EarthBridge Team.
# Credits: Adapted from E3Diff (Qin et al., IEEE GRSL 2024).
# Original: https://github.com/DeepSARRS/E3Diff

"""Network architectures and losses for E3Diff.

Includes:

* ``CPEN`` – Conditional Prior Enhancement Network.
* ``E3DiffUNet`` – Conditional denoising U-Net.
* ``FocalFrequencyLoss`` – Frequency-domain loss (Jiang et al., ICCV 2021).
* ``_NLayerDiscriminator`` – PatchGAN discriminator for Stage 2.
* ``_GANLoss`` – GAN loss (lsgan / vanilla).
* ``_init_weights`` – Weight initialisation helper.
* Internal building blocks (``_PositionalEncoding``, ``_ResnetBlock``, etc.).
"""

from __future__ import annotations

import math
from inspect import isfunction

import torch
import torch.nn as nn
from torch import einsum
from torch.nn import init


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def _exists(x: object) -> bool:
    return x is not None


def _default(val, d):
    if _exists(val):
        return val
    return d() if isfunction(d) else d


# ---------------------------------------------------------------------------
# Focal Frequency Loss
# ---------------------------------------------------------------------------


class FocalFrequencyLoss(nn.Module):
    """Focal Frequency Loss for image synthesis (Jiang et al., ICCV 2021).

    Ref: https://arxiv.org/pdf/2012.12821.pdf

    Parameters
    ----------
    loss_weight : float
        Overall weight applied to the loss value.
    alpha : float
        Scaling factor for the spectrum weight matrix.
    patch_factor : int
        Factor to crop patches for patch-based loss. ``1`` uses the full image.
    ave_spectrum : bool
        Average spectrum over the mini-batch before computing the loss.
    log_matrix : bool
        Apply log-scaling to the spectrum weight matrix.
    batch_matrix : bool
        Normalise the weight matrix using batch-level statistics.
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        alpha: float = 1.0,
        patch_factor: int = 1,
        ave_spectrum: bool = False,
        log_matrix: bool = False,
        batch_matrix: bool = False,
    ) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix

    def _tensor2freq(self, x: torch.Tensor) -> torch.Tensor:
        pf = self.patch_factor
        _, _, h, w = x.shape
        assert h % pf == 0 and w % pf == 0, (
            "patch_factor must divide both image height and width"
        )
        patch_h, patch_w = h // pf, w // pf
        patches = [
            x[:, :, i * patch_h : (i + 1) * patch_h, j * patch_w : (j + 1) * patch_w]
            for i in range(pf)
            for j in range(pf)
        ]
        y = torch.stack(patches, dim=1)
        freq = torch.fft.fft2(y, norm="ortho")
        return torch.stack([freq.real, freq.imag], dim=-1)

    def _loss_formulation(
        self,
        recon_freq: torch.Tensor,
        real_freq: torch.Tensor,
        matrix: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if matrix is not None:
            weight_matrix = matrix.detach()
        else:
            matrix_tmp = (recon_freq - real_freq) ** 2
            matrix_tmp = (matrix_tmp[..., 0] + matrix_tmp[..., 1]).sqrt() ** self.alpha
            if self.log_matrix:
                matrix_tmp = torch.log(matrix_tmp + 1.0)
            if self.batch_matrix:
                matrix_tmp = matrix_tmp / matrix_tmp.max()
            else:
                # Normalise per (batch, patch, channel) retaining spatial dims
                denom = matrix_tmp.max(-1).values.max(-1).values[:, :, :, None, None]
                matrix_tmp = matrix_tmp / (denom + 1e-8)
            matrix_tmp = matrix_tmp.nan_to_num(0.0).clamp(0.0, 1.0)
            weight_matrix = matrix_tmp.detach()
        tmp = (recon_freq - real_freq) ** 2
        freq_distance = tmp[..., 0] + tmp[..., 1]
        return torch.mean(weight_matrix * freq_distance)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        matrix: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the focal frequency loss.

        Parameters
        ----------
        pred : Tensor ``[N, C, H, W]``
            Predicted images.
        target : Tensor ``[N, C, H, W]``
            Ground-truth images.
        matrix : Tensor, optional
            Pre-computed spectrum weight matrix. Computed online if ``None``.
        """
        pred_freq = self._tensor2freq(pred)
        target_freq = self._tensor2freq(target)
        if self.ave_spectrum:
            pred_freq = pred_freq.mean(0, keepdim=True)
            target_freq = target_freq.mean(0, keepdim=True)
        return self._loss_formulation(pred_freq, target_freq, matrix) * self.loss_weight


# ---------------------------------------------------------------------------
# UNet building blocks
# ---------------------------------------------------------------------------


class _PositionalEncoding(nn.Module):
    """Sinusoidal noise-level encoding."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, noise_level: torch.Tensor) -> torch.Tensor:
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype, device=noise_level.device) / count
        enc = noise_level.unsqueeze(1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        return torch.cat([torch.sin(enc), torch.cos(enc)], dim=-1)


class _FeatureWiseAffine(nn.Module):
    """Modulate spatial features with a noise-level embedding."""

    def __init__(self, in_channels: int, out_channels: int, use_affine: bool = False) -> None:
        super().__init__()
        self.use_affine = use_affine
        self.noise_func = nn.Linear(in_channels, out_channels * (1 + use_affine))

    def forward(self, x: torch.Tensor, noise_embed: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        if self.use_affine:
            gamma, beta = self.noise_func(noise_embed).view(b, -1, 1, 1).chunk(2, dim=1)
            return (1 + gamma) * x + beta
        return x + self.noise_func(noise_embed).view(b, -1, 1, 1)


class _Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class _Upsample(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.up(x))


class _Downsample(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _Block(nn.Module):
    def __init__(self, dim: int, dim_out: int, groups: int = 32, dropout: float = 0.0) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            _Swish(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _ResnetBlock(nn.Module):
    """Residual block conditioned on a noise-level embedding and a spatial feature map."""

    def __init__(
        self,
        dim: int,
        dim_out: int,
        noise_level_emb_dim: int | None = None,
        dropout: float = 0.0,
        use_affine_level: bool = False,
        norm_groups: int = 32,
    ) -> None:
        super().__init__()
        self.noise_func = _FeatureWiseAffine(noise_level_emb_dim, dim_out, use_affine_level)
        self.c_proj = nn.Conv2d(dim_out, dim_out, 1)
        self.block1 = _Block(dim, dim_out, groups=norm_groups)
        self.block2 = _Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        h = self.noise_func(h, time_emb)
        h = self.block2(h)
        h = self.c_proj(c) + h
        return h + self.res_conv(x)


class _SelfAttention(nn.Module):
    def __init__(self, in_channel: int, n_head: int = 1, norm_groups: int = 32) -> None:
        super().__init__()
        self.n_head = n_head
        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n_head = self.n_head
        head_dim = c // n_head
        norm = self.norm(x)
        qkv = self.qkv(norm).view(b, n_head, head_dim * 3, h, w)
        q, k, v = qkv.chunk(3, dim=2)
        attn = einsum("bnchw, bncyx -> bnhwyx", q, k).contiguous() / math.sqrt(head_dim)
        attn = attn.view(b, n_head, h, w, -1).softmax(-1).view(b, n_head, h, w, h, w)
        out = einsum("bnhwyx, bncyx -> bnchw", attn, v).contiguous()
        return self.out(out.view(b, c, h, w)) + x


class _ResnetBlocWithAttn(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        *,
        noise_level_emb_dim: int | None = None,
        norm_groups: int = 32,
        dropout: float = 0.0,
        with_attn: bool = False,
    ) -> None:
        super().__init__()
        self.with_attn = with_attn
        self.res_block = _ResnetBlock(dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = _SelfAttention(dim_out, norm_groups=norm_groups)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = self.res_block(x, time_emb, c)
        if self.with_attn:
            x = self.attn(x)
        return x


# ---------------------------------------------------------------------------
# Plain ResNet block (no noise conditioning) used inside CPEN
# ---------------------------------------------------------------------------


class _ResBlockPlain(nn.Module):
    def __init__(self, dim: int, dim_out: int, dropout: float = 0.0, norm_groups: int = 32) -> None:
        super().__init__()
        self.block1 = _Block(dim, dim_out, groups=norm_groups)
        self.block2 = _Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        h = self.block2(h)
        return h + self.res_conv(x)


# ---------------------------------------------------------------------------
# CPEN – Conditional Prior Enhancement Network
# ---------------------------------------------------------------------------


class CPEN(nn.Module):
    """Conditional Prior Enhancement Network.

    Multi-scale encoder for the SAR conditioning image. Produces five feature
    maps at spatial resolutions H, H/2, H/4, H/8, H/16 with channel widths
    matching the E3DiffUNet levels (64, 128, 256, 512, 1024 by default).

    The original implementation used ``SoftPool`` (an external C++/CUDA
    extension). This version uses ``AvgPool2d`` instead to remain fully
    self-contained while preserving the multi-scale structure.

    Parameters
    ----------
    in_channel : int
        Number of input channels of the conditioning image (e.g. 1 for
        single-polarisation SAR, 3 for RGB/multi-spectral).
    base_ch : int
        Base channel width. Levels use [1×, 2×, 4×, 8×, 16×] of *base_ch*.
        Must match the ``inner_channel`` of the associated :class:`E3DiffUNet`.
    """

    def __init__(self, in_channel: int = 1, base_ch: int = 64) -> None:
        super().__init__()
        ch = base_ch
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.E1 = nn.Sequential(nn.Conv2d(in_channel, ch, 3, padding=1), _Swish())
        self.E2 = nn.Sequential(
            _ResBlockPlain(ch, ch * 2, norm_groups=16),
            _ResBlockPlain(ch * 2, ch * 2, norm_groups=16),
        )
        self.E3 = nn.Sequential(
            _ResBlockPlain(ch * 2, ch * 4, norm_groups=16),
            _ResBlockPlain(ch * 4, ch * 4, norm_groups=16),
        )
        self.E4 = nn.Sequential(
            _ResBlockPlain(ch * 4, ch * 8, norm_groups=16),
            _ResBlockPlain(ch * 8, ch * 8, norm_groups=16),
        )
        self.E5 = nn.Sequential(
            _ResBlockPlain(ch * 8, ch * 8, norm_groups=16),
            _ResBlockPlain(ch * 8, ch * 16, norm_groups=16),
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return five multi-scale feature maps.

        Returns
        -------
        c1 : Tensor ``[B, base_ch,    H,    W]``
        c2 : Tensor ``[B, 2*base_ch,  H/2,  W/2]``
        c3 : Tensor ``[B, 4*base_ch,  H/4,  W/4]``
        c4 : Tensor ``[B, 8*base_ch,  H/8,  W/8]``
        c5 : Tensor ``[B, 16*base_ch, H/16, W/16]``
        """
        c1 = self.E1(x)
        c2 = self.E2(self.pool(c1))
        c3 = self.E3(self.pool(c2))
        c4 = self.E4(self.pool(c3))
        c5 = self.E5(self.pool(c4))
        return c1, c2, c3, c4, c5


# ---------------------------------------------------------------------------
# E3DiffUNet
# ---------------------------------------------------------------------------


class E3DiffUNet(nn.Module):
    """Conditional denoising U-Net for E3Diff.

    Receives the concatenation of the conditioning image and the noisy target
    image as input (``condition_ch + out_channel`` channels total). The
    condition is peeled off the front, passed through the CPEN multi-scale
    encoder, and injected at each U-Net residual block.  The noisy target
    image (``out_channel`` channels) is the signal that travels through the
    U-Net layers.

    Parameters
    ----------
    out_channel : int
        Number of channels in the noisy (and predicted) target image.
    inner_channel : int
        Base channel width; multiplied by *channel_mults* at each scale.
    norm_groups : int
        Number of groups for ``GroupNorm``.
    channel_mults : tuple[int, ...]
        Per-scale channel multipliers. Length determines the number of UNet
        scales. Must have exactly **5** entries so that CPEN features align.
    attn_res : tuple[int, ...]
        Spatial resolutions at which to insert self-attention.
    res_blocks : int
        Number of residual blocks per scale.
    dropout : float
        Dropout probability inside residual blocks.
    image_size : int
        Input spatial resolution (used only to determine attention positions).
    condition_ch : int
        Number of channels in the conditioning image (= CPEN input channels).
    """

    def __init__(
        self,
        out_channel: int = 3,
        inner_channel: int = 64,
        norm_groups: int = 32,
        channel_mults: tuple[int, ...] = (1, 2, 4, 8, 16),
        attn_res: tuple[int, ...] = (),
        res_blocks: int = 1,
        dropout: float = 0.0,
        image_size: int = 256,
        condition_ch: int = 3,
    ) -> None:
        super().__init__()
        if len(channel_mults) != 5:
            raise ValueError(
                f"E3DiffUNet requires exactly 5 channel_mults entries to align with CPEN, "
                f"got {len(channel_mults)}."
            )

        noise_level_ch = inner_channel
        self.noise_level_mlp = nn.Sequential(
            _PositionalEncoding(inner_channel),
            nn.Linear(inner_channel, inner_channel * 4),
            _Swish(),
            nn.Linear(inner_channel * 4, inner_channel),
        )

        self.res_blocks = res_blocks
        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels: list[int] = [pre_channel]
        now_res = image_size

        # The initial conv processes only the noisy image (out_channel channels);
        # the conditioning image is handled by CPEN separately.
        # ---- Encoder (downs) ----
        downs: list[nn.Module] = [nn.Conv2d(out_channel, inner_channel, 3, padding=1)]
        for ind in range(num_mults):
            is_last = ind == num_mults - 1
            use_attn = now_res in attn_res
            ch = inner_channel * channel_mults[ind]
            for _ in range(res_blocks):
                downs.append(
                    _ResnetBlocWithAttn(
                        pre_channel,
                        ch,
                        noise_level_emb_dim=noise_level_ch,
                        norm_groups=norm_groups,
                        dropout=dropout,
                        with_attn=use_attn,
                    )
                )
                feat_channels.append(ch)
                pre_channel = ch
            if not is_last:
                downs.append(_Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res //= 2
        self.downs = nn.ModuleList(downs)

        # ---- Bottleneck (mid) ----
        self.mid = nn.ModuleList(
            [
                _ResnetBlocWithAttn(
                    pre_channel,
                    pre_channel,
                    noise_level_emb_dim=noise_level_ch,
                    norm_groups=norm_groups,
                    dropout=dropout,
                    with_attn=True,
                ),
                _ResnetBlocWithAttn(
                    pre_channel,
                    pre_channel,
                    noise_level_emb_dim=noise_level_ch,
                    norm_groups=norm_groups,
                    dropout=dropout,
                    with_attn=False,
                ),
            ]
        )

        # ---- Decoder (ups) ----
        ups: list[nn.Module] = []
        for ind in reversed(range(num_mults)):
            is_last = ind < 1
            use_attn = now_res in attn_res
            ch = inner_channel * channel_mults[ind]
            for _ in range(res_blocks + 1):
                ups.append(
                    _ResnetBlocWithAttn(
                        pre_channel + feat_channels.pop(),
                        ch,
                        noise_level_emb_dim=noise_level_ch,
                        norm_groups=norm_groups,
                        dropout=dropout,
                        with_attn=use_attn,
                    )
                )
                pre_channel = ch
            if not is_last:
                ups.append(_Upsample(pre_channel))
                now_res *= 2
        self.ups = nn.ModuleList(ups)

        self.final_conv = _Block(pre_channel, out_channel, groups=norm_groups)

        self.condition_encoder = CPEN(in_channel=condition_ch, base_ch=inner_channel)
        self.condition_ch = condition_ch

    def forward(self, x: torch.Tensor, noise_level: torch.Tensor) -> torch.Tensor:
        """Forward pass of the denoising U-Net.

        Parameters
        ----------
        x : Tensor ``[B, condition_ch + out_channel, H, W]``
            Concatenation of the conditioning image (first ``condition_ch``
            channels) and the noisy target image (last ``out_channel`` channels).
        noise_level : Tensor ``[B, 1]``
            Continuous sqrt-alpha-cumprod noise level.

        Returns
        -------
        Tensor ``[B, out_channel, H, W]``
            Predicted noise (Stage 1) or predicted ``x_0`` (Stage 2).
        """
        # Split conditioning image from the noisy target
        condition = x[:, : self.condition_ch, ...].clone()
        x = x[:, self.condition_ch :, ...]  # noisy target only

        c1, c2, c3, c4, c5 = self.condition_encoder(condition)
        c_base_down = [c1, c2, c3, c4, c5]

        # Build per-block condition list for the encoder
        c_down: list[torch.Tensor] = []
        for ci in c_base_down:
            for _ in range(self.res_blocks):
                c_down.append(ci)

        t = self.noise_level_mlp(noise_level)

        # Encoder
        feats: list[torch.Tensor] = []
        ci_idx = 0
        for layer in self.downs:
            if isinstance(layer, _ResnetBlocWithAttn):
                x = layer(x, t, c_down[ci_idx])
                ci_idx += 1
            else:
                x = layer(x)
            feats.append(x)

        # Bottleneck
        for layer in self.mid:
            if isinstance(layer, _ResnetBlocWithAttn):
                x = layer(x, t, c5)
            else:
                x = layer(x)

        # Build per-block condition list for the decoder (reversed)
        c_base_up = [c5, c4, c3, c2, c1]
        c_up: list[torch.Tensor] = []
        for ci in c_base_up:
            for _ in range(self.res_blocks + 1):
                c_up.append(ci)

        # Decoder
        ci_idx = 0
        for layer in self.ups:
            if isinstance(layer, _ResnetBlocWithAttn):
                x = layer(torch.cat([x, feats.pop()], dim=1), t, c_up[ci_idx])
                ci_idx += 1
            else:
                x = layer(x)

        return self.final_conv(x)


# ---------------------------------------------------------------------------
# PatchGAN discriminator (used in Stage 2 adversarial fine-tuning)
# ---------------------------------------------------------------------------


class _NLayerDiscriminator(nn.Module):
    """PatchGAN discriminator for Stage-2 adversarial fine-tuning."""

    def __init__(
        self,
        input_nc: int,
        ndf: int = 64,
        n_layers: int = 3,
    ) -> None:
        super().__init__()
        kw, padw = 4, 1
        sequence: list[nn.Module] = [
            nn.Conv2d(input_nc, ndf, kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        nf = 1
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_prev, ndf * nf, kw, stride=2, padding=padw, bias=False),
                nn.BatchNorm2d(ndf * nf),
                nn.LeakyReLU(0.2, True),
            ]
        nf_prev = nf
        nf = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_prev, ndf * nf, kw, stride=1, padding=padw, bias=False),
            nn.BatchNorm2d(ndf * nf),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * nf, 1, kw, stride=1, padding=padw),
        ]
        self.model = nn.Sequential(*sequence)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ---------------------------------------------------------------------------
# GAN loss
# ---------------------------------------------------------------------------


class _GANLoss(nn.Module):
    def __init__(self, gan_mode: str = "lsgan") -> None:
        super().__init__()
        self.gan_mode = gan_mode
        if gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        elif gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError(f"GAN mode '{gan_mode}' is not implemented.")
        self.register_buffer("real_label", torch.tensor(1.0))
        self.register_buffer("fake_label", torch.tensor(0.0))

    def forward(self, pred: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        target = (self.real_label if target_is_real else self.fake_label).expand_as(pred)
        return self.loss(pred, target)


# ---------------------------------------------------------------------------
# Weight initialisation
# ---------------------------------------------------------------------------


def _init_weights(net: nn.Module, init_type: str = "orthogonal") -> None:
    """Apply weight initialisation to *net* in-place."""

    def _init_func(m: nn.Module) -> None:
        cls = m.__class__.__name__
        if hasattr(m, "weight") and ("Conv" in cls or "Linear" in cls):
            if init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=1)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "normal":
                init.normal_(m.weight.data, 0.0, 0.02)
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif "BatchNorm2d" in cls:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)

    net.apply(_init_func)
