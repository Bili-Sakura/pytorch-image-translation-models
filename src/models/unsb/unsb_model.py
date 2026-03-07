# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""UNSB model components built on native PyTorch modules.

This module provides the four networks required by the UNSB (Unpaired Neural
Schrödinger Bridge) method from *Kim et al. "Unpaired Image-to-Image
Translation via Neural Schrödinger Bridge" (ICLR 2024)*:

* :class:`UNSBGenerator` – Time-conditional ResNet-based generator that maps
  source domain images through a multi-step stochastic bridge process.  Each
  ResNet block is conditioned on a timestep embedding **and** a latent noise
  vector via an adaptive modulation layer.
* :class:`UNSBDiscriminator` – Time-conditional PatchGAN discriminator that
  classifies whether images are real or generated *at each timestep*.
* :class:`UNSBEnergyNet` – Energy network (same architecture as the
  discriminator but with concatenated input/output pairs) that estimates
  importance weights for the Schrödinger Bridge loss.
* :class:`PatchSampleMLP` – Lightweight MLP projection head that samples
  spatial patches from multi-layer encoder features for the contrastive loss.

Loss modules:

* :class:`GANLoss` – Configurable GAN objective (LSGAN / Vanilla / WGAN-GP).
* :class:`PatchNCELoss` – Per-layer contrastive loss following
  *Park et al. "Contrastive Learning for Unpaired Image-to-Image Translation"
  (ECCV 2020)*.
"""

from __future__ import annotations

import functools
import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _get_norm_layer(norm_type: str = "instance"):
    """Return a normalisation layer factory."""
    if norm_type == "batch":
        return functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    if norm_type == "instance":
        return functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    if norm_type == "none":
        return lambda x: nn.Identity()
    raise NotImplementedError(f"Norm layer [{norm_type}] is not supported")


def _init_weights(net: nn.Module, init_type: str = "normal", init_gain: float = 0.02) -> None:
    """Initialise network weights (Normal / Xavier / Kaiming / Orthogonal)."""

    def _init_func(m: nn.Module) -> None:
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(f"Init method [{init_type}] not implemented")
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(_init_func)


def _get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    max_positions: int = 10000,
) -> torch.Tensor:
    """Sinusoidal timestep embedding (Vaswani et al.)."""
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1), mode="constant")
    return emb


def _get_filter(filt_size: int = 3) -> torch.Tensor:
    """1-D filter kernel for anti-aliased down/up-sampling."""
    kernels = {
        1: [1.0],
        2: [1.0, 1.0],
        3: [1.0, 2.0, 1.0],
        4: [1.0, 3.0, 3.0, 1.0],
        5: [1.0, 4.0, 6.0, 4.0, 1.0],
    }
    a = np.array(kernels.get(filt_size, kernels[3]))
    filt = torch.tensor(a[:, None] * a[None, :], dtype=torch.float32)
    return filt / filt.sum()


def _get_pad_layer(pad_type: str):
    """Return a padding layer class."""
    if pad_type in ("refl", "reflect"):
        return nn.ReflectionPad2d
    if pad_type in ("repl", "replicate"):
        return nn.ReplicationPad2d
    if pad_type == "zero":
        return nn.ZeroPad2d
    raise NotImplementedError(f"Pad type [{pad_type}] not recognised")


# ---------------------------------------------------------------------------
# Anti-aliased Downsample / Upsample
# ---------------------------------------------------------------------------

class _Downsample(nn.Module):
    """Anti-aliased downsampling with a low-pass filter."""

    def __init__(self, channels: int, pad_type: str = "reflect", filt_size: int = 3, stride: int = 2) -> None:
        super().__init__()
        self.filt_size = filt_size
        pad_sizes = [int(1.0 * (filt_size - 1) / 2)] * 2 + [int(np.ceil(1.0 * (filt_size - 1) / 2))] * 2
        self.pad_sizes = [pad_sizes[0], pad_sizes[2], pad_sizes[1], pad_sizes[3]]
        self.stride = stride
        self.channels = channels

        filt = _get_filter(filt_size)
        self.register_buffer("filt", filt[None, None, :, :].repeat(channels, 1, 1, 1))
        self.pad = _get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.filt_size == 1:
            return x[:, :, :: self.stride, :: self.stride]
        return F.conv2d(self.pad(x), self.filt, stride=self.stride, groups=x.shape[1])


class _Upsample(nn.Module):
    """Anti-aliased upsampling with a low-pass filter."""

    def __init__(self, channels: int, pad_type: str = "repl", filt_size: int = 4, stride: int = 2) -> None:
        super().__init__()
        self.filt_size = filt_size
        self.filt_odd = (filt_size % 2) == 1
        self.pad_size = int((filt_size - 1) / 2)
        self.stride = stride
        self.channels = channels

        filt = _get_filter(filt_size) * (stride ** 2)
        self.register_buffer("filt", filt[None, None, :, :].repeat(channels, 1, 1, 1))
        self.pad = _get_pad_layer(pad_type)([1, 1, 1, 1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ret = F.conv_transpose2d(
            self.pad(x), self.filt,
            stride=self.stride, padding=1 + self.pad_size,
            groups=x.shape[1],
        )[:, :, 1:, 1:]
        if self.filt_odd:
            return ret
        return ret[:, :, :-1, :-1]


# ---------------------------------------------------------------------------
# Timestep Embedding MLP
# ---------------------------------------------------------------------------

class _TimestepEmbedding(nn.Module):
    """MLP that maps sinusoidal timestep embeddings to a hidden space."""

    def __init__(self, embedding_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.main = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.LeakyReLU(0.2),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        temb = _get_timestep_embedding(t, self.embedding_dim)
        return self.main(temb)


# ---------------------------------------------------------------------------
# Pixel Norm (used in z-transform mapping)
# ---------------------------------------------------------------------------

class _PixelNorm(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)


# ---------------------------------------------------------------------------
# Adaptive modulation layer (FiLM-style)
# ---------------------------------------------------------------------------

class _AdaptiveLayer(nn.Module):
    """Feature-wise affine modulation conditioned on a style vector."""

    def __init__(self, in_channel: int, style_dim: int) -> None:
        super().__init__()
        self.style_net = nn.Linear(style_dim, in_channel * 2)
        self.style_net.bias.data[:in_channel] = 1
        self.style_net.bias.data[in_channel:] = 0

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        style = self.style_net(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)
        return gamma * x + beta


# ---------------------------------------------------------------------------
# Conditional ResNet block (time + noise conditioned)
# ---------------------------------------------------------------------------

class _ResnetBlockCond(nn.Module):
    """ResNet block conditioned on timestep embedding and latent noise vector.

    The block applies: Conv → +time_embed → Norm → AdaptiveLayer(z) → ReLU → Conv → Norm + skip.
    """

    def __init__(
        self,
        dim: int,
        norm_layer,
        use_dropout: bool,
        use_bias: bool,
        temb_dim: int,
        z_dim: int,
        padding_type: str = "reflect",
    ) -> None:
        super().__init__()

        # First conv path
        self.conv_block = nn.ModuleList()
        p = 0
        if padding_type == "reflect":
            self.conv_block.append(nn.ReflectionPad2d(1))
        elif padding_type == "replicate":
            self.conv_block.append(nn.ReplicationPad2d(1))
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError(f"Padding [{padding_type}] not implemented")
        self.conv_block.append(nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias))
        self.conv_block.append(norm_layer(dim))

        # Adaptive modulation
        self.adaptive = _AdaptiveLayer(dim, z_dim)

        # Second conv path
        self.conv_fin = nn.ModuleList()
        self.conv_fin.append(nn.ReLU(True))
        if use_dropout:
            self.conv_fin.append(nn.Dropout(0.5))
        p = 0
        if padding_type == "reflect":
            self.conv_fin.append(nn.ReflectionPad2d(1))
        elif padding_type == "replicate":
            self.conv_fin.append(nn.ReplicationPad2d(1))
        elif padding_type == "zero":
            p = 1
        self.conv_fin.append(nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias))
        self.conv_fin.append(norm_layer(dim))

        # Timestep projection
        self.dense_time = nn.Linear(temb_dim, dim)
        nn.init.zeros_(self.dense_time.bias)

    def forward(self, x: torch.Tensor, time_embed: torch.Tensor, z_embed: torch.Tensor) -> torch.Tensor:
        time_input = self.dense_time(time_embed)
        out = x
        for n, layer in enumerate(self.conv_block):
            out = layer(out) if n > 0 else layer(x)
            if n == 0:
                out = out + time_input[:, :, None, None]
        out = self.adaptive(out, z_embed)
        for layer in self.conv_fin:
            out = layer(out)
        return x + out  # skip connection


# ---------------------------------------------------------------------------
# Conditional conv block for discriminator / energy net
# ---------------------------------------------------------------------------

class _ConvBlockCond(nn.Module):
    """Conv block conditioned on timestep embedding, with optional downsample."""

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        t_emb_dim: int,
        kernel_size: int = 4,
        stride: int = 1,
        padding: int = 1,
        norm_layer=None,
        downsample: bool = True,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        self.use_downsample = downsample
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=use_bias)
        self.use_norm = norm_layer is not None
        if self.use_norm:
            self.norm = norm_layer(out_channel)
        self.act = nn.LeakyReLU(0.2, True)
        self.down = _Downsample(out_channel) if downsample else None
        self.dense = nn.Linear(t_emb_dim, out_channel)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = out + self.dense(t_emb)[..., None, None]
        if self.use_norm:
            out = self.norm(out)
        out = self.act(out)
        if self.use_downsample and self.down is not None:
            out = self.down(out)
        return out


# ---------------------------------------------------------------------------
# UNSBGenerator – time-conditional ResNet generator
# ---------------------------------------------------------------------------

class UNSBGenerator(nn.Module):
    """Time-conditional ResNet generator for UNSB.

    The generator accepts a noisy image, a timestep index, and a random noise
    vector, and produces a refined image at the next timestep.  Intermediate
    encoder features can be extracted for the PatchNCE contrastive loss via the
    ``layers`` / ``encode_only`` arguments.

    Parameters
    ----------
    input_nc : int
        Number of input channels.
    output_nc : int
        Number of output channels.
    ngf : int
        Base number of generator filters.
    n_blocks : int
        Number of conditional ResNet blocks.
    n_mlp : int
        Number of MLP layers in the noise mapping network.
    norm_type : str
        Normalisation type (``"instance"`` or ``"batch"``).
    use_dropout : bool
        Whether to use dropout in ResNet blocks.
    no_antialias : bool
        If ``True``, use strided convs instead of anti-aliased downsampling.
    no_antialias_up : bool
        If ``True``, use transposed convs instead of anti-aliased upsampling.
    init_type : str
        Weight initialisation method.
    init_gain : float
        Gain for weight initialisation.
    """

    def __init__(
        self,
        input_nc: int = 3,
        output_nc: int = 3,
        ngf: int = 64,
        n_blocks: int = 9,
        n_mlp: int = 3,
        norm_type: str = "instance",
        use_dropout: bool = False,
        no_antialias: bool = False,
        no_antialias_up: bool = False,
        init_type: str = "normal",
        init_gain: float = 0.02,
    ) -> None:
        super().__init__()
        assert n_blocks >= 0
        self.ngf = ngf
        norm_layer = _get_norm_layer(norm_type)
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # Encoder
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True),
        ]
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            if no_antialias:
                model += [
                    nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                    norm_layer(ngf * mult * 2),
                    nn.ReLU(True),
                ]
            else:
                model += [
                    nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                    norm_layer(ngf * mult * 2),
                    nn.ReLU(True),
                    _Downsample(ngf * mult * 2),
                ]

        # Conditional ResNet blocks
        mult = 2 ** n_downsampling
        self.model_res = nn.ModuleList()
        for _ in range(n_blocks):
            self.model_res.append(
                _ResnetBlockCond(
                    dim=ngf * mult,
                    norm_layer=norm_layer,
                    use_dropout=use_dropout,
                    use_bias=use_bias,
                    temb_dim=4 * ngf,
                    z_dim=4 * ngf,
                )
            )

        # Decoder
        model_upsample: list = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            if no_antialias_up:
                model_upsample += [
                    nn.ConvTranspose2d(
                        ngf * mult, int(ngf * mult / 2),
                        kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias,
                    ),
                    norm_layer(int(ngf * mult / 2)),
                    nn.ReLU(True),
                ]
            else:
                model_upsample += [
                    _Upsample(ngf * mult),
                    nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=1, bias=use_bias),
                    norm_layer(int(ngf * mult / 2)),
                    nn.ReLU(True),
                ]
        model_upsample += [nn.ReflectionPad2d(3)]
        model_upsample += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model_upsample += [nn.Tanh()]

        self.model = nn.Sequential(*model)
        self.model_upsample = nn.Sequential(*model_upsample)

        # Noise mapping network
        z_dim = ngf * 4
        mapping_layers: list = [_PixelNorm(), nn.Linear(z_dim, z_dim), nn.LeakyReLU(0.2)]
        for _ in range(n_mlp):
            mapping_layers.append(nn.Linear(z_dim, z_dim))
            mapping_layers.append(nn.LeakyReLU(0.2))
        self.z_transform = nn.Sequential(*mapping_layers)

        # Time embedding
        modules_emb: list = [nn.Linear(ngf, z_dim)]
        nn.init.zeros_(modules_emb[-1].bias)
        modules_emb += [nn.LeakyReLU(0.2)]
        modules_emb += [nn.Linear(z_dim, z_dim)]
        nn.init.zeros_(modules_emb[-1].bias)
        modules_emb += [nn.LeakyReLU(0.2)]
        self.time_embed = nn.Sequential(*modules_emb)

        _init_weights(self, init_type, init_gain)

    def forward(
        self,
        x: torch.Tensor,
        time_cond: torch.Tensor,
        z: torch.Tensor,
        layers: Optional[List[int]] = None,
        encode_only: bool = False,
    ) -> torch.Tensor | List[torch.Tensor] | Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass.

        Parameters
        ----------
        x : Tensor (B, C, H, W)
            Input image.
        time_cond : Tensor (B,)
            Integer timestep indices.
        z : Tensor (B, 4*ngf)
            Random noise vector for stochastic modulation.
        layers : list of int, optional
            Layer indices from which to extract intermediate features (for NCE).
        encode_only : bool
            If ``True`` and *layers* is given, return only intermediate features.

        Returns
        -------
        Tensor or list of Tensors
            Generated image, or intermediate features when ``encode_only=True``.
        """
        z_embed = self.z_transform(z)
        temb = _get_timestep_embedding(time_cond, self.ngf)
        time_embed = self.time_embed(temb)

        if layers is not None and len(layers) > 0:
            feat = x
            feats: list = []
            for layer_id, layer in enumerate(self.model):
                feat = layer(feat)
                if layer_id in layers:
                    feats.append(feat)
                if layer_id == layers[-1] and encode_only:
                    return feats
            for layer_id, layer in enumerate(self.model_res):
                feat = layer(feat, time_embed, z_embed)
                if layer_id + len(self.model) in layers:
                    feats.append(feat)
                if layer_id + len(self.model) == layers[-1] and encode_only:
                    return feats
            return feat, feats

        out = self.model(x)
        for layer in self.model_res:
            out = layer(out, time_embed, z_embed)
        out = self.model_upsample(out)
        return out


# ---------------------------------------------------------------------------
# UNSBDiscriminator – time-conditional PatchGAN
# ---------------------------------------------------------------------------

class UNSBDiscriminator(nn.Module):
    """Time-conditional PatchGAN discriminator for UNSB.

    Each convolutional block receives a projected timestep embedding that is
    added to the feature maps, allowing the discriminator to adapt its decision
    boundary per timestep.

    Parameters
    ----------
    input_nc : int
        Number of input channels.
    ndf : int
        Base number of discriminator filters.
    n_layers : int
        Number of convolutional layers.
    norm_type : str
        Normalisation type.
    init_type : str
        Weight initialisation method.
    init_gain : float
        Gain for weight initialisation.
    """

    def __init__(
        self,
        input_nc: int = 3,
        ndf: int = 64,
        n_layers: int = 3,
        norm_type: str = "instance",
        init_type: str = "normal",
        init_gain: float = 0.02,
    ) -> None:
        super().__init__()
        norm_layer = _get_norm_layer(norm_type)
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        t_emb_dim = 4 * ndf

        self.model_main = nn.ModuleList()
        # First layer (no norm)
        self.model_main.append(
            _ConvBlockCond(input_nc, ndf, t_emb_dim, kernel_size=kw, stride=1, padding=padw, use_bias=use_bias)
        )

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            self.model_main.append(
                _ConvBlockCond(
                    ndf * nf_mult_prev, ndf * nf_mult, t_emb_dim,
                    kernel_size=kw, stride=1, padding=padw,
                    use_bias=use_bias, norm_layer=norm_layer,
                )
            )

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        self.model_main.append(
            _ConvBlockCond(
                ndf * nf_mult_prev, ndf * nf_mult, t_emb_dim,
                kernel_size=kw, stride=1, padding=padw,
                use_bias=use_bias, norm_layer=norm_layer, downsample=False,
            )
        )

        self.final_conv = nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)

        self.t_embed = _TimestepEmbedding(
            embedding_dim=t_emb_dim, hidden_dim=t_emb_dim, output_dim=t_emb_dim,
        )

        _init_weights(self, init_type, init_gain)

    def forward(
        self,
        x: torch.Tensor,
        time_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor (B, C, H, W)
            Input image.
        time_idx : Tensor (B,) or scalar
            Integer timestep index.

        Returns
        -------
        Tensor
            Patch-level predictions.
        """
        t_emb = self.t_embed(time_idx)
        out = x
        for layer in self.model_main:
            out = layer(out, t_emb)
        return self.final_conv(out)


# ---------------------------------------------------------------------------
# UNSBEnergyNet – energy network for Schrödinger Bridge loss
# ---------------------------------------------------------------------------

class UNSBEnergyNet(nn.Module):
    """Energy network for the UNSB Schrödinger Bridge loss.

    Accepts concatenated ``(X_t, X_{t+1})`` pairs and optionally a second
    pair for contrastive energy estimation.  Shares the same architecture as
    :class:`UNSBDiscriminator` but with ``4 * input_nc`` input channels
    (two concatenated image pairs).

    Parameters
    ----------
    input_nc : int
        Number of channels per single image (the network expects 4× this).
    ndf : int
        Base number of filters.
    n_layers : int
        Number of convolutional layers.
    norm_type : str
        Normalisation type.
    init_type : str
        Weight initialisation method.
    init_gain : float
        Gain for weight initialisation.
    """

    def __init__(
        self,
        input_nc: int = 3,
        ndf: int = 64,
        n_layers: int = 3,
        norm_type: str = "instance",
        init_type: str = "normal",
        init_gain: float = 0.02,
    ) -> None:
        super().__init__()
        # Energy net receives concatenated (Xt, Xt+1) pairs = 4 * input_nc channels
        # when using the second pair via input2, they're handled inside forward
        self.disc = UNSBDiscriminator(
            input_nc=input_nc * 4,
            ndf=ndf,
            n_layers=n_layers,
            norm_type=norm_type,
            init_type=init_type,
            init_gain=init_gain,
        )

    def forward(
        self,
        xt_xt1: torch.Tensor,
        time_idx: torch.Tensor,
        xt_xt1_ref: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute energy scores.

        Parameters
        ----------
        xt_xt1 : Tensor (B, 2C, H, W)
            Concatenated ``(X_t, X_{t+1})`` pair.
        time_idx : Tensor (B,) or scalar
            Integer timestep index.
        xt_xt1_ref : Tensor (B, 2C, H, W), optional
            Reference pair for contrastive estimation.  When provided, the
            network outputs pairwise energy scores between *xt_xt1* and
            *xt_xt1_ref*.

        Returns
        -------
        Tensor
            Energy scores.
        """
        if xt_xt1_ref is not None:
            inp = torch.cat([xt_xt1, xt_xt1_ref], dim=1)
        else:
            inp = torch.cat([xt_xt1, xt_xt1], dim=1)
        return self.disc(inp, time_idx)


# ---------------------------------------------------------------------------
# PatchSampleMLP (feature projection for contrastive loss)
# ---------------------------------------------------------------------------

class _L2Normalize(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).sum(1, keepdim=True).pow(0.5)
        return x.div(norm + 1e-7)


class PatchSampleMLP(nn.Module):
    """MLP projection head that samples spatial patches from encoder features.

    At first call the MLP layers are lazily created based on the input
    feature dimensions (data-dependent initialisation following the CUT/UNSB
    implementation).

    Parameters
    ----------
    use_mlp : bool
        If ``True``, project features through a 2-layer MLP.
    nc : int
        Output dimension of the MLP projection.
    init_type : str
        Weight initialisation method.
    init_gain : float
        Gain for weight initialisation.
    """

    def __init__(
        self,
        use_mlp: bool = True,
        nc: int = 256,
        init_type: str = "normal",
        init_gain: float = 0.02,
    ) -> None:
        super().__init__()
        self.l2norm = _L2Normalize()
        self.use_mlp = use_mlp
        self.nc = nc
        self.mlp_init = False
        self.init_type = init_type
        self.init_gain = init_gain

    def create_mlp(self, feats: List[torch.Tensor]) -> None:
        """Lazily create MLP layers matching the feature channel dims."""
        for mlp_id, feat in enumerate(feats):
            input_nc = feat.shape[1]
            mlp = nn.Sequential(nn.Linear(input_nc, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc))
            mlp = mlp.to(feat.device)
            setattr(self, f"mlp_{mlp_id}", mlp)
        _init_weights(self, self.init_type, self.init_gain)
        self.mlp_init = True

    def forward(
        self,
        feats: List[torch.Tensor],
        num_patches: int = 256,
        patch_ids: Optional[List] = None,
    ) -> Tuple[List[torch.Tensor], List]:
        return_ids: list = []
        return_feats: list = []
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)
        for feat_id, feat in enumerate(feats):
            B, H, W = feat.shape[0], feat.shape[2], feat.shape[3]
            feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)
            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[feat_id]
                else:
                    patch_id = np.random.permutation(feat_reshape.shape[1])
                    patch_id = patch_id[: int(min(num_patches, patch_id.shape[0]))]
                patch_id = torch.as_tensor(patch_id, dtype=torch.long, device=feat.device)
                x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)
            else:
                x_sample = feat_reshape
                patch_id = []
            if self.use_mlp:
                mlp = getattr(self, f"mlp_{feat_id}")
                x_sample = mlp(x_sample)
            return_ids.append(patch_id)
            x_sample = self.l2norm(x_sample)
            if num_patches == 0:
                x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])
            return_feats.append(x_sample)
        return return_feats, return_ids


# ---------------------------------------------------------------------------
# GAN Loss
# ---------------------------------------------------------------------------

class GANLoss(nn.Module):
    """GAN objective supporting LSGAN, Vanilla BCE, and WGAN-GP modes.

    Parameters
    ----------
    gan_mode : str
        ``"lsgan"`` | ``"vanilla"`` | ``"wgangp"``
    target_real_label : float
        Label for real images.
    target_fake_label : float
        Label for fake images.
    """

    def __init__(self, gan_mode: str = "lsgan", target_real_label: float = 1.0, target_fake_label: float = 0.0) -> None:
        super().__init__()
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        elif gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ("wgangp", "nonsaturating"):
            self.loss = None
        else:
            raise NotImplementedError(f"GAN mode [{gan_mode}] not implemented")

    def _get_target_tensor(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        target = self.real_label if target_is_real else self.fake_label
        return target.expand_as(prediction)

    def forward(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        if self.gan_mode in ("lsgan", "vanilla"):
            target_tensor = self._get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == "wgangp":
            loss = -prediction.mean() if target_is_real else prediction.mean()
        elif self.gan_mode == "nonsaturating":
            bs = prediction.size(0)
            if target_is_real:
                loss = F.softplus(-prediction).view(bs, -1).mean(dim=1)
            else:
                loss = F.softplus(prediction).view(bs, -1).mean(dim=1)
        return loss


# ---------------------------------------------------------------------------
# PatchNCE Loss
# ---------------------------------------------------------------------------

class PatchNCELoss(nn.Module):
    """Contrastive loss on sampled feature patches (InfoNCE formulation).

    Parameters
    ----------
    nce_T : float
        Temperature for the contrastive loss softmax.
    batch_size : int
        Batch size used for reshaping negatives.
    nce_includes_all_negatives_from_minibatch : bool
        If ``True``, include negatives from the entire mini-batch.
    """

    def __init__(
        self,
        nce_T: float = 0.07,
        batch_size: int = 1,
        nce_includes_all_negatives_from_minibatch: bool = False,
    ) -> None:
        super().__init__()
        self.nce_T = nce_T
        self.batch_size = batch_size
        self.nce_includes_all_negatives_from_minibatch = nce_includes_all_negatives_from_minibatch
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")

    def forward(self, feat_q: torch.Tensor, feat_k: torch.Tensor) -> torch.Tensor:
        num_patches = feat_q.shape[0]
        feat_k = feat_k.detach()

        # Positive logit
        l_pos = torch.bmm(feat_q.view(num_patches, 1, -1), feat_k.view(num_patches, -1, 1))
        l_pos = l_pos.view(num_patches, 1)

        # Negative logits
        if self.nce_includes_all_negatives_from_minibatch:
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.batch_size

        dim = feat_q.shape[1]
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        diagonal = torch.eye(npatches, device=feat_q.device, dtype=torch.bool)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.nce_T
        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long, device=feat_q.device))
        return loss


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def create_generator(
    input_nc: int = 3,
    output_nc: int = 3,
    ngf: int = 64,
    n_blocks: int = 9,
    n_mlp: int = 3,
    norm_type: str = "instance",
    use_dropout: bool = False,
    no_antialias: bool = False,
    no_antialias_up: bool = False,
    init_type: str = "normal",
    init_gain: float = 0.02,
    **kwargs,
) -> UNSBGenerator:
    """Factory for the UNSB generator."""
    return UNSBGenerator(
        input_nc=input_nc,
        output_nc=output_nc,
        ngf=ngf,
        n_blocks=n_blocks,
        n_mlp=n_mlp,
        norm_type=norm_type,
        use_dropout=use_dropout,
        no_antialias=no_antialias,
        no_antialias_up=no_antialias_up,
        init_type=init_type,
        init_gain=init_gain,
    )


def create_discriminator(
    input_nc: int = 3,
    ndf: int = 64,
    n_layers: int = 3,
    norm_type: str = "instance",
    init_type: str = "normal",
    init_gain: float = 0.02,
    **kwargs,
) -> UNSBDiscriminator:
    """Factory for the UNSB discriminator."""
    return UNSBDiscriminator(
        input_nc=input_nc,
        ndf=ndf,
        n_layers=n_layers,
        norm_type=norm_type,
        init_type=init_type,
        init_gain=init_gain,
    )


def create_energy_net(
    input_nc: int = 3,
    ndf: int = 64,
    n_layers: int = 3,
    norm_type: str = "instance",
    init_type: str = "normal",
    init_gain: float = 0.02,
    **kwargs,
) -> UNSBEnergyNet:
    """Factory for the UNSB energy network."""
    return UNSBEnergyNet(
        input_nc=input_nc,
        ndf=ndf,
        n_layers=n_layers,
        norm_type=norm_type,
        init_type=init_type,
        init_gain=init_gain,
    )


def create_patch_sample_mlp(
    use_mlp: bool = True,
    nc: int = 256,
    init_type: str = "normal",
    init_gain: float = 0.02,
    **kwargs,
) -> PatchSampleMLP:
    """Factory for the PatchSampleMLP feature projection network."""
    return PatchSampleMLP(
        use_mlp=use_mlp,
        nc=nc,
        init_type=init_type,
        init_gain=init_gain,
    )
