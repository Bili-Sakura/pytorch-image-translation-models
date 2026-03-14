# Copyright (c) 2026 EarthBridge Team.
# SelfRDB NCSN++ with self_recursion (Self-Consistent Recursive Diffusion Bridge).
# Imports shared backbones from SynDiff; no backbones subfolder.

"""SelfRDB model: NCSNpp with self_recursion. Single model.py, no backbones subfolder."""

from __future__ import annotations

import functools

import numpy as np
import torch
import torch.nn as nn

from ..syndiff.model import (
    AttnBlockpp,
    Combine,
    Downsample,
    GaussianFourierProjection,
    ResnetBlockBigGANpp_Adagn,
    ResnetBlockBigGANpp_Adagn_one,
    ResnetBlockDDPMpp_Adagn,
    Upsample,
    conv1x1,
    conv3x3,
    default_init,
    dense,
    get_timestep_embedding,
)


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input**2, dim=1, keepdim=True) + 1e-8)


class NCSNpp(nn.Module):
    """NCSN++ model with self_recursion for SelfRDB."""

    def __init__(
        self,
        *,
        self_recursion: bool = True,
        z_emb_dim: int = 256,
        ch_mult=(1, 1, 2, 2, 4, 4),
        num_res_blocks: int = 2,
        attn_resolutions=(16,),
        dropout: float = 0.0,
        resamp_with_conv: bool = True,
        image_size: int = 256,
        conditional: bool = True,
        fir: bool = True,
        fir_kernel=(1, 3, 3, 1),
        skip_rescale: bool = True,
        resblock_type: str = "biggan",
        progressive: str = "none",
        progressive_input: str = "residual",
        embedding_type: str = "positional",
        combine_method: str = "sum",
        fourier_scale: float = 16.0,
        nf: int = 64,
        num_channels: int = 2,
        nz: int = 100,
        n_mlp: int = 3,
        centered: bool = True,
        not_use_tanh: bool = False,
    ):
        super().__init__()
        self.self_recursion = self_recursion
        self.not_use_tanh = not_use_tanh
        self.centered = centered
        self.act = act = nn.SiLU()

        self.nf = nf
        ch_mult = tuple(ch_mult)
        self.num_res_blocks = num_res_blocks
        attn_resolutions = list(attn_resolutions) if attn_resolutions else []
        self.attn_resolutions = attn_resolutions
        self.num_resolutions = num_resolutions = len(ch_mult)
        self.all_resolutions = [image_size // (2**i) for i in range(num_resolutions)]

        self.conditional = conditional
        fir_kernel = tuple(fir_kernel) if isinstance(fir_kernel, (list, tuple)) else fir_kernel
        self.skip_rescale = skip_rescale
        self.resblock_type = resblock_type = resblock_type.lower()
        self.progressive = progressive = progressive.lower()
        self.progressive_input = progressive_input = progressive_input.lower()
        self.embedding_type = embedding_type = embedding_type.lower()
        init_scale = 0.0
        assert progressive in ["none", "output_skip", "residual"]
        assert progressive_input in ["none", "input_skip", "residual"]
        assert embedding_type in ["fourier", "positional"]
        combine_method = combine_method.lower()
        combiner = functools.partial(Combine, method=combine_method)

        modules = []
        if embedding_type == "fourier":
            modules.append(GaussianFourierProjection(embedding_size=nf, scale=fourier_scale))
            embed_dim = 2 * nf
        else:
            embed_dim = nf

        if conditional:
            modules.append(nn.Linear(embed_dim, nf * 4))
            modules[-1].weight.data = default_init()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)
            modules.append(nn.Linear(nf * 4, nf * 4))
            modules[-1].weight.data = default_init()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)

        AttnBlock = functools.partial(AttnBlockpp, init_scale=init_scale, skip_rescale=skip_rescale)
        Upsample = functools.partial(Upsample, with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)
        Downsample = functools.partial(Downsample, with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

        if progressive == "output_skip":
            self.pyramid_upsample = Upsample(with_conv=False)
        elif progressive == "residual":
            pyramid_upsample = functools.partial(Upsample, with_conv=True)

        if progressive_input == "input_skip":
            self.pyramid_downsample = Downsample(with_conv=False)
        elif progressive_input == "residual":
            pyramid_downsample = functools.partial(Downsample, with_conv=True)

        if resblock_type == "ddpm":
            ResnetBlock = functools.partial(
                ResnetBlockDDPMpp_Adagn,
                act=act,
                dropout=dropout,
                init_scale=init_scale,
                skip_rescale=skip_rescale,
                temb_dim=nf * 4,
                zemb_dim=z_emb_dim,
            )
        elif resblock_type == "biggan":
            ResnetBlock = functools.partial(
                ResnetBlockBigGANpp_Adagn,
                act=act,
                dropout=dropout,
                fir=fir,
                fir_kernel=fir_kernel,
                init_scale=init_scale,
                skip_rescale=skip_rescale,
                temb_dim=nf * 4,
                zemb_dim=z_emb_dim,
            )
        elif resblock_type == "biggan_oneadagn":
            ResnetBlock = functools.partial(
                ResnetBlockBigGANpp_Adagn_one,
                act=act,
                dropout=dropout,
                fir=fir,
                fir_kernel=fir_kernel,
                init_scale=init_scale,
                skip_rescale=skip_rescale,
                temb_dim=nf * 4,
                zemb_dim=z_emb_dim,
            )
        else:
            raise ValueError(f"resblock type {resblock_type} unrecognized.")

        channels = num_channels
        if progressive_input != "none":
            input_pyramid_ch = channels

        modules.append(conv3x3(channels, nf))
        hs_c = [nf]
        in_ch = nf

        for i_level in range(num_resolutions):
            for _ in range(num_res_blocks):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch

            if self.all_resolutions[i_level] in attn_resolutions:
                modules.append(AttnBlock(channels=in_ch))
                hs_c.append(in_ch)

            if i_level != num_resolutions - 1:
                if resblock_type == "ddpm":
                    modules.append(Downsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(down=True, in_ch=in_ch))

            if progressive_input == "input_skip":
                modules.append(combiner(dim1=input_pyramid_ch, dim2=in_ch))
                if combine_method == "cat":
                    in_ch *= 2
            elif progressive_input == "residual":
                modules.append(pyramid_downsample(in_ch=input_pyramid_ch, out_ch=in_ch))
                input_pyramid_ch = in_ch

            hs_c.append(in_ch)

        in_ch = hs_c[-1]
        modules.append(ResnetBlock(in_ch=in_ch))
        modules.append(AttnBlock(channels=in_ch))
        modules.append(ResnetBlock(in_ch=in_ch))

        pyramid_ch = 0
        for i_level in reversed(range(num_resolutions)):
            for _ in range(num_res_blocks + 1):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
                in_ch = out_ch

            if self.all_resolutions[i_level] in attn_resolutions:
                modules.append(AttnBlock(channels=in_ch))

            if progressive != "none":
                if i_level == num_resolutions - 1:
                    if progressive == "output_skip":
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6))
                        modules.append(conv3x3(in_ch, channels, init_scale=init_scale))
                        pyramid_ch = channels
                    elif progressive == "residual":
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6))
                        modules.append(conv3x3(in_ch, in_ch, bias=True))
                        pyramid_ch = in_ch
                else:
                    if progressive == "output_skip":
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6))
                        modules.append(conv3x3(in_ch, channels, bias=True, init_scale=init_scale))
                        pyramid_ch = channels
                    elif progressive == "residual":
                        modules.append(pyramid_upsample(in_ch=pyramid_ch, out_ch=in_ch))
                        pyramid_ch = in_ch
            else:
                if i_level != 0:
                    if resblock_type == "ddpm":
                        modules.append(Upsample(in_ch=in_ch))
                    else:
                        modules.append(ResnetBlock(in_ch=in_ch, up=True))

        assert not hs_c

        if progressive != "output_skip":
            modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6))
            modules.append(conv3x3(in_ch, channels, init_scale=init_scale))

        self.all_modules = nn.ModuleList(modules)
        mapping_layers = [PixelNorm(), dense(nz, z_emb_dim), self.act]
        for _ in range(n_mlp):
            mapping_layers.append(dense(z_emb_dim, z_emb_dim))
            mapping_layers.append(self.act)
        self.z_transform = nn.Sequential(*mapping_layers)
        self.nz = nz

    def forward(self, x, time_cond, x_r=None):
        if self.self_recursion:
            if x_r is None:
                x_r = torch.zeros_like(x, device=x.device, dtype=x.dtype)
            x = torch.cat([x_r, x], dim=1)

        b = x.shape[0]
        z = torch.randn(b, self.nz, device=x.device, dtype=x.dtype)
        zemb = self.z_transform(z)
        modules = self.all_modules
        m_idx = 0

        if self.embedding_type == "fourier":
            temb = modules[m_idx](torch.log(time_cond))
            m_idx += 1
        else:
            temb = get_timestep_embedding(time_cond, self.nf)

        if self.conditional:
            temb = modules[m_idx](temb)
            m_idx += 1
            temb = modules[m_idx](self.act(temb))
            m_idx += 1
        else:
            temb = None

        if not self.centered:
            x = 2 * x - 1.0

        input_pyramid = x if self.progressive_input != "none" else None
        hs = [modules[m_idx](x)]
        m_idx += 1

        for i_level in range(self.num_resolutions):
            for _ in range(self.num_res_blocks):
                h = modules[m_idx](hs[-1], temb, zemb)
                m_idx += 1
                if h.shape[-1] in self.attn_resolutions:
                    h = modules[m_idx](h)
                    m_idx += 1
                hs.append(h)

            if i_level != self.num_resolutions - 1:
                if self.resblock_type == "ddpm":
                    h = modules[m_idx](hs[-1])
                    m_idx += 1
                else:
                    h = modules[m_idx](hs[-1], temb, zemb)
                    m_idx += 1

            if self.progressive_input == "input_skip":
                input_pyramid = self.pyramid_downsample(input_pyramid)
                h = modules[m_idx](input_pyramid, h)
                m_idx += 1
            elif self.progressive_input == "residual":
                input_pyramid = modules[m_idx](input_pyramid)
                m_idx += 1
                input_pyramid = (input_pyramid + h) / np.sqrt(2.0) if self.skip_rescale else input_pyramid + h
                h = input_pyramid
            hs.append(h)

        h = hs[-1]
        h = modules[m_idx](h, temb, zemb)
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1
        h = modules[m_idx](h, temb, zemb)
        m_idx += 1

        pyramid = None
        for i_level in reversed(range(self.num_resolutions)):
            for _ in range(self.num_res_blocks + 1):
                h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb, zemb)
                m_idx += 1
            if h.shape[-1] in self.attn_resolutions:
                h = modules[m_idx](h)
                m_idx += 1
            if self.progressive != "none":
                if i_level == self.num_resolutions - 1:
                    pyramid = self.act(modules[m_idx](h))
                    m_idx += 1
                    pyramid = modules[m_idx](pyramid)
                    m_idx += 1
                else:
                    if self.progressive == "output_skip":
                        pyramid = self.pyramid_upsample(pyramid)
                    else:
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                    pyramid_h = self.act(modules[m_idx](h))
                    m_idx += 1
                    pyramid_h = modules[m_idx](pyramid_h)
                    m_idx += 1
                    pyramid = pyramid + pyramid_h
            else:
                if i_level != 0:
                    if self.resblock_type == "ddpm":
                        h = modules[m_idx](h)
                        m_idx += 1
                    else:
                        h = modules[m_idx](h, temb, zemb)
                        m_idx += 1

        assert not hs
        if self.progressive == "output_skip":
            h = pyramid
        else:
            h = self.act(modules[m_idx](h))
            m_idx += 1
            h = modules[m_idx](h)
            m_idx += 1

        if not self.not_use_tanh:
            h = torch.tanh(h)
        return h[:, [0], ...]
