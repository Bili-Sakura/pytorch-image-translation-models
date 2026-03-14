# Copyright (c) 2025 Copyright holder of the paper "Towards General Modality Translation with Contrastive and Predictive Latent Diffusion Bridge" submitted to NeurIPS 2025 for review.
# All rights reserved.

"""Stripped from https://github.com/abraham-ai/stable-diffusion/blob/main/ldm/models/autoencoder.py"""

import torch

from src.models.lddbm.names import Encoders
from src.models.lddbm.vae_modules import (
    Decoder,
    DiagonalGaussianDistribution,
    Downsample,
    Encoder,
    Upsample,
)


class AutoencoderKLInnerExtdEncoder(torch.nn.Module):
    def __init__(
        self,
        ddconfig,
        embed_dim,
        ckpt_path=None,
        ignore_keys=[],
        image_key="image",
        colorize_nlabels=None,
        monitor=None,
        freeze=False,
        model_type="kl_vae_sr128",
    ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        if freeze:
            for param in self.parameters():
                param.requires_grad = False

        self.model_type = model_type

        if model_type == Encoders.KlVaePreTrainedEncoder16.value:
            ch_dim = 32
            self.conv_in = torch.nn.Conv2d(
                3, ch_dim, kernel_size=3, stride=1, padding=1
            )
            self.l1_in = Upsample(in_channels=ch_dim, with_conv=True)
            self.l2_in = Upsample(in_channels=ch_dim, with_conv=True)
            self.l3_in = Upsample(in_channels=ch_dim, with_conv=True)
            self.l4_in = Upsample(in_channels=ch_dim, with_conv=True)
            self.l5_in = torch.nn.Conv2d(ch_dim, 3, kernel_size=3, stride=1, padding=1)

            self.bridging_layer_in = torch.nn.Sequential(
                self.conv_in, self.l1_in, self.l2_in, self.l3_in, self.l4_in, self.l5_in
            )

        elif model_type == Encoders.KlVaePreTrainedEncoder128.value:
            ch_dim = 32
            self.conv_in = torch.nn.Conv2d(
                3, ch_dim, kernel_size=3, stride=1, padding=1
            )
            self.l1_in = Upsample(in_channels=ch_dim, with_conv=True)
            self.l2_in = torch.nn.Conv2d(ch_dim, 3, kernel_size=3, stride=1, padding=1)

            self.bridging_layer_in = torch.nn.Sequential(
                self.conv_in, self.l1_in, self.l2_in
            )
        else:
            raise NotImplementedError("No Such model type")

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        x = self.bridging_layer_in(x)
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior.mean

    def forward(self, x):
        return self.encode(x)

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x


class AutoencoderKLInnerExtdDecoder(torch.nn.Module):
    def __init__(
        self,
        ddconfig,
        embed_dim,
        ckpt_path=None,
        ignore_keys=[],
        image_key="image",
        colorize_nlabels=None,
        monitor=None,
        freeze=False,
    ):
        super().__init__()
        self.image_key = image_key
        self.decoder = Decoder(**ddconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        if freeze:
            for param in self.parameters():
                param.requires_grad = False

        ch_dim = 32
        self.conv_out = torch.nn.Conv2d(3, ch_dim, kernel_size=3, stride=1, padding=1)
        self.l1_out = Downsample(in_channels=ch_dim, with_conv=True)
        self.l2_out = torch.nn.Conv2d(ch_dim, 3, kernel_size=3, stride=1, padding=1)
        self.bridging_layer_out = torch.nn.Sequential(
            self.conv_out, self.l1_out, self.l2_out
        )

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        dec = self.bridging_layer_out(dec)
        return dec

    def forward(self, z):
        return self.decode(z)

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x
