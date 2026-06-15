# Copyright (c) 2026 EarthBridge Team.
# Credits: img2img-turbo from Parmar et al. 2024 —
# https://github.com/GaParmar/img2img-turbo

"""Shared SD-Turbo utilities for CycleGAN-Turbo and pix2pix-turbo."""

from __future__ import annotations

import os
from typing import Union

import requests
import torch
from diffusers import DDPMScheduler
from tqdm import tqdm


def make_1step_sched(device: Union[str, torch.device] = "cpu") -> DDPMScheduler:
    """Build a single-step DDPM scheduler from SD-Turbo weights."""
    device = torch.device(device)
    scheduler = DDPMScheduler.from_pretrained("stabilityai/sd-turbo", subfolder="scheduler")
    scheduler.set_timesteps(1, device=device)
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)
    return scheduler


def my_vae_encoder_fwd(self, sample: torch.Tensor) -> torch.Tensor:
    """VAE encoder forward that caches down-block activations for skip connections."""
    sample = self.conv_in(sample)
    blocks = []
    for down_block in self.down_blocks:
        blocks.append(sample)
        sample = down_block(sample)
    sample = self.mid_block(sample)
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    self.current_down_blocks = blocks
    return sample


def my_vae_decoder_fwd(self, sample: torch.Tensor, latent_embeds=None) -> torch.Tensor:
    """VAE decoder forward with optional skip connections from the encoder."""
    sample = self.conv_in(sample)
    upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
    sample = self.mid_block(sample, latent_embeds)
    sample = sample.to(upscale_dtype)
    if not self.ignore_skip:
        skip_convs = [self.skip_conv_1, self.skip_conv_2, self.skip_conv_3, self.skip_conv_4]
        for idx, up_block in enumerate(self.up_blocks):
            skip_in = skip_convs[idx](self.incoming_skip_acts[::-1][idx] * self.gamma)
            sample = sample + skip_in
            sample = up_block(sample, latent_embeds)
    else:
        for up_block in self.up_blocks:
            sample = up_block(sample, latent_embeds)
    if latent_embeds is None:
        sample = self.conv_norm_out(sample)
    else:
        sample = self.conv_norm_out(sample, latent_embeds)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    return sample


def download_url(url: str, outf: str) -> None:
    """Download a checkpoint URL to a local path if it does not already exist."""
    if os.path.exists(outf):
        return
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
    with open(outf, "wb") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()


__all__ = [
    "make_1step_sched",
    "my_vae_encoder_fwd",
    "my_vae_decoder_fwd",
    "download_url",
]
