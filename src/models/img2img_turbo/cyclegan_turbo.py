# Copyright (c) 2026 EarthBridge Team.
# Credits: img2img-turbo from Parmar et al. 2024 —
# https://github.com/GaParmar/img2img-turbo

"""CycleGAN-Turbo model (one-step unpaired SD-Turbo translation)."""

from __future__ import annotations

import copy
import os
from typing import Optional, Union

import torch
import torch.nn as nn
from diffusers import AutoencoderKL, UNet2DConditionModel
from peft import LoraConfig
from transformers import AutoTokenizer, CLIPTextModel

from src.models.img2img_turbo.utils import (
    download_url,
    make_1step_sched,
    my_vae_decoder_fwd,
    my_vae_encoder_fwd,
)

PRETRAINED_CYCLEGAN_TURBO: dict[str, dict[str, str]] = {
    "day_to_night": {
        "url": "https://www.cs.cmu.edu/~img2img-turbo/models/day2night.pkl",
        "caption": "driving in the night",
        "direction": "a2b",
    },
    "night_to_day": {
        "url": "https://www.cs.cmu.edu/~img2img-turbo/models/night2day.pkl",
        "caption": "driving in the day",
        "direction": "b2a",
    },
    "clear_to_rainy": {
        "url": "https://www.cs.cmu.edu/~img2img-turbo/models/clear2rainy.pkl",
        "caption": "driving in heavy rain",
        "direction": "a2b",
    },
    "rainy_to_clear": {
        "url": "https://www.cs.cmu.edu/~img2img-turbo/models/rainy2clear.pkl",
        "caption": "driving in the day",
        "direction": "b2a",
    },
}


class VAEEncode(nn.Module):
    """Encode images to latents, selecting domain-specific VAE weights."""

    def __init__(self, vae: AutoencoderKL, vae_b2a: Optional[AutoencoderKL] = None) -> None:
        super().__init__()
        self.vae = vae
        self.vae_b2a = vae_b2a

    def forward(self, x: torch.Tensor, direction: str) -> torch.Tensor:
        assert direction in ("a2b", "b2a")
        vae = self.vae if direction == "a2b" else self.vae_b2a
        return vae.encode(x).latent_dist.sample() * vae.config.scaling_factor


class VAEDecode(nn.Module):
    """Decode latents to images, selecting domain-specific VAE weights."""

    def __init__(self, vae: AutoencoderKL, vae_b2a: Optional[AutoencoderKL] = None) -> None:
        super().__init__()
        self.vae = vae
        self.vae_b2a = vae_b2a

    def forward(self, x: torch.Tensor, direction: str) -> torch.Tensor:
        assert direction in ("a2b", "b2a")
        vae = self.vae if direction == "a2b" else self.vae_b2a
        assert vae.encoder.current_down_blocks is not None
        vae.decoder.incoming_skip_acts = vae.encoder.current_down_blocks
        return vae.decode(x / vae.config.scaling_factor).sample.clamp(-1, 1)


def _lora_target_modules(unet: UNet2DConditionModel) -> tuple[list[str], list[str], list[str]]:
    patterns = [
        "to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2",
        "conv_in", "conv_shortcut", "conv_out", "proj_out", "proj_in",
        "ff.net.2", "ff.net.0.proj",
    ]
    encoder, decoder, others = [], [], []
    for name, _ in unet.named_parameters():
        if "bias" in name or "norm" in name:
            continue
        key = name.replace(".weight", "")
        for pattern in patterns:
            if pattern in name and ("down_blocks" in name or "conv_in" in name):
                encoder.append(key)
                break
            if pattern in name and "up_blocks" in name:
                decoder.append(key)
                break
            if pattern in name:
                others.append(key)
                break
    return encoder, decoder, others


def initialize_unet(
    rank: int,
    *,
    return_lora_module_names: bool = False,
    base_model: str = "stabilityai/sd-turbo",
) -> Union[UNet2DConditionModel, tuple[UNet2DConditionModel, list[str], list[str], list[str]]]:
    """Create a LoRA-adapted UNet for CycleGAN-Turbo training."""
    unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet")
    unet.requires_grad_(False)
    unet.train()
    encoder, decoder, others = _lora_target_modules(unet)
    unet.add_adapter(
        LoraConfig(r=rank, init_lora_weights="gaussian", target_modules=encoder, lora_alpha=rank),
        adapter_name="default_encoder",
    )
    unet.add_adapter(
        LoraConfig(r=rank, init_lora_weights="gaussian", target_modules=decoder, lora_alpha=rank),
        adapter_name="default_decoder",
    )
    unet.add_adapter(
        LoraConfig(r=rank, init_lora_weights="gaussian", target_modules=others, lora_alpha=rank),
        adapter_name="default_others",
    )
    unet.set_adapters(["default_encoder", "default_decoder", "default_others"])
    if return_lora_module_names:
        return unet, encoder, decoder, others
    return unet


def initialize_vae(
    rank: int = 4,
    *,
    device: Union[str, torch.device] = "cpu",
    return_lora_module_names: bool = False,
    base_model: str = "stabilityai/sd-turbo",
) -> Union[AutoencoderKL, tuple[AutoencoderKL, list[str]]]:
    """Create a LoRA-adapted VAE with skip-connection convolutions."""
    device = torch.device(device)
    vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae")
    vae.requires_grad_(False)
    vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
    vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
    vae.requires_grad_(True)
    vae.train()
    vae.decoder.skip_conv_1 = nn.Conv2d(512, 512, kernel_size=1, stride=1, bias=False).to(device)
    vae.decoder.skip_conv_2 = nn.Conv2d(256, 512, kernel_size=1, stride=1, bias=False).to(device)
    vae.decoder.skip_conv_3 = nn.Conv2d(128, 512, kernel_size=1, stride=1, bias=False).to(device)
    vae.decoder.skip_conv_4 = nn.Conv2d(128, 256, kernel_size=1, stride=1, bias=False).to(device)
    for conv in (vae.decoder.skip_conv_1, vae.decoder.skip_conv_2, vae.decoder.skip_conv_3, vae.decoder.skip_conv_4):
        nn.init.constant_(conv.weight, 1e-5)
    vae.decoder.ignore_skip = False
    vae.decoder.gamma = 1
    target_modules = [
        "conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
        "skip_conv_1", "skip_conv_2", "skip_conv_3", "skip_conv_4",
        "to_k", "to_q", "to_v", "to_out.0",
    ]
    vae.add_adapter(
        LoraConfig(r=rank, init_lora_weights="gaussian", target_modules=target_modules),
        adapter_name="vae_skip",
    )
    if return_lora_module_names:
        return vae, target_modules
    return vae


class CycleGANTurbo(nn.Module):
    """One-step unpaired image translation built on SD-Turbo."""

    def __init__(
        self,
        *,
        pretrained_name: Optional[str] = None,
        pretrained_path: Optional[str] = None,
        ckpt_folder: str = "checkpoints",
        device: Union[str, torch.device] = "cuda",
        base_model: str = "stabilityai/sd-turbo",
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder")
        self.text_encoder.to(self.device)
        self.sched = make_1step_sched(self.device)

        vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet")
        vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
        vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
        vae.decoder.skip_conv_1 = nn.Conv2d(512, 512, kernel_size=1, stride=1, bias=False)
        vae.decoder.skip_conv_2 = nn.Conv2d(256, 512, kernel_size=1, stride=1, bias=False)
        vae.decoder.skip_conv_3 = nn.Conv2d(128, 512, kernel_size=1, stride=1, bias=False)
        vae.decoder.skip_conv_4 = nn.Conv2d(128, 256, kernel_size=1, stride=1, bias=False)
        vae.decoder.ignore_skip = False
        self.unet, self.vae = unet, vae

        self.timesteps = torch.tensor([999], device=self.device, dtype=torch.long)
        self.caption: Optional[str] = None
        self.direction: Optional[str] = None

        if pretrained_name is not None:
            if pretrained_name not in PRETRAINED_CYCLEGAN_TURBO:
                raise ValueError(
                    f"Unknown pretrained_name {pretrained_name!r}. "
                    f"Choose from {list(PRETRAINED_CYCLEGAN_TURBO)}"
                )
            meta = PRETRAINED_CYCLEGAN_TURBO[pretrained_name]
            self.load_ckpt_from_url(meta["url"], ckpt_folder)
            self.caption = meta["caption"]
            self.direction = meta["direction"]
        elif pretrained_path is not None:
            sd = torch.load(pretrained_path, map_location="cpu")
            self.load_ckpt_from_state_dict(sd)
            self.caption = None
            self.direction = None

        self.vae_enc.to(self.device)
        self.vae_dec.to(self.device)
        self.unet.to(self.device)
        self.vae.to(self.device)

    def load_ckpt_from_state_dict(self, sd: dict) -> None:
        """Load LoRA and VAE encoder/decoder weights from a checkpoint dict."""
        lora_conf_encoder = LoraConfig(
            r=sd["rank_unet"], init_lora_weights="gaussian",
            target_modules=sd["l_target_modules_encoder"], lora_alpha=sd["rank_unet"],
        )
        lora_conf_decoder = LoraConfig(
            r=sd["rank_unet"], init_lora_weights="gaussian",
            target_modules=sd["l_target_modules_decoder"], lora_alpha=sd["rank_unet"],
        )
        lora_conf_others = LoraConfig(
            r=sd["rank_unet"], init_lora_weights="gaussian",
            target_modules=sd["l_modules_others"], lora_alpha=sd["rank_unet"],
        )
        self.unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
        self.unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
        self.unet.add_adapter(lora_conf_others, adapter_name="default_others")
        for name, param in self.unet.named_parameters():
            sd_name = name.replace(".default_encoder.weight", ".weight")
            if "lora" in name and "default_encoder" in name:
                param.data.copy_(sd["sd_encoder"][sd_name])
        for name, param in self.unet.named_parameters():
            sd_name = name.replace(".default_decoder.weight", ".weight")
            if "lora" in name and "default_decoder" in name:
                param.data.copy_(sd["sd_decoder"][sd_name])
        for name, param in self.unet.named_parameters():
            sd_name = name.replace(".default_others.weight", ".weight")
            if "lora" in name and "default_others" in name:
                param.data.copy_(sd["sd_other"][sd_name])
        self.unet.set_adapter(["default_encoder", "default_decoder", "default_others"])

        vae_lora_config = LoraConfig(
            r=sd["rank_vae"], init_lora_weights="gaussian",
            target_modules=sd["vae_lora_target_modules"],
        )
        self.vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
        self.vae.decoder.gamma = 1
        self.vae_b2a = copy.deepcopy(self.vae)
        self.vae_enc = VAEEncode(self.vae, vae_b2a=self.vae_b2a)
        self.vae_dec = VAEDecode(self.vae, vae_b2a=self.vae_b2a)
        self.vae_enc.load_state_dict(sd["sd_vae_enc"])
        self.vae_dec.load_state_dict(sd["sd_vae_dec"])

    def load_ckpt_from_url(self, url: str, ckpt_folder: str) -> None:
        os.makedirs(ckpt_folder, exist_ok=True)
        outf = os.path.join(ckpt_folder, os.path.basename(url))
        download_url(url, outf)
        sd = torch.load(outf, map_location="cpu")
        self.load_ckpt_from_state_dict(sd)

    @staticmethod
    def forward_with_networks(
        x: torch.Tensor,
        direction: str,
        vae_enc: VAEEncode,
        unet: UNet2DConditionModel,
        vae_dec: VAEDecode,
        sched,
        timesteps: torch.Tensor,
        text_emb: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        assert direction in ("a2b", "b2a")
        x_enc = vae_enc(x, direction=direction).to(x.dtype)
        model_pred = unet(x_enc, timesteps, encoder_hidden_states=text_emb).sample
        x_out = torch.stack([
            sched.step(model_pred[i], timesteps[i], x_enc[i], return_dict=True).prev_sample
            for i in range(batch_size)
        ])
        return vae_dec(x_out, direction=direction)

    @staticmethod
    def get_trainable_params(unet, vae_a2b, vae_b2a) -> list[nn.Parameter]:
        params = list(unet.conv_in.parameters())
        unet.conv_in.requires_grad_(True)
        unet.set_adapters(["default_encoder", "default_decoder", "default_others"])
        for name, param in unet.named_parameters():
            if "lora" in name and "default" in name:
                assert param.requires_grad
                params.append(param)
        for vae in (vae_a2b, vae_b2a):
            for name, param in vae.named_parameters():
                if "lora" in name and "vae_skip" in name:
                    assert param.requires_grad
                    params.append(param)
            params.extend(vae.decoder.skip_conv_1.parameters())
            params.extend(vae.decoder.skip_conv_2.parameters())
            params.extend(vae.decoder.skip_conv_3.parameters())
            params.extend(vae.decoder.skip_conv_4.parameters())
        return params

    get_traininable_params = get_trainable_params

    def forward(
        self,
        x_t: torch.Tensor,
        direction: Optional[str] = None,
        caption: Optional[str] = None,
        caption_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        direction = direction or self.direction
        if direction is None:
            raise ValueError("direction must be provided for custom checkpoints")
        if caption_emb is None:
            caption = caption or self.caption
            if caption is None:
                raise ValueError("caption must be provided for custom checkpoints")
            tokens = self.tokenizer(
                caption, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt",
            ).input_ids.to(x_t.device)
            caption_emb = self.text_encoder(tokens)[0].detach().clone()
        return self.forward_with_networks(
            x_t, direction, self.vae_enc, self.unet, self.vae_dec,
            self.sched, self.timesteps.to(x_t.device), caption_emb,
        )


# Backward-compatible aliases matching upstream naming.
VAE_encode = VAEEncode
VAE_decode = VAEDecode
CycleGAN_Turbo = CycleGANTurbo

__all__ = [
    "PRETRAINED_CYCLEGAN_TURBO",
    "VAEEncode",
    "VAEDecode",
    "VAE_encode",
    "VAE_decode",
    "initialize_unet",
    "initialize_vae",
    "CycleGANTurbo",
    "CycleGAN_Turbo",
]
