# Copyright (c) 2026 EarthBridge Team.
# Credits: img2img-turbo from Parmar et al. 2024 —
# https://github.com/GaParmar/img2img-turbo

"""pix2pix-turbo model (one-step paired SD-Turbo translation)."""

from __future__ import annotations

import copy
import os
from typing import Optional, Union

import torch
import torch.nn as nn
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.utils.peft_utils import set_weights_and_activate_adapters
from peft import LoraConfig
from transformers import AutoTokenizer, CLIPTextModel

from src.models.img2img_turbo.utils import (
    download_url,
    make_1step_sched,
    my_vae_decoder_fwd,
    my_vae_encoder_fwd,
)

PRETRAINED_PIX2PIX_TURBO: dict[str, str] = {
    "edge_to_image": "https://www.cs.cmu.edu/~img2img-turbo/models/edge_to_image_loras.pkl",
    "sketch_to_image_stochastic": "https://www.cs.cmu.edu/~img2img-turbo/models/sketch_to_image_stochastic_lora.pkl",
}


class TwinConv(nn.Module):
    """Blend pretrained and fine-tuned conv_in weights for stochastic sketch translation."""

    def __init__(self, conv_pretrained: nn.Module, conv_curr: nn.Module) -> None:
        super().__init__()
        self.conv_in_pretrained = copy.deepcopy(conv_pretrained)
        self.conv_in_curr = copy.deepcopy(conv_curr)
        self.r: Optional[float] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv_in_pretrained(x).detach()
        x2 = self.conv_in_curr(x)
        r = self.r if self.r is not None else 1.0
        return x1 * (1 - r) + x2 * r


class Pix2PixTurbo(nn.Module):
    """One-step paired image translation built on SD-Turbo."""

    def __init__(
        self,
        *,
        pretrained_name: Optional[str] = None,
        pretrained_path: Optional[str] = None,
        ckpt_folder: str = "checkpoints",
        lora_rank_unet: int = 8,
        lora_rank_vae: int = 4,
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
        vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
        vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
        vae.decoder.skip_conv_1 = nn.Conv2d(512, 512, kernel_size=1, stride=1, bias=False)
        vae.decoder.skip_conv_2 = nn.Conv2d(256, 512, kernel_size=1, stride=1, bias=False)
        vae.decoder.skip_conv_3 = nn.Conv2d(128, 512, kernel_size=1, stride=1, bias=False)
        vae.decoder.skip_conv_4 = nn.Conv2d(128, 256, kernel_size=1, stride=1, bias=False)
        vae.decoder.ignore_skip = False
        unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet")
        self.pretrained_name = pretrained_name

        if pretrained_name == "edge_to_image":
            p_ckpt = self._resolve_pretrained_ckpt(pretrained_name, ckpt_folder)
            self._load_lora_checkpoint(vae, unet, p_ckpt)
        elif pretrained_name == "sketch_to_image_stochastic":
            p_ckpt = self._resolve_pretrained_ckpt(pretrained_name, ckpt_folder)
            conv_pretrained = copy.deepcopy(unet.conv_in)
            unet.conv_in = TwinConv(conv_pretrained, unet.conv_in)
            self._load_lora_checkpoint(vae, unet, p_ckpt)
        elif pretrained_path is not None:
            sd = torch.load(pretrained_path, map_location="cpu")
            self._load_lora_checkpoint(vae, unet, sd)
        elif pretrained_name is None:
            self._init_random_lora(vae, unet, lora_rank_unet, lora_rank_vae)

        self.unet = unet.to(self.device)
        self.vae = vae.to(self.device)
        self.vae.decoder.gamma = 1
        self.timesteps = torch.tensor([999], device=self.device, dtype=torch.long)
        self.text_encoder.requires_grad_(False)

    def _resolve_pretrained_ckpt(self, name: str, ckpt_folder: str) -> str:
        if name not in PRETRAINED_PIX2PIX_TURBO:
            raise ValueError(f"Unknown pretrained_name {name!r}")
        url = PRETRAINED_PIX2PIX_TURBO[name]
        os.makedirs(ckpt_folder, exist_ok=True)
        outf = os.path.join(ckpt_folder, os.path.basename(url))
        download_url(url, outf)
        return outf

    def _load_lora_checkpoint(self, vae, unet, sd_or_path) -> None:
        sd = torch.load(sd_or_path, map_location="cpu") if isinstance(sd_or_path, (str, os.PathLike)) else sd_or_path
        unet_lora_config = LoraConfig(
            r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["unet_lora_target_modules"],
        )
        vae_lora_config = LoraConfig(
            r=sd["rank_vae"], init_lora_weights="gaussian", target_modules=sd["vae_lora_target_modules"],
        )
        vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
        state_vae = vae.state_dict()
        for key, value in sd["state_dict_vae"].items():
            state_vae[key] = value
        vae.load_state_dict(state_vae)
        unet.add_adapter(unet_lora_config)
        state_unet = unet.state_dict()
        for key, value in sd["state_dict_unet"].items():
            state_unet[key] = value
        unet.load_state_dict(state_unet)
        self.lora_rank_unet = sd["rank_unet"]
        self.lora_rank_vae = sd["rank_vae"]
        self.target_modules_vae = sd["vae_lora_target_modules"]
        self.target_modules_unet = sd["unet_lora_target_modules"]

    def _init_random_lora(self, vae, unet, lora_rank_unet: int, lora_rank_vae: int) -> None:
        for conv in (vae.decoder.skip_conv_1, vae.decoder.skip_conv_2, vae.decoder.skip_conv_3, vae.decoder.skip_conv_4):
            nn.init.constant_(conv.weight, 1e-5)
        target_modules_vae = [
            "conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
            "skip_conv_1", "skip_conv_2", "skip_conv_3", "skip_conv_4",
            "to_k", "to_q", "to_v", "to_out.0",
        ]
        vae.add_adapter(
            LoraConfig(r=lora_rank_vae, init_lora_weights="gaussian", target_modules=target_modules_vae),
            adapter_name="vae_skip",
        )
        target_modules_unet = [
            "to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2",
            "conv_shortcut", "conv_out", "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj",
        ]
        unet.add_adapter(
            LoraConfig(r=lora_rank_unet, init_lora_weights="gaussian", target_modules=target_modules_unet),
        )
        self.lora_rank_unet = lora_rank_unet
        self.lora_rank_vae = lora_rank_vae
        self.target_modules_vae = target_modules_vae
        self.target_modules_unet = target_modules_unet

    def set_eval(self) -> None:
        self.unet.eval()
        self.vae.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)

    def set_train(self) -> None:
        self.unet.train()
        self.vae.train()
        for name, param in self.unet.named_parameters():
            if "lora" in name:
                param.requires_grad = True
        self.unet.conv_in.requires_grad_(True)
        for name, param in self.vae.named_parameters():
            if "lora" in name:
                param.requires_grad = True
        for conv in (self.vae.decoder.skip_conv_1, self.vae.decoder.skip_conv_2,
                     self.vae.decoder.skip_conv_3, self.vae.decoder.skip_conv_4):
            conv.requires_grad_(True)

    def forward(
        self,
        c_t: torch.Tensor,
        prompt: Optional[str] = None,
        prompt_tokens: Optional[torch.Tensor] = None,
        *,
        deterministic: bool = True,
        r: float = 1.0,
        noise_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if (prompt is None) == (prompt_tokens is None):
            raise ValueError("Provide exactly one of prompt or prompt_tokens")
        if prompt is not None:
            tokens = self.tokenizer(
                prompt, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt",
            ).input_ids.to(c_t.device)
            caption_enc = self.text_encoder(tokens)[0]
        else:
            caption_enc = self.text_encoder(prompt_tokens)[0]

        timesteps = self.timesteps.to(c_t.device)
        if deterministic:
            encoded = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor
            model_pred = self.unet(encoded, timesteps, encoder_hidden_states=caption_enc).sample
            x_denoised = self.sched.step(model_pred, timesteps, encoded, return_dict=True).prev_sample
            x_denoised = x_denoised.to(model_pred.dtype)
            self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
            return self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample.clamp(-1, 1)

        self.unet.set_adapters(["default"], weights=[r])
        set_weights_and_activate_adapters(self.vae, ["vae_skip"], [r])
        encoded = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor
        if noise_map is None:
            raise ValueError("noise_map is required for stochastic inference")
        unet_input = encoded * r + noise_map * (1 - r)
        if isinstance(self.unet.conv_in, TwinConv):
            self.unet.conv_in.r = r
        unet_output = self.unet(unet_input, timesteps, encoder_hidden_states=caption_enc).sample
        if isinstance(self.unet.conv_in, TwinConv):
            self.unet.conv_in.r = None
        x_denoised = self.sched.step(unet_output, timesteps, unet_input, return_dict=True).prev_sample
        x_denoised = x_denoised.to(unet_output.dtype)
        self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
        self.vae.decoder.gamma = r
        return self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample.clamp(-1, 1)

    def save_model(self, outf: str) -> None:
        sd = {
            "unet_lora_target_modules": self.target_modules_unet,
            "vae_lora_target_modules": self.target_modules_vae,
            "rank_unet": self.lora_rank_unet,
            "rank_vae": self.lora_rank_vae,
            "state_dict_unet": {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k},
            "state_dict_vae": {k: v for k, v in self.vae.state_dict().items() if "lora" in k or "skip" in k},
        }
        torch.save(sd, outf)


Pix2Pix_Turbo = Pix2PixTurbo

__all__ = [
    "PRETRAINED_PIX2PIX_TURBO",
    "TwinConv",
    "Pix2PixTurbo",
    "Pix2Pix_Turbo",
]
