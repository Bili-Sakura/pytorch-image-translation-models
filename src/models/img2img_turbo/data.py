# Copyright (c) 2026 EarthBridge Team.
# Credits: img2img-turbo from Parmar et al. 2024 —
# https://github.com/GaParmar/img2img-turbo

"""Dataset and transform utilities for img2img-turbo training."""

from __future__ import annotations

import json
import os
import random
from glob import glob
from typing import Callable

import torch
import torchvision.transforms.functional as F
from PIL import Image
from torchvision import transforms


def build_transform(image_prep: str) -> Callable:
    """Build an image preprocessing transform from a preset name."""
    if image_prep == "resized_crop_512":
        return transforms.Compose([
            transforms.Resize(512, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(512),
        ])
    if image_prep == "resize_286_randomcrop_256x256_hflip":
        return transforms.Compose([
            transforms.Resize((286, 286), interpolation=Image.LANCZOS),
            transforms.RandomCrop((256, 256)),
            transforms.RandomHorizontalFlip(),
        ])
    if image_prep in ("resize_256", "resize_256x256"):
        return transforms.Compose([transforms.Resize((256, 256), interpolation=Image.LANCZOS)])
    if image_prep in ("resize_512", "resize_512x512"):
        return transforms.Compose([transforms.Resize((512, 512), interpolation=Image.LANCZOS)])
    if image_prep == "no_resize":
        return transforms.Lambda(lambda x: x)
    raise ValueError(f"Unknown image_prep preset: {image_prep!r}")


class PairedTurboDataset(torch.utils.data.Dataset):
    """Paired dataset for pix2pix-turbo (train_A/train_B + prompts JSON)."""

    def __init__(self, dataset_folder: str, split: str, image_prep: str, tokenizer) -> None:
        super().__init__()
        if split == "train":
            self.input_folder = os.path.join(dataset_folder, "train_A")
            self.output_folder = os.path.join(dataset_folder, "train_B")
            captions_path = os.path.join(dataset_folder, "train_prompts.json")
        elif split == "test":
            self.input_folder = os.path.join(dataset_folder, "test_A")
            self.output_folder = os.path.join(dataset_folder, "test_B")
            captions_path = os.path.join(dataset_folder, "test_prompts.json")
        else:
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")
        with open(captions_path, encoding="utf-8") as f:
            self.captions = json.load(f)
        self.img_names = list(self.captions.keys())
        self.transform = build_transform(image_prep)
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.captions)

    def __getitem__(self, idx: int) -> dict:
        img_name = self.img_names[idx]
        input_img = Image.open(os.path.join(self.input_folder, img_name))
        output_img = Image.open(os.path.join(self.output_folder, img_name))
        caption = self.captions[img_name]
        input_t = F.to_tensor(self.transform(input_img))
        output_t = F.normalize(F.to_tensor(self.transform(output_img)), mean=[0.5], std=[0.5])
        input_ids = self.tokenizer(
            caption, max_length=self.tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt",
        ).input_ids
        return {
            "output_pixel_values": output_t,
            "conditioning_pixel_values": input_t,
            "caption": caption,
            "input_ids": input_ids,
        }


class UnpairedTurboDataset(torch.utils.data.Dataset):
    """Unpaired dataset for CycleGAN-Turbo (train_A/train_B + fixed prompts)."""

    def __init__(self, dataset_folder: str, split: str, image_prep: str, tokenizer) -> None:
        super().__init__()
        if split == "train":
            self.source_folder = os.path.join(dataset_folder, "train_A")
            self.target_folder = os.path.join(dataset_folder, "train_B")
        elif split == "test":
            self.source_folder = os.path.join(dataset_folder, "test_A")
            self.target_folder = os.path.join(dataset_folder, "test_B")
        else:
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")
        self.tokenizer = tokenizer
        with open(os.path.join(dataset_folder, "fixed_prompt_a.txt"), encoding="utf-8") as f:
            self.fixed_caption_src = f.read().strip()
            self.input_ids_src = self.tokenizer(
                self.fixed_caption_src, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt",
            ).input_ids
        with open(os.path.join(dataset_folder, "fixed_prompt_b.txt"), encoding="utf-8") as f:
            self.fixed_caption_tgt = f.read().strip()
            self.input_ids_tgt = self.tokenizer(
                self.fixed_caption_tgt, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt",
            ).input_ids
        self.l_imgs_src, self.l_imgs_tgt = [], []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"):
            self.l_imgs_src.extend(glob(os.path.join(self.source_folder, ext)))
            self.l_imgs_tgt.extend(glob(os.path.join(self.target_folder, ext)))
        self.transform = build_transform(image_prep)

    def __len__(self) -> int:
        return len(self.l_imgs_src) + len(self.l_imgs_tgt)

    def __getitem__(self, index: int) -> dict:
        img_path_src = self.l_imgs_src[index] if index < len(self.l_imgs_src) else random.choice(self.l_imgs_src)
        img_path_tgt = random.choice(self.l_imgs_tgt)
        img_src = F.normalize(F.to_tensor(self.transform(Image.open(img_path_src).convert("RGB"))), mean=[0.5], std=[0.5])
        img_tgt = F.normalize(F.to_tensor(self.transform(Image.open(img_path_tgt).convert("RGB"))), mean=[0.5], std=[0.5])
        return {
            "pixel_values_src": img_src,
            "pixel_values_tgt": img_tgt,
            "caption_src": self.fixed_caption_src,
            "caption_tgt": self.fixed_caption_tgt,
            "input_ids_src": self.input_ids_src,
            "input_ids_tgt": self.input_ids_tgt,
        }


__all__ = [
    "build_transform",
    "PairedTurboDataset",
    "UnpairedTurboDataset",
]
