# Copyright (c) 2026 EarthBridge Team.
# Credits: pix2pixHD from Wang et al. "High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs" CVPR 2018.

"""Pix2PixHD trainer for paired image-to-image translation."""

from __future__ import annotations

import functools
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.data.datasets import PairedImageDataset
from src.utils.config_yaml import save_config_yaml
from src.losses import GANLoss, PerceptualLoss
from src.models.discriminators import PatchGANDiscriminator
from src.models.pix2pixhd import Pix2PixHDGenerator

from .config import Pix2PixHDConfig

logger = logging.getLogger(__name__)


def _build_transform(resolution: int):
    return transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
    ])


class Pix2PixHDTrainer:
    """Training harness for Pix2PixHD (Global Generator + PatchGAN)."""

    def __init__(self, config: Pix2PixHDConfig) -> None:
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        self.generator = Pix2PixHDGenerator(
            input_nc=config.input_nc,
            output_nc=config.output_nc,
            ngf=config.ngf,
            n_downsampling=config.n_downsampling,
            n_blocks=config.n_blocks,
            norm_layer=norm_layer,
        ).to(self.device)

        self.discriminator = PatchGANDiscriminator(
            in_channels=config.input_nc + config.output_nc,
            base_filters=config.ndf,
            n_layers=config.n_layers_D,
        ).to(self.device)

        self.criterion_gan = GANLoss(config.gan_mode).to(self.device)
        self.criterion_l1 = nn.L1Loss()
        self.criterion_perceptual = PerceptualLoss().to(self.device) if config.lambda_perceptual > 0 else None

        self.optimizer_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=config.lr_g,
            betas=(config.beta1, 0.999),
        )
        self.optimizer_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=config.lr_d,
            betas=(config.beta1, 0.999),
        )

    def build_dataset(
        self,
        root_source: str | Path,
        root_target: str | Path,
    ) -> PairedImageDataset:
        transform = _build_transform(self.config.resolution)
        return PairedImageDataset(
            root_source=root_source,
            root_target=root_target,
            transform_source=transform,
            transform_target=transform,
        )

    def train_step(self, source: torch.Tensor, target: torch.Tensor) -> dict:
        """Single training step."""
        cfg = self.config
        source = source.to(self.device) * 2 - 1
        target = target.to(self.device) * 2 - 1

        fake = self.generator(source)

        # Update D
        self.optimizer_d.zero_grad()
        pred_real = self.discriminator(torch.cat([source, target], dim=1))
        pred_fake = self.discriminator(torch.cat([source, fake.detach()], dim=1))
        loss_d_real = self.criterion_gan(pred_real, target_is_real=True)
        loss_d_fake = self.criterion_gan(pred_fake, target_is_real=False)
        loss_d = (loss_d_real + loss_d_fake) * 0.5
        loss_d.backward()
        self.optimizer_d.step()

        # Update G
        self.optimizer_g.zero_grad()
        pred_fake = self.discriminator(torch.cat([source, fake], dim=1))
        loss_gan = self.criterion_gan(pred_fake, target_is_real=True)
        loss_l1 = self.criterion_l1(fake, target) * cfg.lambda_l1
        loss_g = loss_gan + loss_l1
        if self.criterion_perceptual is not None:
            loss_perc = self.criterion_perceptual(fake, target) * cfg.lambda_perceptual
            loss_g = loss_g + loss_perc
        else:
            loss_perc = torch.tensor(0.0, device=self.device)
        loss_g.backward()
        self.optimizer_g.step()

        return {
            "loss_d": loss_d.item(),
            "loss_gan": loss_gan.item(),
            "loss_l1": loss_l1.item(),
            "loss_perc": loss_perc.item() if torch.is_tensor(loss_perc) else loss_perc,
        }

    def train(
        self,
        root_source: str | Path,
        root_target: str | Path,
    ) -> None:
        cfg = self.config
        os.makedirs(cfg.save_dir, exist_ok=True)

        dataset = self.build_dataset(root_source, root_target)
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )

        global_step = 0
        for epoch in range(cfg.epochs):
            self.generator.train()
            self.discriminator.train()
            pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{cfg.epochs}")

            for batch in pbar:
                logs = self.train_step(batch["source"], batch["target"])
                global_step += 1
                if global_step % cfg.log_every == 0:
                    pbar.set_postfix(logs)
                    logger.info("step %d | loss_d=%.4f loss_gan=%.4f loss_l1=%.4f", global_step, logs["loss_d"], logs["loss_gan"], logs["loss_l1"])

            if (epoch + 1) % cfg.save_every == 0:
                self.save_checkpoint(cfg.save_dir, epoch + 1, global_step=global_step)

        logger.info("Pix2PixHD training complete. Checkpoints saved to %s", cfg.save_dir)

    def save_checkpoint(
        self,
        save_dir: str,
        epoch: int,
        *,
        global_step: int | None = None,
    ) -> None:
        import json
        from safetensors.torch import save_file

        path = Path(save_dir) / f"checkpoint-epoch-{epoch}"
        path.mkdir(parents=True, exist_ok=True)
        cfg = self.config
        gen_path = path / "generator"
        gen_path.mkdir(exist_ok=True)
        gen_config = {
            "input_nc": cfg.input_nc,
            "output_nc": cfg.output_nc,
            "ngf": cfg.ngf,
            "n_downsampling": cfg.n_downsampling,
            "n_blocks": cfg.n_blocks,
        }
        with open(gen_path / "config.json", "w", encoding="utf-8") as f:
            json.dump(gen_config, f, indent=2)
        save_file(self.generator.state_dict(), gen_path / "diffusion_pytorch_model.safetensors")
        disc_path = path / "discriminator"
        disc_path.mkdir(exist_ok=True)
        save_file(self.discriminator.state_dict(), disc_path / "diffusion_pytorch_model.safetensors")
        save_config_yaml(
            self.config,
            path / "config.yaml",
            extra={"epoch": epoch, "global_step": global_step if global_step is not None else epoch},
        )

        training_state = {
            "optimizer_g": self.optimizer_g.state_dict(),
            "optimizer_d": self.optimizer_d.state_dict(),
            "epoch": epoch,
            "global_step": global_step if global_step is not None else epoch,
        }
        torch.save(training_state, path / "training_state.pt")
        logger.info("Saved checkpoint to %s (with optimizer state)", path)

    def load_checkpoint(self, path: str | Path) -> dict:
        """Load checkpoint and restore model + optimizer for resume."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        from safetensors.torch import load_file
        gen_sd = load_file(str(path / "generator" / "diffusion_pytorch_model.safetensors"), device=str(self.device))
        self.generator.load_state_dict(gen_sd, strict=True)
        disc_sd = load_file(str(path / "discriminator" / "diffusion_pytorch_model.safetensors"), device=str(self.device))
        self.discriminator.load_state_dict(disc_sd, strict=True)
        train_state_path = path / "training_state.pt"
        if train_state_path.exists():
            ckpt = torch.load(train_state_path, map_location=self.device, weights_only=False)
            self.optimizer_g.load_state_dict(ckpt["optimizer_g"])
            self.optimizer_d.load_state_dict(ckpt["optimizer_d"])
            return {"epoch": ckpt.get("epoch", 0), "global_step": ckpt.get("global_step", 0)}
        return {"epoch": 0, "global_step": 0}


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--save-dir", type=str, default="./checkpoints/pix2pixhd")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    config = Pix2PixHDConfig(save_dir=args.save_dir, epochs=args.epochs, batch_size=args.batch_size)
    trainer = Pix2PixHDTrainer(config)
    trainer.train(args.source, args.target)
