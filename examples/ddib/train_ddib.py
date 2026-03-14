# Copyright (c) 2026 EarthBridge Team.
"""DDIB trainer. Trains source_unet on source domain and target_unet on target domain."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.data.datasets import PairedImageDataset
from src.models.unet import DDIBUNet
from src.schedulers.ddib import DDIBScheduler

from .config import DDIBConfig

logger = logging.getLogger(__name__)


def _build_transform(resolution: int):
    return transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
    ])


class DDIBTrainer:
    """Trains source and target UNets separately on their domains."""

    def __init__(self, config: DDIBConfig) -> None:
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        shared_kw = dict(
            image_size=config.image_size,
            in_channels=config.in_channels,
            model_channels=config.model_channels,
            num_res_blocks=config.num_res_blocks,
            attention_resolutions=config.attention_resolutions,
            dropout=config.dropout,
        )

        self.source_unet = DDIBUNet(**shared_kw, learn_sigma=False).to(self.device)
        self.target_unet = DDIBUNet(**shared_kw, learn_sigma=False).to(self.device)
        self.scheduler = DDIBScheduler(
            num_train_timesteps=config.num_train_timesteps,
            noise_schedule=config.noise_schedule,
            predict_xstart=config.predict_xstart,
        )

        self.opt_source = torch.optim.AdamW(
            self.source_unet.parameters(),
            lr=config.lr,
            betas=(config.beta1, config.beta2),
        )
        self.opt_target = torch.optim.AdamW(
            self.target_unet.parameters(),
            lr=config.lr,
            betas=(config.beta1, config.beta2),
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

    def train_step_source(self, source: torch.Tensor) -> dict:
        """Train source_unet on source domain (standard DDPM)."""
        x = source.to(self.device) * 2 - 1
        b = x.shape[0]
        t = torch.randint(0, self.config.num_train_timesteps, (b,), device=self.device)
        noise = torch.randn_like(x, device=self.device)
        x_t = self.scheduler.add_noise(x, noise, t)
        pred = self.source_unet(x_t, t)
        if self.config.predict_xstart:
            target = x
        else:
            target = noise
        loss = F.mse_loss(pred, target)
        self.opt_source.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.source_unet.parameters(), 1.0)
        self.opt_source.step()
        return {"loss_source": loss.item()}

    def train_step_target(self, target: torch.Tensor) -> dict:
        """Train target_unet on target domain (standard DDPM)."""
        x = target.to(self.device) * 2 - 1
        b = x.shape[0]
        t = torch.randint(0, self.config.num_train_timesteps, (b,), device=self.device)
        noise = torch.randn_like(x, device=self.device)
        x_t = self.scheduler.add_noise(x, noise, t)
        pred = self.target_unet(x_t, t)
        if self.config.predict_xstart:
            target_val = x
        else:
            target_val = noise
        loss = F.mse_loss(pred, target_val)
        self.opt_target.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.target_unet.parameters(), 1.0)
        self.opt_target.step()
        return {"loss_target": loss.item()}

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
            self.source_unet.train()
            self.target_unet.train()
            pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{cfg.epochs}")

            for batch in pbar:
                logs_s = self.train_step_source(batch["source"])
                logs_t = self.train_step_target(batch["target"])
                logs = {**logs_s, **logs_t}
                global_step += 1

                if global_step % cfg.log_every == 0:
                    pbar.set_postfix(logs)
                    logger.info("step %d | loss_source=%.4f loss_target=%.4f",
                                global_step, logs["loss_source"], logs["loss_target"])

            if (epoch + 1) % cfg.save_every == 0:
                self.save_checkpoint(cfg.save_dir, epoch + 1)

        logger.info("DDIB training complete. Checkpoints saved to %s", cfg.save_dir)

    def save_checkpoint(self, save_dir: str, epoch: int) -> None:
        path = Path(save_dir) / f"checkpoint-epoch-{epoch}"
        path.mkdir(parents=True, exist_ok=True)
        self.source_unet.save_pretrained(path / "source_unet")
        self.target_unet.save_pretrained(path / "target_unet")
        self.scheduler.save_pretrained(path / "scheduler")
        logger.info("Saved checkpoint to %s", path)


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--save-dir", type=str, default="./checkpoints/ddib")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    config = DDIBConfig(save_dir=args.save_dir, epochs=args.epochs, batch_size=args.batch_size)
    trainer = DDIBTrainer(config)
    trainer.train(args.source, args.target)
