# Copyright (c) 2026 EarthBridge Team.
# Credits: BBDM from Li et al. "BBDM: Image-to-image Diffusion with Brownian Bridge" CVPR 2023.

"""BBDM trainer for paired image-to-image translation."""

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
from src.utils.config_yaml import save_config_yaml
from src.models.unet import BBDMUNet
from src.schedulers.bbdm import BBDMScheduler
from src.training.optimizer_utils import create_optimizer

from .config import BBDMConfig

logger = logging.getLogger(__name__)


def _build_transform(resolution: int):
    return transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
    ])


class BBDMTrainer:
    """Training harness for BBDM.

    Uses PairedImageDataset. Source = conditioning endpoint y, Target = x0.
    Bridge: x0 (target) <-> y (source). Reverse sampling goes y -> x0.
    """

    def __init__(self, config: BBDMConfig) -> None:
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        self.unet = BBDMUNet(
            image_size=config.image_size,
            in_channels=config.in_channels,
            model_channels=config.model_channels,
            num_res_blocks=config.num_res_blocks,
            attention_resolutions=config.attention_resolutions,
            dropout=config.dropout,
            condition_mode=config.condition_mode,
        ).to(self.device)

        self.scheduler = BBDMScheduler(
            num_timesteps=config.num_timesteps,
            mt_type=config.mt_type,
            objective=config.objective,
        )

        weight_decay = config.weight_decay if config.optimizer.lower() in ("adamw", "prodigy", "muon") else 0.0
        self.optimizer = create_optimizer(
            self.unet.parameters(),
            optimizer_type=config.optimizer,
            lr=config.lr,
            weight_decay=weight_decay,
            betas=(config.beta1, config.beta2),
            prodigy_d0=config.prodigy_d0,
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
        """Single training step. source=y (condition), target=x0."""
        cfg = self.config
        source = source.to(self.device) * 2 - 1
        target = target.to(self.device) * 2 - 1

        b = target.shape[0]
        timesteps = torch.randint(
            0, cfg.num_timesteps, (b,), device=self.device, dtype=torch.long,
        )
        noise = torch.randn_like(target, device=self.device)

        x_t = self.scheduler.add_noise(target, source, timesteps, noise=noise)
        objective_target = self.scheduler.get_objective(target, source, timesteps, noise)

        pred = self.unet(x_t, timesteps, context=source)
        loss = F.mse_loss(pred, objective_target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.unet.parameters(), 1.0)
        self.optimizer.step()

        return {"loss": loss.item()}

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
            self.unet.train()
            pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{cfg.epochs}")

            for batch in pbar:
                source = batch["source"]
                target = batch["target"]
                logs = self.train_step(source, target)
                global_step += 1

                if global_step % cfg.log_every == 0:
                    pbar.set_postfix(logs)
                    logger.info("step %d | loss=%.4f", global_step, logs["loss"])

            if (epoch + 1) % cfg.save_every == 0:
                self.save_checkpoint(cfg.save_dir, epoch + 1, global_step=global_step)

        logger.info("BBDM training complete. Checkpoints saved to %s", cfg.save_dir)

    def save_checkpoint(
        self,
        save_dir: str,
        epoch: int,
        *,
        global_step: int | None = None,
    ) -> None:
        path = Path(save_dir) / f"checkpoint-epoch-{epoch}"
        path.mkdir(parents=True, exist_ok=True)
        unet_path = path / "unet"
        unet_path.mkdir(exist_ok=True)
        self.unet.save_pretrained(unet_path)
        scheduler_path = path / "scheduler"
        scheduler_path.mkdir(exist_ok=True)
        self.scheduler.save_pretrained(scheduler_path)
        save_config_yaml(
            self.config,
            path / "config.yaml",
            extra={"epoch": epoch, "global_step": global_step if global_step is not None else epoch},
        )
        training_state = {
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step if global_step is not None else epoch,
        }
        torch.save(training_state, path / "training_state.pt")
        logger.info("Saved checkpoint to %s (with optimizer state)", path)

    def load_checkpoint(self, path: str | Path) -> dict:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        self.unet = self.unet.from_pretrained(path / "unet").to(self.device)
        sched_path = path / "scheduler"
        if sched_path.exists():
            self.scheduler = self.scheduler.from_pretrained(str(sched_path))
        train_state_path = path / "training_state.pt"
        if train_state_path.exists():
            ckpt = torch.load(train_state_path, map_location=self.device, weights_only=False)
            self.optimizer.load_state_dict(ckpt["optimizer"])
            return {"epoch": ckpt.get("epoch", 0), "global_step": ckpt.get("global_step", 0)}
        return {"epoch": 0, "global_step": 0}


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="Root dir for source images")
    parser.add_argument("--target", type=str, required=True, help="Root dir for target images")
    parser.add_argument("--save-dir", type=str, default="./checkpoints/bbdm")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resolution", type=int, default=256)
    args = parser.parse_args()

    config = BBDMConfig(
        save_dir=args.save_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        resolution=args.resolution,
    )
    trainer = BBDMTrainer(config)
    trainer.train(args.source, args.target)
