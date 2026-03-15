# Copyright (c) 2026 EarthBridge Team.
"""DBIM trainer for paired image-to-image translation."""

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
from src.models.unet import DBIMUNet
from src.schedulers.dbim import DBIMScheduler

from .config import DBIMConfig

logger = logging.getLogger(__name__)


def _build_transform(resolution: int):
    return transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
    ])


class DBIMTrainer:
    """Training harness for DBIM (Diffusion Bridge Implicit Models)."""

    def __init__(self, config: DBIMConfig) -> None:
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        self.unet = DBIMUNet(
            image_size=config.image_size,
            in_channels=config.in_channels,
            model_channels=config.model_channels,
            num_res_blocks=config.num_res_blocks,
            attention_resolutions=config.attention_resolutions,
            dropout=config.dropout,
            condition_mode=config.condition_mode,
        ).to(self.device)

        self.scheduler = DBIMScheduler(
            sigma_min=config.sigma_min,
            sigma_max=config.sigma_max,
            sigma_data=config.sigma_data,
            num_train_timesteps=config.num_train_timesteps,
        )

        self.optimizer = torch.optim.AdamW(
            self.unet.parameters(),
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

    def train_step(self, source: torch.Tensor, target: torch.Tensor) -> dict:
        """DBIM: bridge_sample, predict denoised, MSE loss."""
        source = source.to(self.device) * 2 - 1
        target = target.to(self.device) * 2 - 1

        cfg = self.config
        b = target.shape[0]
        t = torch.rand(b, device=self.device) * (cfg.sigma_max - cfg.sigma_min) + cfg.sigma_min
        t = t.view(-1, 1, 1, 1)

        noise = torch.randn_like(target, device=self.device)
        x_t = self.scheduler.add_noise(target, noise, t.squeeze(), source)

        c_skip, c_out, c_in = self._get_bridge_scalings(t)
        model_input = c_in * x_t
        c_noise = 1000.0 * 0.25 * torch.log(t.clamp(min=1e-44)).squeeze()
        pred_raw = self.unet(model_input, c_noise, xT=source)
        denoised = c_out * pred_raw + c_skip * x_t
        loss = F.mse_loss(denoised, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.unet.parameters(), 1.0)
        self.optimizer.step()

        return {"loss": loss.item()}

    def _get_bridge_scalings(self, sigma: torch.Tensor) -> tuple:
        """DBIM bridge preconditioning."""
        a_t, b_t, c_t = self.scheduler.get_abc(sigma.squeeze())
        sigma_data = self.config.sigma_data
        sigma_4d = sigma
        a_t = a_t.view(-1, 1, 1, 1).to(sigma_4d.device)
        b_t = b_t.view(-1, 1, 1, 1).to(sigma_4d.device)
        c_t = c_t.view(-1, 1, 1, 1).to(sigma_4d.device)
        A = a_t**2 * sigma_data**2 + b_t**2 * sigma_data**2 + c_t**2
        c_in = torch.rsqrt(A.clamp(min=1e-20))
        c_skip = (b_t * sigma_data**2) / A.clamp(min=1e-20)
        c_out = (a_t**2 * sigma_data**4 + sigma_data**2 * c_t**2).clamp(min=1e-20).sqrt() * c_in
        return c_skip, c_out, c_in

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
                logs = self.train_step(batch["source"], batch["target"])
                global_step += 1
                if global_step % cfg.log_every == 0:
                    pbar.set_postfix(logs)
                    logger.info("step %d | loss=%.4f", global_step, logs["loss"])

            if (epoch + 1) % cfg.save_every == 0:
                self.save_checkpoint(cfg.save_dir, epoch + 1, global_step=global_step)

        logger.info("DBIM training complete. Checkpoints saved to %s", cfg.save_dir)

    def save_checkpoint(
        self,
        save_dir: str,
        epoch: int,
        *,
        global_step: int | None = None,
    ) -> None:
        path = Path(save_dir) / f"checkpoint-epoch-{epoch}"
        path.mkdir(parents=True, exist_ok=True)
        self.unet.save_pretrained(path / "unet")
        self.scheduler.save_pretrained(path / "scheduler")
        save_config_yaml(
            self.config, path / "config.yaml",
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
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--save-dir", type=str, default="./checkpoints/dbim")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    config = DBIMConfig(save_dir=args.save_dir, epochs=args.epochs, batch_size=args.batch_size)
    trainer = DBIMTrainer(config)
    trainer.train(args.source, args.target)
