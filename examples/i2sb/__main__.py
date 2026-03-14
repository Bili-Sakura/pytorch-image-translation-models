# Copyright (c) 2026 EarthBridge Team.
# Credits: I2SB from Liu et al. "I2SB: Image-to-Image Schrödinger Bridge" ICML 2024.
"""I2SB training entry point: python -m examples.i2sb --source DIR --target DIR."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from examples.i2sb.config import TaskConfig
from examples.i2sb.trainer import I2SBTrainer
from src.data.datasets import PairedImageDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _build_transform(resolution: int):
    return transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="Root dir for source images")
    parser.add_argument("--target", type=str, required=True, help="Root dir for target images")
    parser.add_argument("--save-dir", type=str, default="./checkpoints/i2sb")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resolution", type=int, default=256)
    args = parser.parse_args()

    cfg = TaskConfig(
        resolution=args.resolution,
        train_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
    )

    trainer = I2SBTrainer(cfg)
    model = trainer.build_model()
    scheduler = trainer.build_scheduler()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    transform = _build_transform(cfg.resolution)
    dataset = PairedImageDataset(
        root_source=args.source,
        root_target=args.target,
        transform_source=transform,
        transform_target=transform,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    os.makedirs(args.save_dir, exist_ok=True)
    global_step = 0

    for epoch in range(cfg.num_train_epochs):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{cfg.num_train_epochs}")
        for batch in pbar:
            source = batch["source"].to(device) * 2 - 1
            target = batch["target"].to(device) * 2 - 1
            loss = trainer.compute_training_loss(model, scheduler, source, target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            global_step += 1
            if global_step % 100 == 0:
                pbar.set_postfix(loss=loss.item())
                logger.info("step %d | loss=%.4f", global_step, loss.item())

        if (epoch + 1) % 10 == 0:
            ckpt_path = Path(args.save_dir) / f"checkpoint-epoch-{epoch + 1}"
            ckpt_path.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_path / "model.pt")
            logger.info("Saved checkpoint to %s", ckpt_path)

    logger.info("I2SB training complete. Checkpoints saved to %s", args.save_dir)


if __name__ == "__main__":
    main()
