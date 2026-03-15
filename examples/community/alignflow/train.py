# Copyright (c) 2026 EarthBridge Team.
# Credits: AlignFlow (Grover et al., AAAI 2020) - https://github.com/ermongroup/alignflow

"""Training harness for AlignFlow (CycleFlow, Flow2Flow)."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.data.datasets import UnpairedImageDataset
from src.utils.config_yaml import save_config_yaml

from .config import AlignFlowConfig
from .models import CycleFlow, Flow2Flow

logger = logging.getLogger(__name__)

MODEL_MAP = {"cycleflow": CycleFlow, "flow2flow": Flow2Flow}


def _build_transform(resolution: int):
    """Resize, to tensor, normalize to [-1, 1] (AlignFlow convention)."""
    return transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


class AlignFlowTrainer:
    """Trainer for AlignFlow CycleFlow and Flow2Flow models.

    Uses unpaired images from domain A and B. Expects images in [0,1];
    internally converts to [-1,1] for the flow models.
    """

    def __init__(self, config: AlignFlowConfig) -> None:
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        model_cls = MODEL_MAP.get(config.model.lower())
        if model_cls is None:
            raise ValueError(
                f"Unknown model: {config.model_name}. Choose from {list(MODEL_MAP.keys())}"
            )

        args = config.to_args()
        self.model = model_cls(args)
        self.model = self.model.to(self.device)
        self.model.train()

    def build_dataset(
        self,
        root_a: str | Path,
        root_b: str | Path,
    ) -> UnpairedImageDataset:
        """Build unpaired dataset with AlignFlow-compatible transforms."""
        transform = _build_transform(self.config.resolution)
        return UnpairedImageDataset(
            root_a=root_a,
            root_b=root_b,
            transform_a=transform,
            transform_b=transform,
        )

    def train_step(self, real_a: torch.Tensor, real_b: torch.Tensor) -> dict:
        """Single training step. Inputs expected in [-1, 1]."""
        real_a = real_a.to(self.device)
        real_b = real_b.to(self.device)
        self.model.set_inputs(real_a, real_b)
        self.model.train_iter()
        return self.model.get_loss_dict()

    def train(
        self,
        root_a: str | Path,
        root_b: str | Path,
    ) -> None:
        """Run full training loop."""
        cfg = self.config
        os.makedirs(cfg.save_dir, exist_ok=True)

        dataset = self.build_dataset(root_a, root_b)
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )

        global_step = 0
        for epoch in range(cfg.epochs):
            self.model.train()
            pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{cfg.epochs}")

            for batch in pbar:
                real_a = batch["A"]
                real_b = batch["B"]
                logs = self.train_step(real_a, real_b)
                global_step += 1
                pbar.set_postfix(logs)

                if global_step % cfg.log_every == 0:
                    logger.info(
                        "step %d | loss_g=%.4f loss_d=%.4f",
                        global_step,
                        logs.get("loss_g", 0),
                        logs.get("loss_d", 0),
                    )

            self.model.on_epoch_end()

            if (epoch + 1) % cfg.save_every == 0:
                self.save_checkpoint(cfg.save_dir, epoch + 1, global_step=global_step)

        logger.info("AlignFlow training complete. Checkpoints saved to %s", cfg.save_dir)

    def save_checkpoint(
        self,
        save_dir: str,
        epoch: int,
        *,
        global_step: int | None = None,
    ) -> None:
        """Save model checkpoint and config."""
        path = Path(save_dir) / f"checkpoint-epoch-{epoch}"
        path.mkdir(parents=True, exist_ok=True)
        step = global_step if global_step is not None else epoch
        state = {"model": self.model.state_dict(), "epoch": epoch, "global_step": step}
        torch.save(state, path / "alignflow.pt")
        save_config_yaml(
            self.config,
            path / "config.yaml",
            extra={"epoch": epoch, "global_step": step},
        )
        logger.info("Saved checkpoint to %s", path)
