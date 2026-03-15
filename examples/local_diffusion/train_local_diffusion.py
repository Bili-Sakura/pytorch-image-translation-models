# Copyright (c) 2026 EarthBridge Team.
# Credits: Local Diffusion from Kim et al. "Tackling Structural Hallucination in Image Translation with Local Diffusion" ECCV 2024 Oral.

"""Local Diffusion trainer for paired conditional image-to-image translation."""

from __future__ import annotations

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
from src.losses import get_diffusion_loss
from src.models.local_diffusion import create_unet
from src.schedulers.local_diffusion import LocalDiffusionScheduler

from .config import LocalDiffusionConfig

logger = logging.getLogger(__name__)


def _build_transform(resolution: int):
    """Build transform: resize, to tensor."""
    return transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
    ])


class LocalDiffusionTrainer:
    """Training harness for Local Diffusion.

    Uses :class:`PairedImageDataset` for paired source (conditioning)
    and target (ground truth) images.  The model learns to denoise the
    target image conditioned on the source.
    """

    def __init__(self, config: LocalDiffusionConfig) -> None:
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        # Model
        self.model = create_unet(
            dim=config.dim,
            channels=config.channels,
            dim_mults=config.dim_mults,
            resnet_block_groups=config.resnet_block_groups,
            attn_dim_head=config.attn_dim_head,
            attn_heads=config.attn_heads,
            full_attn=config.full_attn,
            cond_in_channels=config.cond_in_channels,
            cond_filters=config.cond_filters,
            init_type=config.init_type,
            init_gain=config.init_gain,
        ).to(self.device)

        # Scheduler
        self.scheduler = LocalDiffusionScheduler(
            num_train_timesteps=config.num_train_timesteps,
            beta_schedule=config.beta_schedule,
            objective=config.objective,
            min_snr_loss_weight=config.min_snr_loss_weight,
            min_snr_gamma=config.min_snr_gamma,
        )

        # Loss (one-line choice: mse | min_snr | sid2 | edm)
        self.loss_fn = get_diffusion_loss(
            loss_type=config.loss_type,
            prediction_type=config.objective,
            sid2_bias=getattr(config, "sid2_bias", -3.0),
            sigma_data=getattr(config, "sigma_data", 0.5),
            loss_norm=getattr(config, "loss_norm", "mse"),
        )

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.lr,
            betas=(config.beta1, config.beta2),
        )

    def build_dataset(
        self,
        root_source: str | Path,
        root_target: str | Path,
    ) -> PairedImageDataset:
        """Build paired dataset with transforms."""
        transform = _build_transform(self.config.resolution)
        return PairedImageDataset(
            root_source=root_source,
            root_target=root_target,
            transform_source=transform,
            transform_target=transform,
        )

    def train_step(self, source: torch.Tensor, target: torch.Tensor) -> dict:
        """Single training step.

        Parameters
        ----------
        source : Tensor (B, C, H, W)
            Source/conditioning images in [0, 1].
        target : Tensor (B, C, H, W)
            Target/ground-truth images in [0, 1].

        Returns
        -------
        dict
            Loss values for logging.
        """
        cfg = self.config
        source = source.to(self.device) * 2 - 1  # [0,1] → [-1,1]
        target = target.to(self.device) * 2 - 1

        b = target.shape[0]
        t = torch.randint(0, cfg.num_train_timesteps, (b,), device=self.device).long()

        # Forward diffusion
        noise = torch.randn_like(target)
        x_t = self.scheduler.q_sample(target, t, noise=noise)

        # Model prediction
        model_output = self.model(x_t, source, t)

        # Compute loss (unified DiffusionLoss)
        loss = self.loss_fn(
            model_output,
            clean_images=target,
            noise=noise,
            timesteps_or_sigma=t,
            scheduler=self.scheduler,
        )

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
        self.optimizer.step()

        return {"loss": loss.item()}

    def train(
        self,
        root_source: str | Path,
        root_target: str | Path,
    ) -> None:
        """Run full training loop."""
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
            self.model.train()
            pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{cfg.epochs}")

            for batch in pbar:
                source = batch["source"]
                target = batch["target"]
                logs = self.train_step(source, target)
                global_step += 1

                if global_step % cfg.log_every == 0:
                    pbar.set_postfix(logs)
                    logger.info(
                        "step %d | loss=%.4f",
                        global_step, logs["loss"],
                    )

            if (epoch + 1) % cfg.save_every == 0:
                self.save_checkpoint(cfg.save_dir, epoch + 1, global_step=global_step)

        logger.info("Local Diffusion training complete. Checkpoints saved to %s", cfg.save_dir)

    def save_checkpoint(
        self,
        save_dir: str,
        epoch: int,
        *,
        global_step: int | None = None,
    ) -> None:
        """Save model checkpoint and optimizer state for resume."""
        path = Path(save_dir) / f"checkpoint-epoch-{epoch}"
        path.mkdir(parents=True, exist_ok=True)

        from safetensors.torch import save_file

        model_path = path / "model"
        model_path.mkdir(exist_ok=True)
        save_file(self.model.state_dict(), model_path / "model.safetensors")
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
        """Load checkpoint and restore model + optimizer for resume."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        from safetensors.torch import load_file
        model_path = path / "model" / "model.safetensors"
        if model_path.exists():
            self.model.load_state_dict(load_file(str(model_path), device=str(self.device)), strict=True)
        train_state_path = path / "training_state.pt"
        if train_state_path.exists():
            ckpt = torch.load(train_state_path, map_location=self.device, weights_only=False)
            self.optimizer.load_state_dict(ckpt["optimizer"])
            return {"epoch": ckpt.get("epoch", 0), "global_step": ckpt.get("global_step", 0)}
        return {"epoch": 0, "global_step": 0}
