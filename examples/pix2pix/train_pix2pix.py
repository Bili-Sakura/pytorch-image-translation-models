# Credits: pix2pix (Isola et al., CVPR 2017) - https://github.com/phillipi/pix2pix
"""Training loop for paired image translation (Pix2Pix-style)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training a paired image translation model.

    Parameters
    ----------
    epochs:
        Total number of training epochs.
    lr_g:
        Learning rate for the generator.
    lr_d:
        Learning rate for the discriminator.
    beta1:
        Adam beta-1 parameter.
    lambda_l1:
        Weight for L1 reconstruction loss.
    lambda_perceptual:
        Weight for perceptual loss (0 to disable).
    save_dir:
        Directory to save checkpoints.
    save_interval:
        Save a checkpoint every *n* epochs.
    log_interval:
        Log training metrics every *n* steps.
    device:
        Device to train on (``"cuda"`` or ``"cpu"``).
    """

    epochs: int = 200
    lr_g: float = 2e-4
    lr_d: float = 2e-4
    beta1: float = 0.5
    lambda_l1: float = 100.0
    lambda_perceptual: float = 0.0
    save_dir: str = "checkpoints"
    save_interval: int = 10
    log_interval: int = 100
    device: str = "cuda"
    optimizer: str = "adamw"  # "adamw" | "adam"
    extra: dict[str, Any] = field(default_factory=dict)


class Pix2PixTrainer:
    """Trainer for paired image-to-image translation.

    Wraps a generator and discriminator with GAN + L1 loss training.

    Parameters
    ----------
    generator:
        The generator network.
    discriminator:
        The discriminator network.
    config:
        Training hyper-parameters.
    gan_loss:
        GAN loss module (defaults to vanilla BCE).
    perceptual_loss:
        Optional perceptual loss module.
    """

    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        config: TrainingConfig | None = None,
        gan_loss: nn.Module | None = None,
        perceptual_loss: nn.Module | None = None,
    ) -> None:
        from src.losses.adversarial import GANLoss

        self.config = config or TrainingConfig()
        self.device = torch.device(self.config.device)

        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)

        self.gan_loss = (gan_loss or GANLoss("vanilla")).to(self.device)
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = perceptual_loss.to(self.device) if perceptual_loss else None

        betas = (self.config.beta1, 0.999)
        opt_cls = torch.optim.AdamW if self.config.optimizer.lower() == "adamw" else torch.optim.Adam
        opt_kw = dict(betas=betas)
        if opt_cls == torch.optim.AdamW:
            opt_kw["weight_decay"] = 0.01

        self.optimizer_g = opt_cls(self.generator.parameters(), lr=self.config.lr_g, **opt_kw)
        self.optimizer_d = opt_cls(self.discriminator.parameters(), lr=self.config.lr_d, **opt_kw)

    # ------------------------------------------------------------------
    # Training steps
    # ------------------------------------------------------------------

    def _train_discriminator(
        self, source: torch.Tensor, target: torch.Tensor, fake: torch.Tensor
    ) -> torch.Tensor:
        self.optimizer_d.zero_grad()

        # Real
        real_input = torch.cat([source, target], dim=1)
        pred_real = self.discriminator(real_input)
        loss_real = self.gan_loss(pred_real, target_is_real=True)

        # Fake
        fake_input = torch.cat([source, fake.detach()], dim=1)
        pred_fake = self.discriminator(fake_input)
        loss_fake = self.gan_loss(pred_fake, target_is_real=False)

        loss_d = (loss_real + loss_fake) * 0.5
        loss_d.backward()
        self.optimizer_d.step()
        return loss_d

    def _train_generator(
        self, source: torch.Tensor, target: torch.Tensor, fake: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        self.optimizer_g.zero_grad()

        fake_input = torch.cat([source, fake], dim=1)
        pred_fake = self.discriminator(fake_input)
        loss_gan = self.gan_loss(pred_fake, target_is_real=True)
        loss_l1 = self.l1_loss(fake, target) * self.config.lambda_l1

        loss_g = loss_gan + loss_l1
        losses = {"g_gan": loss_gan, "g_l1": loss_l1}

        if self.perceptual_loss is not None and self.config.lambda_perceptual > 0:
            loss_perc = self.perceptual_loss(fake, target) * self.config.lambda_perceptual
            loss_g = loss_g + loss_perc
            losses["g_perceptual"] = loss_perc

        loss_g.backward()
        self.optimizer_g.step()
        losses["g_total"] = loss_g
        return losses

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_epoch(self, dataloader: DataLoader) -> dict[str, float]:
        """Run one training epoch.

        Parameters
        ----------
        dataloader:
            Must yield dicts with ``"source"`` and ``"target"`` tensors.

        Returns
        -------
        dict:
            Average losses for the epoch.
        """
        self.generator.train()
        self.discriminator.train()

        totals: dict[str, float] = {}
        count = 0

        for step, batch in enumerate(dataloader):
            source = batch["source"].to(self.device)
            target = batch["target"].to(self.device)

            fake = self.generator(source)

            loss_d = self._train_discriminator(source, target, fake)

            # Recompute fake after discriminator update so the generator
            # sees the current discriminator weights.
            fake = self.generator(source)
            losses_g = self._train_generator(source, target, fake)

            # Accumulate
            for k, v in losses_g.items():
                totals[k] = totals.get(k, 0.0) + v.item()
            totals["d"] = totals.get("d", 0.0) + loss_d.item()
            count += 1

            if (step + 1) % self.config.log_interval == 0:
                avg = {k: v / count for k, v in totals.items()}
                logger.info("step %d | %s", step + 1, avg)

        return {k: v / max(count, 1) for k, v in totals.items()}

    def fit(self, dataloader: DataLoader) -> None:
        """Run the full training loop.

        Parameters
        ----------
        dataloader:
            Training data loader yielding ``{"source": …, "target": …}``.
        """
        save_dir = Path(self.config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, self.config.epochs + 1):
            avg_losses = self.train_epoch(dataloader)
            logger.info("Epoch %d/%d | %s", epoch, self.config.epochs, avg_losses)

            if epoch % self.config.save_interval == 0:
                self.save_checkpoint(save_dir / f"epoch_{epoch}.pt")

        self.save_checkpoint(save_dir / "latest.pt")

    def save_checkpoint(self, path: str | Path) -> None:
        """Save generator, discriminator, and optimizer states."""
        torch.save(
            {
                "generator": self.generator.state_dict(),
                "discriminator": self.discriminator.state_dict(),
                "optimizer_g": self.optimizer_g.state_dict(),
                "optimizer_d": self.optimizer_d.state_dict(),
                "config": self.config,
            },
            path,
        )
        logger.info("Checkpoint saved to %s", path)

    def load_checkpoint(self, path: str | Path) -> None:
        """Load a training checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.generator.load_state_dict(ckpt["generator"])
        self.discriminator.load_state_dict(ckpt["discriminator"])
        self.optimizer_g.load_state_dict(ckpt["optimizer_g"])
        self.optimizer_d.load_state_dict(ckpt["optimizer_d"])
        logger.info("Checkpoint loaded from %s", path)
