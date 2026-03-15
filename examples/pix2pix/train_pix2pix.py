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

from src.training import create_optimizer
from src.utils.config_yaml import save_config_yaml

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
    optimizer: str = "adamw"  # "adamw" | "adam" | "prodigy" | "muon"
    weight_decay: float = 0.01
    prodigy_d0: float = 1e-6
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
        weight_decay = self.config.weight_decay if self.config.optimizer.lower() in ("adamw", "prodigy", "muon") else 0.0
        self.optimizer_g = create_optimizer(
            self.generator.parameters(),
            optimizer_type=self.config.optimizer,
            lr=self.config.lr_g,
            weight_decay=weight_decay,
            betas=betas,
            prodigy_d0=self.config.prodigy_d0,
        )
        self.optimizer_d = create_optimizer(
            self.discriminator.parameters(),
            optimizer_type=self.config.optimizer,
            lr=self.config.lr_d,
            weight_decay=weight_decay,
            betas=betas,
            prodigy_d0=self.config.prodigy_d0,
        )

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

        global_step = 0
        steps_per_epoch = len(dataloader)
        for epoch in range(1, self.config.epochs + 1):
            avg_losses = self.train_epoch(dataloader)
            global_step += steps_per_epoch
            logger.info("Epoch %d/%d | %s", epoch, self.config.epochs, avg_losses)

            if epoch % self.config.save_interval == 0:
                self.save_checkpoint(
                    save_dir / f"checkpoint-epoch-{epoch}",
                    epoch=epoch,
                    global_step=global_step,
                )

        self.save_checkpoint(
            save_dir / "latest",
            epoch=self.config.epochs,
            global_step=global_step,
        )

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        epoch: int | None = None,
        global_step: int | None = None,
        use_hf_format: bool = True,
    ) -> None:
        """Save checkpoint in Hugging Face / diffusers style (config.json + safetensors).

        Layout: generator/config.json, generator/diffusion_pytorch_model.safetensors,
                discriminator/config.json, discriminator/diffusion_pytorch_model.safetensors.
        Optionally training_state.pt for optimizer (resume). Set use_hf_format=False for legacy .pt only.
        """
        path = Path(path)
        # Infer epoch from path if not provided (e.g. checkpoint-epoch-10)
        if epoch is None and path.name.startswith("checkpoint-epoch-"):
            try:
                epoch = int(path.name.replace("checkpoint-epoch-", ""))
            except ValueError:
                pass
        extra = {}
        if epoch is not None:
            extra["epoch"] = epoch
        if global_step is not None:
            extra["global_step"] = global_step

        if use_hf_format and not str(path).endswith(".pt"):
            path.mkdir(parents=True, exist_ok=True)
            from safetensors.torch import save_file
            import json

            # Generator config (UNetGenerator)
            gen_dir = path / "generator"
            gen_dir.mkdir(exist_ok=True)
            gen_cfg = {"in_channels": 3, "out_channels": 3, "base_filters": 64, "num_downs": 8, "use_dropout": True}
            if hasattr(self.generator, "_config"):
                gen_cfg.update(getattr(self.generator, "_config", {}))
            with open(gen_dir / "config.json", "w", encoding="utf-8") as f:
                json.dump(gen_cfg, f, indent=2)
            save_file(self.generator.state_dict(), gen_dir / "diffusion_pytorch_model.safetensors")

            # Discriminator config (PatchGANDiscriminator)
            disc_dir = path / "discriminator"
            disc_dir.mkdir(exist_ok=True)
            disc_cfg = {"in_channels": 6, "base_filters": 64, "n_layers": 3}
            if hasattr(self.discriminator, "_config"):
                disc_cfg.update(getattr(self.discriminator, "_config", {}))
            with open(disc_dir / "config.json", "w", encoding="utf-8") as f:
                json.dump(disc_cfg, f, indent=2)
            save_file(self.discriminator.state_dict(), disc_dir / "diffusion_pytorch_model.safetensors")

            # Training state for resume (optional)
            torch.save(
                {
                    "optimizer_g": self.optimizer_g.state_dict(),
                    "optimizer_d": self.optimizer_d.state_dict(),
                    "config": self.config,
                },
                path / "training_state.pt",
            )
            save_config_yaml(self.config, path / "config.yaml", extra=extra or None)
            logger.info("Checkpoint saved to %s (HF format)", path)
        else:
            torch.save(
                {
                    "generator": self.generator.state_dict(),
                    "discriminator": self.discriminator.state_dict(),
                    "optimizer_g": self.optimizer_g.state_dict(),
                    "optimizer_d": self.optimizer_d.state_dict(),
                    "config": self.config,
                },
                path if str(path).endswith(".pt") else path / "model.pt",
            )
            if not str(path).endswith(".pt"):
                save_config_yaml(self.config, path / "config.yaml", extra=extra or None)
            logger.info("Checkpoint saved to %s", path)

    def load_checkpoint(self, path: str | Path) -> None:
        """Load a training checkpoint. Supports HF format (dir with config + safetensors) or legacy .pt."""
        path = Path(path)
        if path.is_dir() and (path / "generator" / "config.json").exists():
            from safetensors.torch import load_file
            gen_state = load_file(str(path / "generator" / "diffusion_pytorch_model.safetensors"), device="cpu")
            self.generator.load_state_dict(gen_state, strict=True)
            disc_state = load_file(str(path / "discriminator" / "diffusion_pytorch_model.safetensors"), device="cpu")
            self.discriminator.load_state_dict(disc_state, strict=True)
            train_state = path / "training_state.pt"
            if train_state.exists():
                ckpt = torch.load(train_state, map_location=self.device, weights_only=False)
                self.optimizer_g.load_state_dict(ckpt["optimizer_g"])
                self.optimizer_d.load_state_dict(ckpt["optimizer_d"])
            logger.info("Checkpoint loaded from %s (HF format)", path)
        else:
            ckpt_path = path if path.suffix == ".pt" else path / "model.pt"
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            self.generator.load_state_dict(ckpt["generator"])
            self.discriminator.load_state_dict(ckpt["discriminator"])
            self.optimizer_g.load_state_dict(ckpt["optimizer_g"])
            self.optimizer_d.load_state_dict(ckpt["optimizer_d"])
            logger.info("Checkpoint loaded from %s", ckpt_path)
