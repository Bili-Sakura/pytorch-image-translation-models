# Copyright (c) 2026 EarthBridge Team.
# Credits: CUT from Park et al. "Contrastive Learning for Unpaired Image-to-Image Translation" ECCV 2020.

"""CUT trainer for unpaired image-to-image translation."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.data.datasets import UnpairedImageDataset
from src.utils.config_yaml import save_config_yaml
from src.training import create_optimizer
from src.models.cut import (
    create_generator,
    create_discriminator,
    create_patch_sample_mlp,
    GANLoss,
    PatchNCELoss,
)
from src.pipelines.cut import CUTPipeline

from .config import CUTConfig

logger = logging.getLogger(__name__)


def _build_transform(resolution: int):
    """Build transform: resize, random crop, to tensor, normalize to [0,1]."""
    return transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
    ])


def _adapt_for_encoder(x: torch.Tensor, actual_ch: int, encoder_ch: int) -> torch.Tensor:
    """Adapt tensor to encoder's expected channel count."""
    if actual_ch == encoder_ch:
        return x
    if actual_ch < encoder_ch:
        return x.repeat(1, encoder_ch // actual_ch, 1, 1)
    return x.mean(dim=1, keepdim=True)


def _calculate_nce_loss(
    netG, netF, nce_criteria, src, tgt, nce_layers, lambda_nce, num_patches,
    source_channels: int = 3, target_channels: int = 3,
):
    """Compute contrastive loss across multiple encoder layers."""
    if src.shape[1] != source_channels:
        src = _adapt_for_encoder(src, src.shape[1], source_channels)
    if tgt.shape[1] != source_channels:
        tgt = _adapt_for_encoder(tgt, tgt.shape[1], source_channels)
    n_layers = len(nce_layers)
    feat_q = netG(tgt, nce_layers, encode_only=True)
    feat_k = netG(src, nce_layers, encode_only=True)
    feat_k_pool, sample_ids = netF(feat_k, num_patches, None)
    feat_q_pool, _ = netF(feat_q, num_patches, sample_ids)

    total_nce_loss = 0.0
    for f_q, f_k, crit in zip(feat_q_pool, feat_k_pool, nce_criteria):
        loss = crit(f_q, f_k) * lambda_nce
        total_nce_loss += loss.mean()
    return total_nce_loss / n_layers


class CUTTrainer:
    """Training harness for CUT (Contrastive Unpaired Translation).

    Uses :class:`UnpairedImageDataset` for unpaired domain A and B images.
    Generator maps A → B; discriminator and PatchNCE operate on target domain.
    """

    def __init__(self, config: CUTConfig) -> None:
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        # Models
        self.netG = create_generator(
            input_nc=config.input_nc,
            output_nc=config.output_nc,
            ngf=config.ngf,
            netG=config.netG,
            norm_type=config.normG,
            use_dropout=not config.no_dropout,
            no_antialias=config.no_antialias,
            no_antialias_up=config.no_antialias_up,
            init_type=config.init_type,
            init_gain=config.init_gain,
        ).to(self.device)

        self.netD = create_discriminator(
            input_nc=config.output_nc,
            ndf=config.ndf,
            netD=config.netD,
            n_layers_D=config.n_layers_D,
            norm_type=config.normD,
            no_antialias=config.no_antialias,
            init_type=config.init_type,
            init_gain=config.init_gain,
        ).to(self.device)

        self.netF = create_patch_sample_mlp(
            use_mlp=(config.netF == "mlp_sample"),
            nc=config.netF_nc,
            init_type=config.init_type,
            init_gain=config.init_gain,
        ).to(self.device)

        # Losses
        self.criterion_GAN = GANLoss(config.gan_mode).to(self.device)
        nce_layers = [int(i) for i in config.nce_layers.split(",")]
        self.nce_criteria = [
            PatchNCELoss(
                nce_T=config.nce_T,
                batch_size=config.batch_size,
                nce_includes_all_negatives_from_minibatch=config.nce_includes_all_negatives_from_minibatch,
            )
            for _ in nce_layers
        ]
        for c in self.nce_criteria:
            c.to(self.device)
        self.nce_layers = nce_layers

        # Optimizers
        betas = (config.beta1, config.beta2)
        weight_decay = config.weight_decay if config.optimizer.lower() in ("adamw", "prodigy", "muon") else 0.0

        self.optimizer_G = create_optimizer(
            self.netG.parameters(),
            optimizer_type=config.optimizer,
            lr=config.lr,
            weight_decay=weight_decay,
            betas=betas,
            prodigy_d0=config.prodigy_d0,
        )
        self.optimizer_D = create_optimizer(
            self.netD.parameters(),
            optimizer_type=config.optimizer,
            lr=config.lr,
            weight_decay=weight_decay,
            betas=betas,
            prodigy_d0=config.prodigy_d0,
        )
        self.optimizer_F = None  # Created after first forward (netF lazy init)

    def build_dataset(
        self,
        root_a: str | Path,
        root_b: str | Path,
    ) -> UnpairedImageDataset:
        """Build unpaired dataset with transforms."""
        transform = _build_transform(self.config.resolution)
        return UnpairedImageDataset(
            root_a=root_a,
            root_b=root_b,
            transform_a=transform,
            transform_b=transform,
        )

    def _ensure_optimizer_F(self):
        """Create optimizer_F after netF is initialized (first forward)."""
        if self.optimizer_F is None and self.netF.mlp_init:
            weight_decay = self.config.weight_decay if self.config.optimizer.lower() in ("adamw", "prodigy", "muon") else 0.0
            self.optimizer_F = create_optimizer(
                self.netF.parameters(),
                optimizer_type=self.config.optimizer,
                lr=self.config.lr,
                weight_decay=weight_decay,
                betas=(self.config.beta1, self.config.beta2),
                prodigy_d0=self.config.prodigy_d0,
            )

    def train_step(self, real_A: torch.Tensor, real_B: torch.Tensor) -> dict:
        """Single training step. Inputs in [0,1]; scale to [-1,1] internally."""
        cfg = self.config
        real_A = real_A.to(self.device) * 2 - 1
        real_B = real_B.to(self.device) * 2 - 1

        # Forward G
        fake_B = self.netG(real_A)

        # Initialize netF on first step (lazy MLP creation)
        if not self.netF.mlp_init:
            with torch.no_grad():
                fake_B_for_enc = _adapt_for_encoder(
                    fake_B, cfg.output_nc, cfg.input_nc
                )
                feat_init = self.netG(fake_B_for_enc, self.nce_layers, encode_only=True)
                self.netF(feat_init, cfg.num_patches, None)
            self._ensure_optimizer_F()

        # Update D
        self.optimizer_D.zero_grad()
        loss_D = self._compute_D_loss(real_B, fake_B)
        loss_D.backward()
        self.optimizer_D.step()

        # Update G + F
        self.optimizer_G.zero_grad()
        if self.optimizer_F is not None:
            self.optimizer_F.zero_grad()

        loss_G, loss_G_GAN, loss_NCE, loss_NCE_Y = self._compute_G_loss(
            real_A, fake_B, real_B
        )
        loss_G.backward()
        self.optimizer_G.step()
        if self.optimizer_F is not None:
            self.optimizer_F.step()

        return {
            "loss_D": loss_D.item(),
            "loss_G": loss_G.item(),
            "loss_G_GAN": loss_G_GAN.item() if torch.is_tensor(loss_G_GAN) else loss_G_GAN,
            "loss_NCE": loss_NCE.item() if torch.is_tensor(loss_NCE) else loss_NCE,
            "loss_NCE_Y": loss_NCE_Y.item() if torch.is_tensor(loss_NCE_Y) else loss_NCE_Y,
        }

    def _compute_D_loss(self, real_B: torch.Tensor, fake_B: torch.Tensor) -> torch.Tensor:
        pred_fake = self.netD(fake_B.detach())
        loss_D_fake = self.criterion_GAN(pred_fake, False).mean()
        pred_real = self.netD(real_B)
        loss_D_real = self.criterion_GAN(pred_real, True).mean()
        return (loss_D_fake + loss_D_real) * 0.5

    def _compute_G_loss(
        self,
        real_A: torch.Tensor,
        fake_B: torch.Tensor,
        real_B: torch.Tensor,
    ) -> tuple:
        cfg = self.config
        lambda_GAN = cfg.lambda_GAN
        lambda_NCE = cfg.lambda_NCE
        nce_idt = cfg.nce_idt

        # GAN loss
        if lambda_GAN > 0:
            pred_fake = self.netD(fake_B)
            loss_G_GAN = self.criterion_GAN(pred_fake, True).mean() * lambda_GAN
        else:
            loss_G_GAN = torch.tensor(0.0, device=real_A.device)

        # NCE loss
        if lambda_NCE > 0:
            loss_NCE = _calculate_nce_loss(
                self.netG, self.netF, self.nce_criteria,
                real_A, fake_B, self.nce_layers, lambda_NCE, cfg.num_patches,
                source_channels=cfg.input_nc, target_channels=cfg.output_nc,
            )
        else:
            loss_NCE = torch.tensor(0.0, device=real_A.device)

        # Identity NCE
        loss_NCE_Y = torch.tensor(0.0, device=real_A.device)
        if nce_idt and lambda_NCE > 0:
            real_B_for_G = real_B
            if real_B.shape[1] != cfg.input_nc:
                if cfg.output_nc < cfg.input_nc:
                    real_B_for_G = real_B.repeat(1, cfg.input_nc // cfg.output_nc, 1, 1)
                else:
                    real_B_for_G = real_B.mean(dim=1, keepdim=True)
            idt_B = self.netG(real_B_for_G)
            loss_NCE_Y = _calculate_nce_loss(
                self.netG, self.netF, self.nce_criteria,
                real_B_for_G, idt_B, self.nce_layers, lambda_NCE, cfg.num_patches,
                source_channels=cfg.input_nc, target_channels=cfg.output_nc,
            )
            loss_NCE_both = (loss_NCE + loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = loss_NCE

        loss_G = loss_G_GAN + loss_NCE_both
        return loss_G, loss_G_GAN, loss_NCE, loss_NCE_Y

    def train(
        self,
        root_a: str | Path,
        root_b: str | Path,
    ) -> None:
        """Run full CUT training loop."""
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
            self.netG.train()
            self.netD.train()
            self.netF.train()
            pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{cfg.epochs}")

            for batch in pbar:
                real_A = batch["A"]
                real_B = batch["B"]
                logs = self.train_step(real_A, real_B)
                global_step += 1

                if global_step % cfg.log_every == 0:
                    pbar.set_postfix(logs)
                    logger.info(
                        "step %d | loss_D=%.4f loss_G=%.4f loss_NCE=%.4f",
                        global_step, logs["loss_D"], logs["loss_G"], logs["loss_NCE"],
                    )

            if (epoch + 1) % cfg.save_every == 0:
                self.save_checkpoint(cfg.save_dir, epoch + 1, global_step=global_step)

        logger.info("CUT training complete. Checkpoints saved to %s", cfg.save_dir)

    def save_checkpoint(
        self,
        save_dir: str,
        epoch_or_step: int,
        *,
        global_step: int | None = None,
        scaler: torch.amp.GradScaler | None = None,
    ) -> None:
        """Save generator, discriminator, feature network, and optimizer states for resume.

        Layout: generator/, discriminator/, feature_network/ (HF style) +
        training_state.pt (optimizer_G, optimizer_D, optimizer_F, epoch, global_step).
        """
        path = Path(save_dir) / f"checkpoint-epoch-{epoch_or_step}"
        path.mkdir(parents=True, exist_ok=True)

        # Diffusers-style layout for from_pretrained
        gen_path = path / "generator"
        gen_path.mkdir(exist_ok=True)
        self.netG.save_pretrained(gen_path)

        from safetensors.torch import save_file
        disc_path = path / "discriminator"
        disc_path.mkdir(exist_ok=True)
        save_file(
            self.netD.state_dict(),
            disc_path / "diffusion_pytorch_model.safetensors",
        )
        feat_path = path / "feature_network"
        feat_path.mkdir(exist_ok=True)
        save_file(
            self.netF.state_dict(),
            feat_path / "diffusion_pytorch_model.safetensors",
        )

        # Training state for resume (optimizer states, epoch, global_step)
        step = global_step if global_step is not None else epoch_or_step
        training_state = {
            "optimizer_G": self.optimizer_G.state_dict(),
            "optimizer_D": self.optimizer_D.state_dict(),
            "epoch": epoch_or_step,
            "global_step": step,
        }
        if self.optimizer_F is not None:
            training_state["optimizer_F"] = self.optimizer_F.state_dict()
        if scaler is not None:
            training_state["scaler"] = scaler.state_dict()
        torch.save(training_state, path / "training_state.pt")

        save_config_yaml(
            self.config,
            path / "config.yaml",
            extra={"epoch": epoch_or_step, "global_step": step},
        )
        logger.info("Saved checkpoint to %s (with optimizer states)", path)

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        dummy_batch: dict[str, torch.Tensor] | None = None,
    ) -> dict:
        """Load checkpoint and restore model + optimizer states for resume.

        Returns a dict with keys: "epoch", "global_step" for the caller to restore
        training loop. If dummy_batch is None, creates one (needed to init netF's
        lazy MLP before loading optimizer_F).
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        # Resolve "latest" to output_dir/latest
        if path.name == "latest" and path.is_dir():
            pass
        elif path.is_dir() and (path / "generator").exists():
            pass
        else:
            raise FileNotFoundError(f"Invalid checkpoint dir (expected generator/): {path}")

        from safetensors.torch import load_file

        # Load models
        gen_path = path / "generator"
        if gen_path.exists():
            self.netG = self.netG.from_pretrained(gen_path).to(self.device)
        disc_path = path / "discriminator" / "diffusion_pytorch_model.safetensors"
        if disc_path.exists():
            self.netD.load_state_dict(
                load_file(str(disc_path), device=str(self.device)),
                strict=True,
            )
        feat_path = path / "feature_network" / "diffusion_pytorch_model.safetensors"
        if feat_path.exists():
            # netF has lazy MLP: run one forward to create it before load_state_dict
            if not self.netF.mlp_init:
                cfg = self.config
                if dummy_batch is not None:
                    real_A = dummy_batch.get("A", dummy_batch.get("source"))
                    real_B = dummy_batch.get("B", dummy_batch.get("target"))
                else:
                    res = cfg.resolution
                    real_A = torch.randn(1, cfg.input_nc, res, res, device=self.device) * 2 - 1
                    real_B = torch.randn(1, cfg.output_nc, res, res, device=self.device) * 2 - 1
                with torch.no_grad():
                    fake_B = self.netG(real_A)
                    fake_B_for_enc = _adapt_for_encoder(
                        fake_B, cfg.output_nc, cfg.input_nc
                    )
                    feat_init = self.netG(fake_B_for_enc, self.nce_layers, encode_only=True)
                    self.netF(feat_init, cfg.num_patches, None)
                self._ensure_optimizer_F()
            self.netF.load_state_dict(
                load_file(str(feat_path), device=str(self.device)),
                strict=True,
            )

        # Load optimizer states
        train_state_path = path / "training_state.pt"
        if train_state_path.exists():
            ckpt = torch.load(train_state_path, map_location=self.device, weights_only=False)
            self.optimizer_G.load_state_dict(ckpt["optimizer_G"])
            self.optimizer_D.load_state_dict(ckpt["optimizer_D"])
            if "optimizer_F" in ckpt and self.optimizer_F is not None:
                self.optimizer_F.load_state_dict(ckpt["optimizer_F"])
            logger.info(
                "Loaded checkpoint from %s (epoch=%s, global_step=%s)",
                path, ckpt.get("epoch"), ckpt.get("global_step"),
            )
            return {
                "epoch": ckpt.get("epoch", 0),
                "global_step": ckpt.get("global_step", 0),
            }

        logger.info("Loaded checkpoint from %s (no training_state.pt, optimizers reset)", path)
        # Infer epoch from path (e.g. checkpoint-epoch-50 -> 50)
        try:
            if path.name.startswith("checkpoint-epoch-"):
                n = int(path.name.replace("checkpoint-epoch-", ""))
                return {"epoch": n, "global_step": n}
        except ValueError:
            pass
        return {"epoch": 0, "global_step": 0}
