# Copyright (c) 2026 EarthBridge Team.
# Credits: UNSB from Kim et al. "Unpaired Image-to-Image Translation via Neural Schrödinger Bridge" ICLR 2024.

"""UNSB trainer for unpaired image-to-image translation."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.data.datasets import UnpairedImageDataset
from src.training import create_optimizer
from src.utils.config_yaml import save_config_yaml
from src.models.unsb import (
    create_generator,
    create_discriminator,
    create_energy_net,
    create_patch_sample_mlp,
    GANLoss,
    PatchNCELoss,
)
from src.schedulers.unsb import UNSBScheduler

from .config import UNSBConfig

logger = logging.getLogger(__name__)


def _build_transform(resolution: int):
    """Build transform: resize, to tensor."""
    return transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
    ])


class UNSBTrainer:
    """Training harness for UNSB (Unpaired Neural Schrödinger Bridge).

    Uses :class:`UnpairedImageDataset` for unpaired domain A and B images.
    The method employs four networks (G, D, E, F) with a combined GAN +
    Schrödinger Bridge + NCE contrastive loss.
    """

    def __init__(self, config: UNSBConfig) -> None:
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        # Models
        self.netG = create_generator(
            input_nc=config.input_nc,
            output_nc=config.output_nc,
            ngf=config.ngf,
            n_blocks=config.n_blocks,
            n_mlp=config.n_mlp,
            norm_type=config.normG,
            use_dropout=config.use_dropout,
            no_antialias=config.no_antialias,
            no_antialias_up=config.no_antialias_up,
            init_type=config.init_type,
            init_gain=config.init_gain,
        ).to(self.device)

        self.netD = create_discriminator(
            input_nc=config.output_nc,
            ndf=config.ndf,
            n_layers=config.n_layers_D,
            norm_type=config.normD,
            init_type=config.init_type,
            init_gain=config.init_gain,
        ).to(self.device)

        self.netE = create_energy_net(
            input_nc=config.output_nc,
            ndf=config.ndf,
            n_layers=config.n_layers_D,
            norm_type=config.normD,
            init_type=config.init_type,
            init_gain=config.init_gain,
        ).to(self.device)

        self.netF = create_patch_sample_mlp(
            use_mlp=(config.netF == "mlp_sample"),
            nc=config.netF_nc,
            init_type=config.init_type,
            init_gain=config.init_gain,
        ).to(self.device)

        # Scheduler (for time schedule computation)
        self.scheduler = UNSBScheduler(
            num_timesteps=config.num_timesteps,
            tau=config.tau,
        )

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
        self.optimizer_E = create_optimizer(
            self.netE.parameters(),
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

    def train_step(
        self,
        real_A: torch.Tensor,
        real_B: torch.Tensor,
        real_A2: torch.Tensor | None = None,
    ) -> dict:
        """Single training step.

        Parameters
        ----------
        real_A : Tensor (B, C, H, W)
            Source domain images in [0, 1].
        real_B : Tensor (B, C, H, W)
            Target domain images in [0, 1].
        real_A2 : Tensor (B, C, H, W), optional
            Second source sample for SB contrastive estimation.
            If None, uses real_A with different noise.

        Returns
        -------
        dict
            Loss values for logging.
        """
        cfg = self.config
        real_A = real_A.to(self.device) * 2 - 1
        real_B = real_B.to(self.device) * 2 - 1
        if real_A2 is not None:
            real_A2 = real_A2.to(self.device) * 2 - 1
        else:
            real_A2 = real_A.clone()

        tau = cfg.tau
        T = cfg.num_timesteps
        times = self.scheduler.times.to(self.device)
        bs = real_A.size(0)
        ngf = cfg.ngf

        # Sample random timestep
        time_idx = torch.randint(T, size=[1], device=self.device).long()

        # Forward pass: propagate source through bridge up to time_idx
        with torch.no_grad():
            self.netG.eval()
            Xt = real_A
            Xt2 = real_A2
            XtB = real_B if cfg.nce_idt else None

            for t in range(time_idx.item() + 1):
                if t > 0:
                    delta = times[t] - times[t - 1]
                    denom = times[-1] - times[t - 1]
                    inter = (delta / denom).reshape(-1, 1, 1, 1)
                    scale = (delta * (1 - delta / denom)).reshape(-1, 1, 1, 1)

                Xt = real_A if t == 0 else (1 - inter) * Xt + inter * Xt_1.detach() + (scale * tau).sqrt() * torch.randn_like(Xt)
                t_idx = (t * torch.ones(bs, device=self.device)).long()
                z = torch.randn(bs, 4 * ngf, device=self.device)
                Xt_1 = self.netG(Xt, t_idx, z)

                Xt2 = real_A2 if t == 0 else (1 - inter) * Xt2 + inter * Xt_12.detach() + (scale * tau).sqrt() * torch.randn_like(Xt2)
                z2 = torch.randn(bs, 4 * ngf, device=self.device)
                Xt_12 = self.netG(Xt2, t_idx, z2)

                if cfg.nce_idt and XtB is not None:
                    XtB = real_B if t == 0 else (1 - inter) * XtB + inter * Xt_1B.detach() + (scale * tau).sqrt() * torch.randn_like(XtB)
                    zB = torch.randn(bs, 4 * ngf, device=self.device)
                    Xt_1B = self.netG(XtB, t_idx, zB)

        real_A_noisy = Xt.detach()
        real_A_noisy2 = Xt2.detach()

        # Generator forward
        self.netG.train()
        self.netD.train()
        self.netE.train()
        self.netF.train()

        z_in = torch.randn(bs, 4 * ngf, device=self.device)
        z_in2 = torch.randn(bs, 4 * ngf, device=self.device)
        fake_B = self.netG(real_A_noisy, time_idx, z_in)
        fake_B2 = self.netG(real_A_noisy2, time_idx, z_in2)

        # Identity pass for NCE
        idt_B = None
        if cfg.nce_idt:
            z_idt = torch.randn(bs, 4 * ngf, device=self.device)
            idt_B = self.netG(XtB.detach(), time_idx, z_idt)

        # Initialize netF on first step (lazy MLP creation)
        if not self.netF.mlp_init:
            with torch.no_grad():
                z_init = torch.randn(bs, 4 * ngf, device=self.device)
                feat_init = self.netG(fake_B.detach(), time_idx * 0, z_init, self.nce_layers, encode_only=True)
                self.netF(feat_init, cfg.num_patches, None)
            self._ensure_optimizer_F()

        # Update D
        self.optimizer_D.zero_grad()
        loss_D = self._compute_D_loss(real_B, fake_B, time_idx)
        loss_D.backward()
        self.optimizer_D.step()

        # Update E
        self.optimizer_E.zero_grad()
        loss_E = self._compute_E_loss(real_A_noisy, real_A_noisy2, fake_B, fake_B2, time_idx)
        loss_E.backward()
        self.optimizer_E.step()

        # Update G + F
        self.optimizer_G.zero_grad()
        if self.optimizer_F is not None:
            self.optimizer_F.zero_grad()
        loss_G, loss_G_GAN, loss_SB, loss_NCE, loss_NCE_Y = self._compute_G_loss(
            real_A, real_B, real_A_noisy, real_A_noisy2, fake_B, fake_B2, idt_B, time_idx,
        )
        loss_G.backward()
        self.optimizer_G.step()
        if self.optimizer_F is not None:
            self.optimizer_F.step()

        return {
            "loss_D": loss_D.item(),
            "loss_E": loss_E.item(),
            "loss_G": loss_G.item(),
            "loss_G_GAN": loss_G_GAN.item() if torch.is_tensor(loss_G_GAN) else loss_G_GAN,
            "loss_SB": loss_SB.item() if torch.is_tensor(loss_SB) else loss_SB,
            "loss_NCE": loss_NCE.item() if torch.is_tensor(loss_NCE) else loss_NCE,
            "loss_NCE_Y": loss_NCE_Y.item() if torch.is_tensor(loss_NCE_Y) else loss_NCE_Y,
        }

    def _compute_D_loss(
        self, real_B: torch.Tensor, fake_B: torch.Tensor, time_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Compute discriminator loss."""
        pred_fake = self.netD(fake_B.detach(), time_idx)
        loss_D_fake = self.criterion_GAN(pred_fake, False).mean()
        pred_real = self.netD(real_B, time_idx)
        loss_D_real = self.criterion_GAN(pred_real, True).mean()
        return (loss_D_fake + loss_D_real) * 0.5

    def _compute_E_loss(
        self,
        real_A_noisy: torch.Tensor,
        real_A_noisy2: torch.Tensor,
        fake_B: torch.Tensor,
        fake_B2: torch.Tensor,
        time_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Compute energy network loss for SB contrastive estimation."""
        XtXt_1 = torch.cat([real_A_noisy, fake_B.detach()], dim=1)
        XtXt_2 = torch.cat([real_A_noisy2, fake_B2.detach()], dim=1)
        temp = torch.logsumexp(self.netE(XtXt_1, time_idx, XtXt_2).reshape(-1), dim=0).mean()
        loss_E = -self.netE(XtXt_1, time_idx, XtXt_1).mean() + temp + temp ** 2
        return loss_E

    def _compute_G_loss(
        self,
        real_A: torch.Tensor,
        real_B: torch.Tensor,
        real_A_noisy: torch.Tensor,
        real_A_noisy2: torch.Tensor,
        fake_B: torch.Tensor,
        fake_B2: torch.Tensor,
        idt_B: torch.Tensor | None,
        time_idx: torch.Tensor,
    ) -> tuple:
        """Compute generator loss: GAN + SB + NCE."""
        cfg = self.config
        device = real_A.device

        # GAN loss
        if cfg.lambda_GAN > 0:
            pred_fake = self.netD(fake_B, time_idx)
            loss_G_GAN = self.criterion_GAN(pred_fake, True).mean() * cfg.lambda_GAN
        else:
            loss_G_GAN = torch.tensor(0.0, device=device)

        # SB loss
        loss_SB = torch.tensor(0.0, device=device)
        if cfg.lambda_SB > 0:
            XtXt_1 = torch.cat([real_A_noisy, fake_B], dim=1)
            XtXt_2 = torch.cat([real_A_noisy2, fake_B2], dim=1)
            ET_XY = (
                self.netE(XtXt_1, time_idx, XtXt_1).mean()
                - torch.logsumexp(self.netE(XtXt_1, time_idx, XtXt_2).reshape(-1), dim=0)
            )
            loss_SB = -(cfg.num_timesteps - time_idx[0]) / cfg.num_timesteps * cfg.tau * ET_XY
            loss_SB = loss_SB + cfg.tau * torch.mean((real_A_noisy - fake_B) ** 2)

        # NCE loss
        ngf = cfg.ngf
        if cfg.lambda_NCE > 0:
            z_nce = torch.randn(real_A.size(0), 4 * ngf, device=device)
            loss_NCE = self._calculate_nce_loss(real_A, fake_B, time_idx, z_nce)
        else:
            loss_NCE = torch.tensor(0.0, device=device)

        # Identity NCE
        loss_NCE_Y = torch.tensor(0.0, device=device)
        if cfg.nce_idt and cfg.lambda_NCE > 0 and idt_B is not None:
            z_idt = torch.randn(real_B.size(0), 4 * ngf, device=device)
            loss_NCE_Y = self._calculate_nce_loss(real_B, idt_B, time_idx, z_idt)
            loss_NCE_both = (loss_NCE + loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = loss_NCE

        loss_G = loss_G_GAN + cfg.lambda_SB * loss_SB + cfg.lambda_NCE * loss_NCE_both
        return loss_G, loss_G_GAN, loss_SB, loss_NCE, loss_NCE_Y

    def _calculate_nce_loss(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        time_idx: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Compute PatchNCE loss across multiple encoder layers."""
        n_layers = len(self.nce_layers)
        feat_q = self.netG(tgt, time_idx * 0, z, self.nce_layers, encode_only=True)
        feat_k = self.netG(src, time_idx * 0, z, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.config.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.config.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit in zip(feat_q_pool, feat_k_pool, self.nce_criteria):
            loss = crit(f_q, f_k) * self.config.lambda_NCE
            total_nce_loss += loss.mean()
        return total_nce_loss / n_layers

    def train(
        self,
        root_a: str | Path,
        root_b: str | Path,
    ) -> None:
        """Run full UNSB training loop."""
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
            self.netE.train()
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
                        "step %d | loss_D=%.4f loss_G=%.4f loss_SB=%.4f loss_NCE=%.4f",
                        global_step, logs["loss_D"], logs["loss_G"],
                        logs["loss_SB"], logs["loss_NCE"],
                    )

            if (epoch + 1) % cfg.save_every == 0:
                self.save_checkpoint(cfg.save_dir, epoch + 1, global_step=global_step)

        logger.info("UNSB training complete. Checkpoints saved to %s", cfg.save_dir)

    def save_checkpoint(
        self,
        save_dir: str,
        epoch: int,
        *,
        global_step: int | None = None,
        scaler: torch.amp.GradScaler | None = None,
    ) -> None:
        """Save all networks and optimizer states for resume."""
        path = Path(save_dir) / f"checkpoint-epoch-{epoch}"
        path.mkdir(parents=True, exist_ok=True)

        from safetensors.torch import save_file

        for name, net in [("generator", self.netG), ("discriminator", self.netD),
                          ("energy_net", self.netE), ("feature_network", self.netF)]:
            net_path = path / name
            net_path.mkdir(exist_ok=True)
            save_file(net.state_dict(), net_path / "model.safetensors")

        # Training state for resume (optimizer states, epoch, global_step)
        training_state = {
            "optimizer_G": self.optimizer_G.state_dict(),
            "optimizer_D": self.optimizer_D.state_dict(),
            "optimizer_E": self.optimizer_E.state_dict(),
            "epoch": epoch,
            "global_step": global_step if global_step is not None else epoch,
        }
        if self.optimizer_F is not None:
            training_state["optimizer_F"] = self.optimizer_F.state_dict()
        if scaler is not None:
            training_state["scaler"] = scaler.state_dict()
        torch.save(training_state, path / "training_state.pt")

        from src.utils.config_yaml import save_config_yaml
        save_config_yaml(
            self.config,
            path / "config.yaml",
            extra={"epoch": epoch, "global_step": global_step if global_step is not None else epoch},
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
        if not (path / "generator").exists():
            raise FileNotFoundError(f"Invalid checkpoint dir (expected generator/): {path}")

        from safetensors.torch import load_file

        for name, net in [("generator", self.netG), ("discriminator", self.netD),
                          ("energy_net", self.netE), ("feature_network", self.netF)]:
            net_path = path / name / "model.safetensors"
            if net_path.exists():
                net.load_state_dict(
                    load_file(str(net_path), device=str(self.device)),
                    strict=True,
                )

        # netF has lazy MLP: run one forward to create it before load_state_dict
        if not self.netF.mlp_init and (path / "feature_network" / "model.safetensors").exists():
            cfg = self.config
            if dummy_batch is not None:
                real_A = dummy_batch.get("A", dummy_batch.get("source"))
                real_B = dummy_batch.get("B", dummy_batch.get("target"))
            else:
                res = cfg.resolution
                real_A = torch.randn(1, cfg.input_nc, res, res, device=self.device) * 2 - 1
                real_B = torch.randn(1, cfg.output_nc, res, res, device=self.device) * 2 - 1
            with torch.no_grad():
                # Minimal forward to init netF's MLP (UNSB uses time_idx and z)
                t = torch.zeros(1, dtype=torch.long, device=self.device)
                z = self.scheduler.sample_t(1).to(self.device)
                _ = self.netG(real_A, t, z, self.nce_layers, encode_only=True)
                self.netF(_, cfg.num_patches, None)
            self._ensure_optimizer_F()

        # Load optimizer states
        train_state_path = path / "training_state.pt"
        if train_state_path.exists():
            ckpt = torch.load(train_state_path, map_location=self.device, weights_only=False)
            self.optimizer_G.load_state_dict(ckpt["optimizer_G"])
            self.optimizer_D.load_state_dict(ckpt["optimizer_D"])
            self.optimizer_E.load_state_dict(ckpt["optimizer_E"])
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
        try:
            if path.name.startswith("checkpoint-epoch-"):
                n = int(path.name.replace("checkpoint-epoch-", ""))
                return {"epoch": n, "global_step": n}
        except ValueError:
            pass
        return {"epoch": 0, "global_step": 0}
