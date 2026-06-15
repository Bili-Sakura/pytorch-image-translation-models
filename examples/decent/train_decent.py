# Copyright (c) 2026 EarthBridge Team.
# Credits: Decent from Xie et al. NeurIPS 2022 — https://github.com/Mid-Push/Decent

"""Decent trainer for unpaired image-to-image translation."""

from __future__ import annotations

import logging
import os
from itertools import chain
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.data.datasets import UnpairedImageDataset
from src.models.cut import GANLoss, create_discriminator, create_generator
from src.models.decent import (
    FlowConfig,
    FlowAdam,
    PatchDensityEstimator,
    compute_density_changing_loss,
    compute_flow_nll_loss,
)
from src.training import create_optimizer
from src.utils.config_yaml import save_config_yaml

from .config import DecentConfig

logger = logging.getLogger(__name__)


def _build_transform(resolution: int):
    return transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
    ])


class DecentTrainer:
    """Training harness for Decent unpaired translation (A→B).

    Uses a CUT ResNet generator with per-domain flow density estimators
    and density-changing regularization instead of PatchNCE.
    """

    def __init__(self, config: DecentConfig) -> None:
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.var_layers = [int(i) for i in config.var_layers.split(",")]

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

        flow_cfg = FlowConfig(
            flow_type=config.flow_type,
            flow_blocks=config.flow_blocks,
            bnaf_layers=config.bnaf_layers,
            bnaf_dim=config.bnaf_dim,
            maf_dim=config.maf_dim,
            maf_layers=config.maf_layers,
            maf_comps=config.maf_comps,
        )
        self.netF_A = PatchDensityEstimator(flow_cfg).to(self.device)
        self.netF_B = PatchDensityEstimator(flow_cfg).to(self.device)

        self.criterion_GAN = GANLoss(config.gan_mode).to(self.device)
        self.criterion_idt = nn.L1Loss().to(self.device)

        betas = (config.beta1, config.beta2)
        weight_decay = (
            config.weight_decay
            if config.optimizer.lower() in ("adamw", "prodigy", "muon")
            else 0.0
        )

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
        self.optimizer_F: FlowAdam | None = None
        self._flows_initialized = False
        self.current_epoch = 0

    def build_dataset(self, root_a: str | Path, root_b: str | Path) -> UnpairedImageDataset:
        transform = _build_transform(self.config.resolution)
        return UnpairedImageDataset(
            root_a=root_a,
            root_b=root_b,
            transform_a=transform,
            transform_b=transform,
        )

    def _forward(self, real_A: torch.Tensor, real_B: torch.Tensor):
        fake_B, patches_A = self.netG(real_A, layers=self.var_layers)
        idt_B, patches_B = self.netG(real_B, layers=self.var_layers)
        return fake_B, idt_B, patches_A, patches_B

    def _ensure_flow_optimizer(self) -> None:
        if self.optimizer_F is None and self._flows_initialized:
            self.optimizer_F = FlowAdam(
                chain(self.netF_A.parameters(), self.netF_B.parameters()),
                lr=self.config.flow_lr,
                amsgrad=True,
                polyak=self.config.flow_ema,
            )

    def _initialize_flows(self, real_A: torch.Tensor, real_B: torch.Tensor) -> None:
        if self._flows_initialized:
            return
        real_A = real_A.to(self.device) * 2 - 1
        real_B = real_B.to(self.device) * 2 - 1
        with torch.no_grad():
            _, _, patches_A, patches_B = self._forward(real_A, real_B)
            self.netF_A(patches_A, self.config.num_patches, None, detach=True)
            self.netF_B(patches_B, self.config.num_patches, None, detach=True)
        self._flows_initialized = True
        self._ensure_flow_optimizer()
        if self.optimizer_F is not None:
            self.optimizer_F.zero_grad()
        loss_F = self._compute_F_loss(patches_A, patches_B)
        loss_F.backward()
        if self.optimizer_F is not None:
            torch.nn.utils.clip_grad_norm_(self.netF_A.parameters(), max_norm=0.1)
            torch.nn.utils.clip_grad_norm_(self.netF_B.parameters(), max_norm=0.1)
            self.optimizer_F.step()
            self.optimizer_F.zero_grad()

    def _compute_D_loss(self, real_B: torch.Tensor, fake_B: torch.Tensor) -> torch.Tensor:
        pred_fake = self.netD(fake_B.detach())
        loss_D_fake = self.criterion_GAN(pred_fake, False).mean()
        pred_real = self.netD(real_B)
        loss_D_real = self.criterion_GAN(pred_real, True).mean()
        return (loss_D_fake + loss_D_real) * 0.5

    def _compute_F_loss(self, patches_A, patches_B) -> torch.Tensor:
        log_probs_A, _, _ = self.netF_A(
            patches_A, self.config.num_patches, None, detach=True
        )
        log_probs_B, _, _ = self.netF_B(
            patches_B, self.config.num_patches, None, detach=True
        )
        return compute_flow_nll_loss(log_probs_A, log_probs_B)

    def _compute_G_loss(
        self,
        real_B: torch.Tensor,
        fake_B: torch.Tensor,
        idt_B: torch.Tensor,
        patches_A,
        patches_B,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.config
        loss_G_GAN = self.criterion_GAN(self.netD(fake_B), True).mean() * cfg.lambda_GAN
        loss_idt = self.criterion_idt(idt_B, real_B) * cfg.lambda_idt
        loss_G = loss_G_GAN + loss_idt

        loss_var = torch.tensor(0.0, device=real_B.device)
        if cfg.lambda_var > 0:
            with torch.no_grad():
                log_probs_A, feat_lens, sample_ids = self.netF_A(
                    patches_A, cfg.num_patches, None, detach=True
                )
            patches_fake_B = self.netG(fake_B, self.var_layers, encode_only=True)
            log_probs_fake_B, _, _ = self.netF_B(
                patches_fake_B, cfg.num_patches, sample_ids
            )
            loss_var = compute_density_changing_loss(
                log_probs_A,
                log_probs_fake_B,
                feat_lens,
                batch_size=cfg.batch_size,
                num_patches=cfg.num_patches,
                var_all=cfg.var_all,
            )
            loss_G = loss_G + cfg.lambda_var * loss_var

        return loss_G, loss_G_GAN, loss_idt, loss_var

    def train_step(self, real_A: torch.Tensor, real_B: torch.Tensor) -> dict:
        cfg = self.config
        real_A = real_A.to(self.device) * 2 - 1
        real_B = real_B.to(self.device) * 2 - 1

        if not self._flows_initialized:
            self._initialize_flows(real_A, real_B)

        fake_B, idt_B, patches_A, patches_B = self._forward(real_A, real_B)

        for net in (self.netG, self.netF_A, self.netF_B):
            net.requires_grad_(False)
        self.netF_A.requires_grad_(True)
        self.netF_B.requires_grad_(True)
        if self.optimizer_F is not None:
            self.optimizer_F.zero_grad()
        loss_F = self._compute_F_loss(patches_A, patches_B)
        loss_F.backward()
        torch.nn.utils.clip_grad_norm_(self.netF_A.parameters(), max_norm=0.1)
        torch.nn.utils.clip_grad_norm_(self.netF_B.parameters(), max_norm=0.1)
        if self.optimizer_F is not None:
            self.optimizer_F.step()

        self.netD.requires_grad_(True)
        self.optimizer_D.zero_grad()
        loss_D = self._compute_D_loss(real_B, fake_B)
        loss_D.backward()
        self.optimizer_D.step()

        if self.optimizer_F is not None:
            self.optimizer_F.swap()

        self.netD.requires_grad_(False)
        self.netG.requires_grad_(True)
        self.netF_A.requires_grad_(False)
        self.netF_B.requires_grad_(False)
        self.optimizer_G.zero_grad()
        loss_G, loss_G_GAN, loss_idt, loss_var = self._compute_G_loss(
            real_B, fake_B, idt_B, patches_A, patches_B
        )
        loss_G.backward()
        self.optimizer_G.step()

        if self.optimizer_F is not None:
            self.optimizer_F.swap()

        return {
            "loss_D": loss_D.item(),
            "loss_F": loss_F.item(),
            "loss_G": loss_G.item(),
            "loss_G_GAN": loss_G_GAN.item(),
            "loss_idt": loss_idt.item(),
            "loss_var": loss_var.item() if torch.is_tensor(loss_var) else loss_var,
        }

    def train(self, root_a: str | Path, root_b: str | Path) -> None:
        cfg = self.config
        save_dir = cfg.output_dir or cfg.save_dir
        os.makedirs(save_dir, exist_ok=True)

        dataset = self.build_dataset(root_a, root_b)
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.dataloader_num_workers,
            drop_last=True,
        )

        global_step = 0
        for epoch in range(cfg.epochs):
            self.current_epoch = epoch
            self.netG.train()
            self.netD.train()
            self.netF_A.train()
            self.netF_B.train()
            pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{cfg.epochs}")

            for batch in pbar:
                logs = self.train_step(batch["A"], batch["B"])
                global_step += 1

                if global_step % cfg.log_every == 0:
                    pbar.set_postfix(logs)
                    logger.info(
                        "step %d | loss_D=%.4f loss_F=%.4f loss_G=%.4f loss_var=%.4f",
                        global_step,
                        logs["loss_D"],
                        logs["loss_F"],
                        logs["loss_G"],
                        logs["loss_var"],
                    )

            if (epoch + 1) % cfg.save_every == 0:
                self.save_checkpoint(save_dir, epoch + 1, global_step=global_step)

        logger.info("Decent training complete. Checkpoints saved to %s", save_dir)

    def save_checkpoint(
        self,
        save_dir: str,
        epoch_or_step: int,
        *,
        global_step: int | None = None,
        scaler: torch.amp.GradScaler | None = None,
    ) -> None:
        path = Path(save_dir) / f"checkpoint-epoch-{epoch_or_step}"
        path.mkdir(parents=True, exist_ok=True)

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

        flow_a_path = path / "flow_A"
        flow_a_path.mkdir(exist_ok=True)
        save_file(
            self.netF_A.state_dict(),
            flow_a_path / "diffusion_pytorch_model.safetensors",
        )
        flow_b_path = path / "flow_B"
        flow_b_path.mkdir(exist_ok=True)
        save_file(
            self.netF_B.state_dict(),
            flow_b_path / "diffusion_pytorch_model.safetensors",
        )

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
        logger.info("Saved checkpoint to %s", path)
