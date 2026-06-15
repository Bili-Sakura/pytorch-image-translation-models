# Copyright (c) 2026 EarthBridge Team.
# Credits: F-LSeSim from Zheng et al. CVPR 2021 —
# https://github.com/lyndonzheng/F-LSeSim

"""F-LSeSim trainer for unpaired image-to-image translation."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.data.datasets import UnpairedImageDataset
from src.models.cut import GANLoss, create_discriminator, create_generator
from src.models.flsesim import (
    ImageNetNormalization,
    ImagePool,
    SpatialCorrelativeLoss,
    VGG16FeatureExtractor,
    compute_spatial_correlative_loss,
)
from src.training import create_optimizer
from src.utils.config_yaml import save_config_yaml

from .config import FLSeSimConfig

logger = logging.getLogger(__name__)


def _build_transform(resolution: int):
    return transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
    ])


def _to_imagenet_range(x: torch.Tensor) -> torch.Tensor:
    """Map [-1, 1] tensors to [0, 1] for VGG preprocessing."""
    return (x + 1) * 0.5


class FLSeSimTrainer:
    """Training harness for F-LSeSim unpaired translation.

    Uses a ResNet generator with PatchGAN discriminator and
    spatially-correlative loss computed on VGG16 features.
    """

    def __init__(self, config: FLSeSimConfig) -> None:
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

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

        self.netPre = VGG16FeatureExtractor().to(self.device)
        self.normalization = ImageNetNormalization().to(self.device)
        self.criterion_spatial = SpatialCorrelativeLoss(
            loss_mode=config.loss_mode,
            patch_nums=config.patch_nums,
            patch_size=config.patch_size,
            use_norm=config.use_norm,
            use_conv=config.learned_attn,
            init_type=config.init_type,
            init_gain=config.init_gain,
            temperature=config.temperature,
        ).to(self.device)
        self.criterion_GAN = GANLoss(config.gan_mode).to(self.device)
        self.criterion_idt = torch.nn.L1Loss().to(self.device)

        self.attn_layers = [int(i) for i in config.attn_layers.split(",")]
        self.fake_B_pool = ImagePool(config.pool_size)
        self.optimizer_F = None
        self.current_epoch = 0

        betas = (config.beta1, config.beta2)
        weight_decay = (
            config.weight_decay if config.optimizer.lower() in ("adamw", "prodigy", "muon") else 0.0
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

        if config.learned_attn:
            self._ensure_optimizer_F()

        if not config.learned_attn:
            for param in self.netPre.parameters():
                param.requires_grad = False

    def _ensure_optimizer_F(self) -> None:
        if self.optimizer_F is not None:
            return
        cfg = self.config
        weight_decay = (
            cfg.weight_decay if cfg.optimizer.lower() in ("adamw", "prodigy", "muon") else 0.0
        )
        params = list(filter(lambda p: p.requires_grad, self.criterion_spatial.parameters()))
        if cfg.learned_attn:
            params += list(filter(lambda p: p.requires_grad, self.netPre.parameters()))
        self.optimizer_F = create_optimizer(
            params,
            optimizer_type=cfg.optimizer,
            lr=cfg.lr,
            weight_decay=weight_decay,
            betas=(cfg.beta1, cfg.beta2),
            prodigy_d0=cfg.prodigy_d0,
        )

    def build_dataset(self, root_a: str | Path, root_b: str | Path) -> UnpairedImageDataset:
        transform = _build_transform(self.config.resolution)
        return UnpairedImageDataset(
            root_a=root_a,
            root_b=root_b,
            transform_a=transform,
            transform_b=transform,
        )

    def _normalize_batch(self, *images: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(self.normalization(_to_imagenet_range(img)) for img in images)

    def _spatial_loss(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        other: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return compute_spatial_correlative_loss(
            self.netPre,
            self.criterion_spatial,
            src,
            tgt,
            other,
            self.attn_layers,
        )

    def _forward_generator(
        self,
        real_A: torch.Tensor,
        real_B: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        cfg = self.config
        use_idt = cfg.lambda_identity > 0 or cfg.lambda_spatial_idt > 0
        if use_idt:
            assert cfg.input_nc == cfg.output_nc
            combined = torch.cat((real_A, real_B), dim=0)
            output = self.netG(combined)
            fake_B = output[: real_A.size(0)]
            idt_B = output[real_A.size(0) :]
            return fake_B, idt_B
        return self.netG(real_A), None

    def _compute_F_loss(
        self,
        real_A: torch.Tensor,
        fake_B: torch.Tensor,
        real_B: torch.Tensor,
    ) -> torch.Tensor:
        norm_real_A, norm_fake_B, norm_real_B = self._normalize_batch(real_A, fake_B, real_B)
        return self._spatial_loss(norm_real_A, norm_fake_B, norm_real_B)

    def _compute_D_loss(self, real_B: torch.Tensor, fake_B: torch.Tensor) -> torch.Tensor:
        fake_B = self.fake_B_pool.query(fake_B.detach())
        pred_fake = self.netD(fake_B)
        loss_D_fake = self.criterion_GAN(pred_fake, False).mean()
        pred_real = self.netD(real_B)
        loss_D_real = self.criterion_GAN(pred_real, True).mean()
        return (loss_D_fake + loss_D_real) * 0.5

    def _compute_G_loss(
        self,
        real_A: torch.Tensor,
        fake_B: torch.Tensor,
        real_B: torch.Tensor,
        idt_B: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        cfg = self.config
        logs: dict[str, float] = {}

        pred_fake = self.netD(fake_B)
        loss_G_GAN = self.criterion_GAN(pred_fake, True).mean() * cfg.lambda_GAN
        logs["loss_G_GAN"] = loss_G_GAN.item()

        norm_real_A, norm_fake_B, norm_real_B = self._normalize_batch(real_A, fake_B, real_B)
        loss_spatial = (
            self._spatial_loss(norm_real_A, norm_fake_B, None) * cfg.lambda_spatial
            if cfg.lambda_spatial > 0
            else torch.tensor(0.0, device=real_A.device)
        )
        logs["loss_spatial"] = loss_spatial.item() if torch.is_tensor(loss_spatial) else 0.0

        if idt_B is not None and cfg.lambda_spatial_idt > 0:
            _, norm_idt_B = self._normalize_batch(real_B, idt_B)
            loss_spatial_idt = self._spatial_loss(norm_real_B, norm_idt_B, None) * cfg.lambda_spatial_idt
        else:
            loss_spatial_idt = torch.tensor(0.0, device=real_A.device)
        logs["loss_spatial_idt"] = loss_spatial_idt.item()

        if idt_B is not None and cfg.lambda_identity > 0:
            loss_idt = self.criterion_idt(real_B, idt_B) * cfg.lambda_identity
        else:
            loss_idt = torch.tensor(0.0, device=real_A.device)
        logs["loss_idt"] = loss_idt.item()

        loss_G = loss_G_GAN + loss_spatial + loss_spatial_idt + loss_idt
        logs["loss_G"] = loss_G.item()
        return loss_G, logs

    def train_step(self, real_A: torch.Tensor, real_B: torch.Tensor) -> dict:
        cfg = self.config
        real_A = real_A.to(self.device) * 2 - 1
        real_B = real_B.to(self.device) * 2 - 1

        fake_B, idt_B = self._forward_generator(real_A, real_B)

        if cfg.learned_attn:
            self._ensure_optimizer_F()
            assert self.optimizer_F is not None
            for param in self.netPre.parameters():
                param.requires_grad = True
            for param in self.criterion_spatial.parameters():
                param.requires_grad = True
            self.optimizer_F.zero_grad()
            loss_F = self._compute_F_loss(real_A, fake_B.detach(), real_B)
            loss_F.backward()
            self.optimizer_F.step()
            for param in self.netPre.parameters():
                param.requires_grad = False

        self.optimizer_D.zero_grad()
        loss_D = self._compute_D_loss(real_B, fake_B)
        loss_D.backward()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        loss_G, logs = self._compute_G_loss(real_A, fake_B, real_B, idt_B)
        loss_G.backward()
        self.optimizer_G.step()

        logs["loss_D"] = loss_D.item()
        return logs

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
            pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{cfg.epochs}")

            for batch in pbar:
                logs = self.train_step(batch["A"], batch["B"])
                global_step += 1

                if global_step % cfg.log_every == 0:
                    pbar.set_postfix({k: f"{v:.4f}" for k, v in logs.items()})
                    logger.info(
                        "step %d | loss_D=%.4f loss_G=%.4f loss_spatial=%.4f",
                        global_step,
                        logs["loss_D"],
                        logs["loss_G"],
                        logs["loss_spatial"],
                    )

            if (epoch + 1) % cfg.save_every == 0:
                self.save_checkpoint(save_dir, epoch + 1, global_step=global_step)

        logger.info("F-LSeSim training complete. Checkpoints saved to %s", save_dir)

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

        spatial_path = path / "spatial_loss"
        spatial_path.mkdir(exist_ok=True)
        save_file(
            self.criterion_spatial.state_dict(),
            spatial_path / "diffusion_pytorch_model.safetensors",
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
