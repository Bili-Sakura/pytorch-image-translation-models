# Copyright (c) 2026 EarthBridge Team.
# Credits: Hneg-SRC from Jung et al. CVPR 2022 — https://github.com/jcy132/Hneg_SRC

"""Hneg-SRC trainer for unpaired image-to-image translation."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.data.datasets import UnpairedImageDataset
from src.models.cut import (
    GANLoss,
    create_discriminator,
    create_generator,
    create_patch_sample_mlp,
)
from src.models.hneg_src import PatchHDCELoss, SRCLoss
from src.training import create_optimizer
from src.utils.config_yaml import save_config_yaml

from .config import HnegSRCConfig

logger = logging.getLogger(__name__)


def _build_transform(resolution: int):
    return transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
    ])


def _adapt_for_encoder(x: torch.Tensor, actual_ch: int, encoder_ch: int) -> torch.Tensor:
    if actual_ch == encoder_ch:
        return x
    if actual_ch < encoder_ch:
        return x.repeat(1, encoder_ch // actual_ch, 1, 1)
    return x.mean(dim=1, keepdim=True)


def _pool_encoder_features(
    netG,
    netF,
    src: torch.Tensor,
    tgt: torch.Tensor,
    nce_layers: list[int],
    num_patches: int,
    input_nc: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    if src.shape[1] != input_nc:
        src = _adapt_for_encoder(src, src.shape[1], input_nc)
    if tgt.shape[1] != input_nc:
        tgt = _adapt_for_encoder(tgt, tgt.shape[1], input_nc)

    src_feat = netG(src, nce_layers, encode_only=True)
    tgt_feat = netG(tgt, nce_layers, encode_only=True)
    tgt_pool, sample_ids = netF(tgt_feat, num_patches, None)
    src_pool, _ = netF(src_feat, num_patches, sample_ids)
    return src_pool, tgt_pool


class HnegSRCTrainer:
    """Training harness for Hneg-SRC unpaired translation.

    Uses a CUT ResNet generator with SRC (semantic relation) and HDCE
    (hard-negative DCE) contrastive losses instead of standard PatchNCE.
    """

    def __init__(self, config: HnegSRCConfig) -> None:
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

        self.netF = create_patch_sample_mlp(
            use_mlp=(config.netF == "mlp_sample"),
            nc=config.netF_nc,
            init_type=config.init_type,
            init_gain=config.init_gain,
        ).to(self.device)

        self.criterion_GAN = GANLoss(config.gan_mode).to(self.device)
        nce_layers = [int(i) for i in config.nce_layers.split(",")]
        self.nce_layers = nce_layers

        self.src_criteria = [
            SRCLoss(
                num_patches=config.num_patches,
                hdce_gamma=config.hdce_gamma,
                hdce_gamma_min=config.hdce_gamma_min,
                use_curriculum=config.use_curriculum,
                step_gamma=config.step_gamma,
                step_gamma_epoch=config.step_gamma_epoch,
                n_epochs=config.epochs,
                n_epochs_decay=config.n_epochs_decay,
                lambda_src=config.lambda_SRC,
            ).to(self.device)
            for _ in nce_layers
        ]
        self.hdce_criteria = [
            PatchHDCELoss(
                nce_T=config.nce_T,
                batch_size=config.batch_size,
                nce_includes_all_negatives_from_minibatch=config.nce_includes_all_negatives_from_minibatch,
                lambda_hdce=config.lambda_HDCE,
            ).to(self.device)
            for _ in nce_layers
        ]

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
        self.optimizer_F = None
        self.current_epoch = 0

    def build_dataset(self, root_a: str | Path, root_b: str | Path) -> UnpairedImageDataset:
        transform = _build_transform(self.config.resolution)
        return UnpairedImageDataset(
            root_a=root_a,
            root_b=root_b,
            transform_a=transform,
            transform_b=transform,
        )

    def _ensure_optimizer_F(self) -> None:
        if self.optimizer_F is None and self.netF.mlp_init:
            cfg = self.config
            weight_decay = cfg.weight_decay if cfg.optimizer.lower() in ("adamw", "prodigy", "muon") else 0.0
            self.optimizer_F = create_optimizer(
                self.netF.parameters(),
                optimizer_type=cfg.optimizer,
                lr=cfg.lr,
                weight_decay=weight_decay,
                betas=(cfg.beta1, cfg.beta2),
                prodigy_d0=cfg.prodigy_d0,
            )

    def train_step(self, real_A: torch.Tensor, real_B: torch.Tensor) -> dict:
        cfg = self.config
        real_A = real_A.to(self.device) * 2 - 1
        real_B = real_B.to(self.device) * 2 - 1

        fake_B = self.netG(real_A)

        if not self.netF.mlp_init:
            with torch.no_grad():
                fake_B_for_enc = _adapt_for_encoder(fake_B, cfg.output_nc, cfg.input_nc)
                feat_init = self.netG(fake_B_for_enc, self.nce_layers, encode_only=True)
                self.netF(feat_init, cfg.num_patches, None)
            self._ensure_optimizer_F()

        self.optimizer_D.zero_grad()
        loss_D = self._compute_D_loss(real_B, fake_B)
        loss_D.backward()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        if self.optimizer_F is not None:
            self.optimizer_F.zero_grad()

        loss_G, loss_G_GAN, loss_HDCE, loss_HDCE_Y, loss_SRC = self._compute_G_loss(
            real_A, fake_B, real_B, epoch=self.current_epoch
        )
        loss_G.backward()
        self.optimizer_G.step()
        if self.optimizer_F is not None:
            self.optimizer_F.step()

        return {
            "loss_D": loss_D.item(),
            "loss_G": loss_G.item(),
            "loss_G_GAN": loss_G_GAN.item() if torch.is_tensor(loss_G_GAN) else loss_G_GAN,
            "loss_HDCE": loss_HDCE.item() if torch.is_tensor(loss_HDCE) else loss_HDCE,
            "loss_HDCE_Y": loss_HDCE_Y.item() if torch.is_tensor(loss_HDCE_Y) else loss_HDCE_Y,
            "loss_SRC": loss_SRC.item() if torch.is_tensor(loss_SRC) else loss_SRC,
        }

    def _compute_D_loss(self, real_B: torch.Tensor, fake_B: torch.Tensor) -> torch.Tensor:
        pred_fake = self.netD(fake_B.detach())
        loss_D_fake = self.criterion_GAN(pred_fake, False).mean()
        pred_real = self.netD(real_B)
        loss_D_real = self.criterion_GAN(pred_real, True).mean()
        return (loss_D_fake + loss_D_real) * 0.5

    def _compute_src_and_weights(
        self,
        src_pool: list[torch.Tensor],
        tgt_pool: list[torch.Tensor],
        *,
        only_weight: bool = False,
        epoch: int | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        n_layers = len(self.nce_layers)
        total_src = torch.tensor(0.0, device=src_pool[0].device)
        weights: list[torch.Tensor] = []
        for f_src, f_tgt, crit in zip(src_pool, tgt_pool, self.src_criteria):
            loss_src, weight = crit(f_tgt, f_src, only_weight=only_weight, epoch=epoch)
            total_src = total_src + loss_src
            weights.append(weight)
        return total_src / n_layers, weights

    def _compute_hdce_loss(
        self,
        src_pool: list[torch.Tensor],
        tgt_pool: list[torch.Tensor],
        weights: list[torch.Tensor] | None,
    ) -> torch.Tensor:
        cfg = self.config
        n_layers = len(self.nce_layers)
        total = torch.tensor(0.0, device=src_pool[0].device)
        for f_src, f_tgt, crit, weight in zip(src_pool, tgt_pool, self.hdce_criteria, weights or [None] * n_layers):
            w = None if cfg.no_hneg or weight is None else weight
            total = total + crit(f_tgt, f_src, w).mean()
        return total / n_layers

    def _compute_G_loss(
        self,
        real_A: torch.Tensor,
        fake_B: torch.Tensor,
        real_B: torch.Tensor,
        *,
        epoch: int | None = None,
    ) -> tuple:
        cfg = self.config

        if cfg.lambda_GAN > 0:
            pred_fake = self.netD(fake_B)
            loss_G_GAN = self.criterion_GAN(pred_fake, True).mean() * cfg.lambda_GAN
        else:
            loss_G_GAN = torch.tensor(0.0, device=real_A.device)

        real_A_pool, fake_B_pool = _pool_encoder_features(
            self.netG,
            self.netF,
            real_A,
            fake_B,
            self.nce_layers,
            cfg.num_patches,
            cfg.input_nc,
        )

        loss_SRC, weights = self._compute_src_and_weights(
            real_A_pool, fake_B_pool, epoch=epoch
        )

        if cfg.lambda_HDCE > 0:
            loss_HDCE = self._compute_hdce_loss(real_A_pool, fake_B_pool, weights)
        else:
            loss_HDCE = torch.tensor(0.0, device=real_A.device)

        loss_HDCE_Y = torch.tensor(0.0, device=real_A.device)
        if cfg.dce_idt and cfg.lambda_HDCE > 0:
            real_B_for_G = real_B
            if real_B.shape[1] != cfg.input_nc:
                if cfg.output_nc < cfg.input_nc:
                    real_B_for_G = real_B.repeat(1, cfg.input_nc // cfg.output_nc, 1, 1)
                else:
                    real_B_for_G = real_B.mean(dim=1, keepdim=True)
            idt_B = self.netG(real_B_for_G)
            real_B_pool, idt_B_pool = _pool_encoder_features(
                self.netG,
                self.netF,
                real_B_for_G,
                idt_B,
                self.nce_layers,
                cfg.num_patches,
                cfg.input_nc,
            )
            _, weights_idt = self._compute_src_and_weights(
                real_B_pool, idt_B_pool, only_weight=True, epoch=epoch
            )
            loss_HDCE_Y = self._compute_hdce_loss(real_B_pool, idt_B_pool, weights_idt)
            loss_HDCE_both = (loss_HDCE + loss_HDCE_Y) * 0.5
        else:
            loss_HDCE_both = loss_HDCE

        loss_G = loss_G_GAN + loss_HDCE_both + loss_SRC
        return loss_G, loss_G_GAN, loss_HDCE, loss_HDCE_Y, loss_SRC

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
            self.netF.train()
            pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{cfg.epochs}")

            for batch in pbar:
                logs = self.train_step(batch["A"], batch["B"])
                global_step += 1

                if global_step % cfg.log_every == 0:
                    pbar.set_postfix(logs)
                    logger.info(
                        "step %d | loss_D=%.4f loss_G=%.4f loss_HDCE=%.4f loss_SRC=%.4f",
                        global_step,
                        logs["loss_D"],
                        logs["loss_G"],
                        logs["loss_HDCE"],
                        logs["loss_SRC"],
                    )

            if (epoch + 1) % cfg.save_every == 0:
                self.save_checkpoint(save_dir, epoch + 1, global_step=global_step)

        logger.info("Hneg-SRC training complete. Checkpoints saved to %s", save_dir)

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
        feat_path = path / "feature_network"
        feat_path.mkdir(exist_ok=True)
        save_file(
            self.netF.state_dict(),
            feat_path / "diffusion_pytorch_model.safetensors",
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
