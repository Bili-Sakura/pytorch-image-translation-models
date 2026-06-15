# Credits: CycleGAN (Zhu et al., ICCV 2017) — https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

"""CycleGAN trainer for unpaired image-to-image translation."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.data.datasets import UnpairedImageDataset
from src.models.cyclegan_pix2pix import GANLoss, create_discriminator, create_generator
from src.models.flsesim import ImagePool
from src.training import create_optimizer
from src.utils.config_yaml import save_config_yaml

from .config import CycleGANConfig

logger = logging.getLogger(__name__)


def _build_transform(resolution: int):
    return transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
    ])


class CycleGANTrainer:
    """Training harness for CycleGAN (G_A, G_B, D_A, D_B)."""

    def __init__(self, config: CycleGANConfig) -> None:
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        self.netG_A = create_generator(
            input_nc=config.input_nc,
            output_nc=config.output_nc,
            ngf=config.ngf,
            netG=config.netG,
            norm=config.normG,
            use_dropout=not config.no_dropout,
            init_type=config.init_type,
            init_gain=config.init_gain,
        ).to(self.device)
        self.netG_B = create_generator(
            input_nc=config.output_nc,
            output_nc=config.input_nc,
            ngf=config.ngf,
            netG=config.netG,
            norm=config.normG,
            use_dropout=not config.no_dropout,
            init_type=config.init_type,
            init_gain=config.init_gain,
        ).to(self.device)
        self.netD_A = create_discriminator(
            input_nc=config.output_nc,
            ndf=config.ndf,
            netD=config.netD,
            n_layers_D=config.n_layers_D,
            norm=config.normD,
            init_type=config.init_type,
            init_gain=config.init_gain,
        ).to(self.device)
        self.netD_B = create_discriminator(
            input_nc=config.input_nc,
            ndf=config.ndf,
            netD=config.netD,
            n_layers_D=config.n_layers_D,
            norm=config.normD,
            init_type=config.init_type,
            init_gain=config.init_gain,
        ).to(self.device)

        self.criterion_GAN = GANLoss(config.gan_mode).to(self.device)
        self.criterion_cycle = torch.nn.L1Loss()
        self.criterion_idt = torch.nn.L1Loss()
        self.fake_A_pool = ImagePool(config.pool_size)
        self.fake_B_pool = ImagePool(config.pool_size)

        betas = (config.beta1, config.beta2)
        weight_decay = (
            config.weight_decay if config.optimizer.lower() in ("adamw", "prodigy", "muon") else 0.0
        )
        g_params = list(self.netG_A.parameters()) + list(self.netG_B.parameters())
        d_params = list(self.netD_A.parameters()) + list(self.netD_B.parameters())

        self.optimizer_G = create_optimizer(
            g_params,
            optimizer_type=config.optimizer,
            lr=config.lr,
            weight_decay=weight_decay,
            betas=betas,
            prodigy_d0=config.prodigy_d0,
        )
        self.optimizer_D = create_optimizer(
            d_params,
            optimizer_type=config.optimizer,
            lr=config.lr,
            weight_decay=weight_decay,
            betas=betas,
            prodigy_d0=config.prodigy_d0,
        )

    def build_dataset(self, root_a: str | Path, root_b: str | Path) -> UnpairedImageDataset:
        transform = _build_transform(self.config.resolution)
        return UnpairedImageDataset(
            root_a=root_a,
            root_b=root_b,
            transform_a=transform,
            transform_b=transform,
        )

    def _set_requires_grad(self, nets: list[torch.nn.Module], requires_grad: bool) -> None:
        for net in nets:
            for param in net.parameters():
                param.requires_grad = requires_grad

    def _backward_D_basic(self, netD, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        pred_real = netD(real)
        loss_real = self.criterion_GAN(pred_real, True)
        pred_fake = netD(fake.detach())
        loss_fake = self.criterion_GAN(pred_fake, False)
        loss_D = (loss_real + loss_fake) * 0.5
        loss_D.backward()
        return loss_D

    def train_step(self, real_A: torch.Tensor, real_B: torch.Tensor) -> dict[str, float]:
        cfg = self.config
        real_A = real_A.to(self.device) * 2 - 1
        real_B = real_B.to(self.device) * 2 - 1

        fake_B = self.netG_A(real_A)
        rec_A = self.netG_B(fake_B)
        fake_A = self.netG_B(real_B)
        rec_B = self.netG_A(fake_A)

        # Generator
        self._set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()

        loss_G_A = self.criterion_GAN(self.netD_A(fake_B), True)
        loss_G_B = self.criterion_GAN(self.netD_B(fake_A), True)
        loss_cycle_A = self.criterion_cycle(rec_A, real_A) * cfg.lambda_A
        loss_cycle_B = self.criterion_cycle(rec_B, real_B) * cfg.lambda_B

        loss_idt_A = torch.tensor(0.0, device=self.device)
        loss_idt_B = torch.tensor(0.0, device=self.device)
        if cfg.lambda_identity > 0:
            assert cfg.input_nc == cfg.output_nc
            idt_A = self.netG_A(real_B)
            loss_idt_A = self.criterion_idt(idt_A, real_B) * cfg.lambda_B * cfg.lambda_identity
            idt_B = self.netG_B(real_A)
            loss_idt_B = self.criterion_idt(idt_B, real_A) * cfg.lambda_A * cfg.lambda_identity

        loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B
        loss_G.backward()
        self.optimizer_G.step()

        # Discriminators
        self._set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        loss_D_A = self._backward_D_basic(
            self.netD_A, real_B, self.fake_B_pool.query(fake_B)
        )
        loss_D_B = self._backward_D_basic(
            self.netD_B, real_A, self.fake_A_pool.query(fake_A)
        )
        self.optimizer_D.step()

        return {
            "loss_D_A": float(loss_D_A),
            "loss_D_B": float(loss_D_B),
            "loss_G": float(loss_G),
            "loss_cycle_A": float(loss_cycle_A),
            "loss_cycle_B": float(loss_cycle_B),
        }

    def train(self, root_a: str | Path, root_b: str | Path) -> None:
        cfg = self.config
        os.makedirs(cfg.save_dir, exist_ok=True)

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
            self.netG_A.train()
            self.netG_B.train()
            self.netD_A.train()
            self.netD_B.train()
            pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{cfg.epochs}")

            for batch in pbar:
                logs = self.train_step(batch["A"], batch["B"])
                global_step += 1
                if global_step % cfg.log_every == 0:
                    pbar.set_postfix(logs)
                    logger.info(
                        "step %d | loss_G=%.4f loss_D_A=%.4f loss_D_B=%.4f",
                        global_step,
                        logs["loss_G"],
                        logs["loss_D_A"],
                        logs["loss_D_B"],
                    )

            if (epoch + 1) % cfg.save_every == 0:
                self.save_checkpoint(cfg.save_dir, epoch + 1, global_step=global_step)

        logger.info("CycleGAN training complete. Checkpoints saved to %s", cfg.save_dir)

    def save_checkpoint(
        self,
        save_dir: str,
        epoch_or_step: int,
        *,
        global_step: int | None = None,
    ) -> None:
        path = Path(save_dir) / f"checkpoint-epoch-{epoch_or_step}"
        path.mkdir(parents=True, exist_ok=True)

        gen_a_path = path / "generator_a"
        gen_a_path.mkdir(exist_ok=True)
        self.netG_A.save_pretrained(gen_a_path)

        gen_b_path = path / "generator_b"
        gen_b_path.mkdir(exist_ok=True)
        self.netG_B.save_pretrained(gen_b_path)

        from safetensors.torch import save_file

        disc_a_path = path / "discriminator_a"
        disc_a_path.mkdir(exist_ok=True)
        save_file(self.netD_A.state_dict(), disc_a_path / "diffusion_pytorch_model.safetensors")

        disc_b_path = path / "discriminator_b"
        disc_b_path.mkdir(exist_ok=True)
        save_file(self.netD_B.state_dict(), disc_b_path / "diffusion_pytorch_model.safetensors")

        step = global_step if global_step is not None else epoch_or_step
        torch.save(
            {
                "optimizer_G": self.optimizer_G.state_dict(),
                "optimizer_D": self.optimizer_D.state_dict(),
                "epoch": epoch_or_step,
                "global_step": step,
            },
            path / "training_state.pt",
        )
        save_config_yaml(self.config, path / "config.yaml", extra={"epoch": epoch_or_step, "global_step": step})
        logger.info("Checkpoint saved to %s", path)
