# Credits: SAR2Optical (https://github.com/yuuIind/SAR2Optical), pix2pix (Isola et al.).
"""Training config and harness for SAR2Optical pix2pix cGAN."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from examples.community.sar2optical.model import (
    SAR2OpticalDiscriminator,
    SAR2OpticalGenerator,
    init_weights,
)


@dataclass
class SAR2OpticalConfig:
    """Configuration for SAR2Optical trainer."""

    c_in: int = 3
    c_out: int = 3
    netD: str = "patch"
    lambda_l1: float = 100.0
    is_cgan: bool = True
    use_upsampling: bool = False
    mode: str = "nearest"
    c_hid: int = 64
    n_layers: int = 3
    lr: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    device: str = "cpu"


class SAR2OpticalTrainer:
    """Single-step training harness matching SAR2Optical pix2pix logic."""

    def __init__(self, config: SAR2OpticalConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.is_cgan = config.is_cgan
        self.lambda_l1 = config.lambda_l1

        self.gen = SAR2OpticalGenerator(
            c_in=config.c_in,
            c_out=config.c_out,
            use_upsampling=config.use_upsampling,
            mode=config.mode,
        ).to(self.device)
        self.gen.apply(init_weights)

        disc_in = config.c_in + config.c_out if config.is_cgan else config.c_out
        self.disc = SAR2OpticalDiscriminator(
            c_in=disc_in,
            c_hid=config.c_hid,
            mode=config.netD,
            n_layers=config.n_layers,
        ).to(self.device)
        self.disc.apply(init_weights)

        self.gen_optimizer = torch.optim.Adam(
            self.gen.parameters(),
            lr=config.lr,
            betas=(config.beta1, config.beta2),
        )
        self.disc_optimizer = torch.optim.Adam(
            self.disc.parameters(),
            lr=config.lr,
            betas=(config.beta1, config.beta2),
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self.criterion_l1 = nn.L1Loss()

    def _get_disc_inputs(
        self,
        real_images: torch.Tensor,
        target_images: torch.Tensor,
        fake_images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.is_cgan:
            real_ab = torch.cat([real_images, target_images], dim=1)
            fake_ab = torch.cat([real_images, fake_images.detach()], dim=1)
        else:
            real_ab = target_images
            fake_ab = fake_images.detach()
        return real_ab, fake_ab

    def _get_gen_inputs(self, real_images: torch.Tensor, fake_images: torch.Tensor) -> torch.Tensor:
        if self.is_cgan:
            return torch.cat([real_images, fake_images], dim=1)
        return fake_images

    def step_discriminator(
        self,
        real_images: torch.Tensor,
        target_images: torch.Tensor,
        fake_images: torch.Tensor,
    ) -> torch.Tensor:
        real_ab, fake_ab = self._get_disc_inputs(real_images, target_images, fake_images)
        pred_real = self.disc(real_ab)
        pred_fake = self.disc(fake_ab)
        loss_d_real = self.criterion(pred_real, torch.ones_like(pred_real))
        loss_d_fake = self.criterion(pred_fake, torch.zeros_like(pred_fake))
        return (loss_d_real + loss_d_fake) * 0.5

    def step_generator(
        self,
        real_images: torch.Tensor,
        target_images: torch.Tensor,
        fake_images: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        fake_ab = self._get_gen_inputs(real_images, fake_images)
        pred_fake = self.disc(fake_ab)
        loss_g_gan = self.criterion(pred_fake, torch.ones_like(pred_fake))
        loss_g_l1 = self.criterion_l1(fake_images, target_images)
        loss_g = loss_g_gan + self.lambda_l1 * loss_g_l1
        return loss_g, {
            "loss_G": loss_g.item(),
            "loss_G_GAN": loss_g_gan.item(),
            "loss_G_L1": loss_g_l1.item(),
        }

    def train_step(
        self,
        real_images: torch.Tensor,
        target_images: torch.Tensor,
    ) -> dict[str, float]:
        real_images = real_images.to(self.device)
        target_images = target_images.to(self.device)
        fake_images = self.gen(real_images)

        self.disc_optimizer.zero_grad()
        loss_d = self.step_discriminator(real_images, target_images, fake_images)
        loss_d.backward()
        self.disc_optimizer.step()

        self.gen_optimizer.zero_grad()
        loss_g, g_losses = self.step_generator(real_images, target_images, fake_images)
        loss_g.backward()
        self.gen_optimizer.step()

        return {"loss_D": loss_d.item(), **g_losses}

    @torch.no_grad()
    def validation_step(
        self,
        real_images: torch.Tensor,
        target_images: torch.Tensor,
    ) -> dict[str, float]:
        real_images = real_images.to(self.device)
        target_images = target_images.to(self.device)
        fake_images = self.gen(real_images)
        loss_d = self.step_discriminator(real_images, target_images, fake_images)
        _, g_losses = self.step_generator(real_images, target_images, fake_images)
        return {"loss_D": loss_d.item(), **g_losses}

    @torch.no_grad()
    def sample(self, real_images: torch.Tensor) -> torch.Tensor:
        """Generate translated images in [-1, 1] range."""
        self.gen.eval()
        out = self.gen(real_images.to(self.device)).clamp(-1, 1)
        self.gen.train()
        return out

    def save_checkpoint(self, gen_path: str, disc_path: str | None = None) -> None:
        """Save generator and optional discriminator checkpoints."""
        Path(gen_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.gen.state_dict(), gen_path)
        if disc_path is not None:
            Path(disc_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.disc.state_dict(), disc_path)

    def load_checkpoint(self, gen_path: str, disc_path: str | None = None) -> None:
        """Load generator and optional discriminator checkpoints."""
        self.gen.load_state_dict(torch.load(gen_path, map_location=self.device, weights_only=True), strict=False)
        if disc_path is not None:
            self.disc.load_state_dict(
                torch.load(disc_path, map_location=self.device, weights_only=True),
                strict=False,
            )
