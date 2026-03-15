# Copyright (c) 2026 EarthBridge Team.
# Credits: Adapted from StegoGAN (Wu et al., CVPR 2024).
# Original: https://github.com/sian-wusidi/StegoGAN

"""StegoGAN trainer for non-bijective image-to-image translation.

Implements the StegoGAN training loop which uses steganographic masking
to detect and mitigate semantic misalignment between unpaired domains.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import torch
import torch.nn as nn

from src.losses.adversarial import GANLoss
from src.training import create_optimizer
from src.models.discriminators import PatchGANDiscriminator
from src.models.stegogan.generators import (
    ResnetMaskV1Generator,
    ResnetMaskV3Generator,
)
from src.models.stegogan.networks import get_norm_layer


@dataclass
class StegoGANConfig:
    """Configuration for the StegoGAN trainer.

    Parameters
    ----------
    input_nc : int
        Number of channels in input (domain A) images.
    output_nc : int
        Number of channels in output (domain B) images.
    ngf : int
        Base number of generator filters.
    ndf : int
        Base number of discriminator filters.
    n_layers_D : int
        Number of layers in the PatchGAN discriminator.
    norm : str
        Normalisation type (``"batch"`` | ``"instance"``).
    no_dropout : bool
        If ``True``, disable dropout in generators.
    gan_mode : str
        GAN loss type: ``"lsgan"`` | ``"vanilla"`` | ``"hinge"``.
    lambda_A : float
        Weight for forward cycle loss (A → B → A).
    lambda_B : float
        Weight for backward cycle loss (B → A → B).
    lambda_identity : float
        Weight for identity mapping loss.
    lambda_reg : float
        Weight for mask regularisation loss.
    lambda_consistency : float
        Weight for steganographic consistency loss.
    resnet_layer : int
        ResNet block index for steganographic feature injection.
    fusionblock : bool
        Whether to use a fusion block for injected features.
    mask_group : int
        Number of mask output channels.
    lr_G : float
        Learning rate for generators.
    lr_D : float
        Learning rate for discriminators.
    beta1 : float
        Adam beta1.
    epochs : int
        Number of training epochs.
    optimizer : str
        Optimizer type: ``"adamw"`` | ``"adam"`` | ``"prodigy"`` | ``"muon"``.
    weight_decay : float
        Weight decay for adamw/prodigy/muon.
    prodigy_d0 : float
        Prodigy d0 parameter (for optimizer_type=prodigy).
    """

    input_nc: int = 3
    output_nc: int = 3
    ngf: int = 64
    ndf: int = 64
    n_layers_D: int = 3
    norm: str = "batch"
    no_dropout: bool = True
    gan_mode: str = "lsgan"
    lambda_A: float = 10.0
    lambda_B: float = 10.0
    lambda_identity: float = 0.5
    lambda_reg: float = 0.2
    lambda_consistency: float = 3.0
    resnet_layer: int = 8
    fusionblock: bool = True
    mask_group: int = 256
    lr_G: float = 2e-4
    lr_D: float = 2e-4
    beta1: float = 0.5
    epochs: int = 100
    optimizer: str = "adam"  # "adamw" | "adam" | "prodigy" | "muon"
    weight_decay: float = 0.01
    prodigy_d0: float = 1e-6
    device: str = "cpu"
    checkpoint_dir: str = "./checkpoints"
    save_every: int = 10
    log_every: int = 100


class StegoGANTrainer:
    """Training harness for the StegoGAN model.

    Orchestrates two generator-discriminator pairs with additional
    steganographic consistency and mask regularisation losses.

    Parameters
    ----------
    config : StegoGANConfig
        Training configuration.
    """

    def __init__(self, config: StegoGANConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        norm_layer = get_norm_layer(config.norm)

        # Generators
        self.netG_A = ResnetMaskV1Generator(
            input_nc=config.input_nc,
            output_nc=config.output_nc,
            ngf=config.ngf,
            norm_layer=norm_layer,
            use_dropout=not config.no_dropout,
            resnet_layer=config.resnet_layer,
            fusionblock=config.fusionblock,
        ).to(self.device)

        self.netG_B = ResnetMaskV3Generator(
            input_nc=config.output_nc,
            output_nc=config.input_nc,
            ngf=config.ngf,
            norm_layer=norm_layer,
            use_dropout=not config.no_dropout,
            input_dim=config.ngf * 4,
            out_dim=config.mask_group,
            resnet_layer=config.resnet_layer,
        ).to(self.device)

        # Discriminators
        self.netD_A = PatchGANDiscriminator(
            in_channels=config.output_nc,
            base_filters=config.ndf,
            n_layers=config.n_layers_D,
        ).to(self.device)

        self.netD_B = PatchGANDiscriminator(
            in_channels=config.input_nc,
            base_filters=config.ndf,
            n_layers=config.n_layers_D,
        ).to(self.device)

        # Losses
        self.criterion_GAN = GANLoss(mode=config.gan_mode).to(self.device)
        self.criterion_cycle = nn.L1Loss()
        self.criterion_idt = nn.L1Loss()

        # Optimizers
        betas = (config.beta1, 0.999)
        weight_decay = config.weight_decay if config.optimizer.lower() in ("adamw", "prodigy", "muon") else 0.0
        self.optimizer_G = create_optimizer(
            itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
            optimizer_type=config.optimizer,
            lr=config.lr_G,
            weight_decay=weight_decay,
            betas=betas,
            prodigy_d0=config.prodigy_d0,
        )
        self.optimizer_D = create_optimizer(
            itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
            optimizer_type=config.optimizer,
            lr=config.lr_D,
            weight_decay=weight_decay,
            betas=betas,
            prodigy_d0=config.prodigy_d0,
        )

    def forward(
        self,
        real_A: torch.Tensor,
        real_B: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run the full forward pass for one batch.

        Returns a dictionary of all intermediate tensors needed for
        loss computation and visualisation.
        """
        # B → A with masking
        fake_A, latent_real_B, latent_real_B_mask = self.netG_B(real_B)

        # A → B (clean, without steganographic injection)
        fake_B_clean = self.netG_A(real_A)

        # A → B (with steganographic feature injection from G_B)
        fake_B = self.netG_A(real_A, latent_real_B.detach())

        # Cycle: A → B → A (with stego)
        rec_A, latent_fake_B, latent_fake_B_mask = self.netG_B(fake_B)

        # Cycle: B → A → B (with noise for robustness)
        noise = 0.01 * torch.randn_like(fake_A)
        rec_B = self.netG_A(fake_A + noise, latent_real_B)

        # Cycle: B → A → B (clean)
        rec_B_clean = self.netG_A(fake_A)

        # Cycle: A → B_clean → A (clean)
        rec_A_clean, _, _ = self.netG_B(fake_B_clean)

        # Upsample masks for visualisation / consistency loss
        upsample = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        latent_real_B_mask_up = upsample(latent_real_B_mask)
        latent_fake_B_mask_up = upsample(latent_fake_B_mask)

        return {
            "real_A": real_A,
            "real_B": real_B,
            "fake_A": fake_A,
            "fake_B": fake_B,
            "fake_B_clean": fake_B_clean,
            "rec_A": rec_A,
            "rec_A_clean": rec_A_clean,
            "rec_B": rec_B,
            "rec_B_clean": rec_B_clean,
            "latent_real_B": latent_real_B,
            "latent_fake_B": latent_fake_B,
            "latent_real_B_mask": latent_real_B_mask,
            "latent_fake_B_mask": latent_fake_B_mask,
            "latent_real_B_mask_up": latent_real_B_mask_up,
            "latent_fake_B_mask_up": latent_fake_B_mask_up,
        }

    def compute_G_loss(self, fwd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Compute all generator losses.

        Returns a dictionary of named scalar losses plus ``"total"``
        for the combined generator loss.
        """
        cfg = self.config
        losses: dict[str, torch.Tensor] = {}

        # GAN losses
        losses["G_A"] = self.criterion_GAN(self.netD_A(fwd["fake_B"]), target_is_real=True)
        losses["G_B"] = self.criterion_GAN(self.netD_B(fwd["fake_A"]), target_is_real=True)

        # Cycle losses
        losses["cycle_A"] = self.criterion_cycle(fwd["rec_A"], fwd["real_A"]) * cfg.lambda_A
        losses["cycle_B"] = self.criterion_cycle(fwd["rec_B"], fwd["real_B"]) * cfg.lambda_B

        # Identity losses
        if cfg.lambda_identity > 0:
            idt_A = self.netG_A(fwd["real_B"])
            losses["idt_A"] = self.criterion_idt(idt_A, fwd["real_B"]) * cfg.lambda_B * cfg.lambda_identity
            idt_B, _, _ = self.netG_B(fwd["real_A"])
            losses["idt_B"] = self.criterion_idt(idt_B, fwd["real_A"]) * cfg.lambda_A * cfg.lambda_identity
        else:
            losses["idt_A"] = torch.tensor(0.0, device=self.device)
            losses["idt_B"] = torch.tensor(0.0, device=self.device)

        # Mask regularisation
        losses["reg"] = cfg.lambda_reg * (
            torch.mean((fwd["latent_real_B_mask"] + 1e-10) ** 0.5)
            + torch.mean((fwd["latent_fake_B_mask"] + 1e-10) ** 0.5)
        )

        # Steganographic consistency loss
        losses["consistency_B"] = cfg.lambda_consistency * (
            self.criterion_cycle(
                fwd["rec_B_clean"] * (1 - fwd["latent_real_B_mask_up"]),
                fwd["real_B"] * (1 - fwd["latent_real_B_mask_up"]),
            )
            + self.criterion_cycle(
                fwd["fake_B_clean"] * (1 - fwd["latent_fake_B_mask_up"]),
                fwd["fake_B"] * (1 - fwd["latent_fake_B_mask_up"]),
            )
        )

        # Feature consistency
        losses["consistency_feat"] = cfg.lambda_consistency * self.criterion_cycle(
            fwd["latent_fake_B"], fwd["latent_real_B"].detach()
        )

        losses["total"] = (
            losses["G_A"]
            + losses["G_B"]
            + losses["cycle_A"]
            + losses["cycle_B"]
            + losses["idt_A"]
            + losses["idt_B"]
            + losses["reg"]
            + losses["consistency_B"]
            + losses["consistency_feat"]
        )
        return losses

    def compute_D_loss(
        self,
        netD: nn.Module,
        real: torch.Tensor,
        fake: torch.Tensor,
    ) -> torch.Tensor:
        """Compute discriminator loss for one discriminator."""
        loss_real = self.criterion_GAN(netD(real), target_is_real=True)
        loss_fake = self.criterion_GAN(netD(fake.detach()), target_is_real=False)
        return (loss_real + loss_fake) * 0.5

    def train_step(
        self,
        real_A: torch.Tensor,
        real_B: torch.Tensor,
    ) -> dict[str, float]:
        """Execute one training step.

        Returns a dictionary of scalar loss values for logging.
        """
        real_A = real_A.to(self.device)
        real_B = real_B.to(self.device)

        # Forward
        fwd = self.forward(real_A, real_B)

        # Generator update
        self._set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        g_losses = self.compute_G_loss(fwd)
        g_losses["total"].backward()
        self.optimizer_G.step()

        # Discriminator update
        self._set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        loss_D_A = self.compute_D_loss(self.netD_A, fwd["real_B"], fwd["fake_B"])
        loss_D_B = self.compute_D_loss(self.netD_B, fwd["real_A"], fwd["fake_A"])
        loss_D_A.backward()
        loss_D_B.backward()
        self.optimizer_D.step()

        return {k: v.item() for k, v in g_losses.items() if k != "total"} | {
            "D_A": loss_D_A.item(),
            "D_B": loss_D_B.item(),
            "G_total": g_losses["total"].item(),
        }

    @staticmethod
    def _set_requires_grad(nets: list[nn.Module], requires_grad: bool) -> None:
        for net in nets:
            for param in net.parameters():
                param.requires_grad = requires_grad
