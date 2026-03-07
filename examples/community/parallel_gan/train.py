# Copyright (c) 2026 EarthBridge Team.
# Credits: Adapted from Parallel-GAN (Wang et al., TGRS 2022).

"""Training configuration and harness for Parallel-GAN."""

from __future__ import annotations

import functools
from dataclasses import dataclass

import torch
import torch.nn as nn

from examples.community.parallel_gan.model import (
    ParaGAN,
    Resrecon,
    VGGLoss,
    _GANLoss,
    _NLayerDiscriminator,
    _init_weights,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ParallelGANConfig:
    """Configuration for :class:`ParallelGANTrainer`.

    Parameters
    ----------
    input_nc : int
        SAR input channels (3 or 4).
    output_nc : int
        Optical output channels.
    ngf : int
        Generator feature width.
    ndf : int
        Discriminator feature width.
    n_layers_D : int
        Discriminator layers.
    norm : str
        ``"batch"`` or ``"instance"``.
    gan_mode : str
        ``"vanilla"`` or ``"lsgan"``.
    lambda_L1 : float
        Weight for pixel-wise L1 loss.
    lambda_vgg : float
        Weight for VGG perceptual loss (Stage 1 only).
    lambda_feat : float
        Weight for hierarchical feature loss (Stage 2 only).
    feature_weights : tuple[float, ...]
        Per-level weights for the feature loss (5 levels).
    n_blocks : int
        Number of residual blocks in the translation generator.
    lr : float
        Learning rate.
    beta1 : float
        Adam beta1.
    device : str
        ``"cpu"`` or ``"cuda"``.
    """

    input_nc: int = 3
    output_nc: int = 3
    ngf: int = 64
    ndf: int = 64
    n_layers_D: int = 3
    norm: str = "batch"
    gan_mode: str = "vanilla"
    lambda_L1: float = 100.0
    lambda_vgg: float = 10.0
    lambda_feat: float = 10.0
    feature_weights: tuple[float, ...] = (1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2)
    n_blocks: int = 6
    lr: float = 2e-4
    beta1: float = 0.5
    device: str = "cpu"


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class ParallelGANTrainer:
    """Two-stage training harness for Parallel-GAN.

    Instantiate with a :class:`ParallelGANConfig`, then call
    :meth:`train_step_recon` for Stage 1 (reconstruction) or
    :meth:`train_step_trans` for Stage 2 (translation).

    Parameters
    ----------
    config : ParallelGANConfig
        Training configuration.
    recon_net : Resrecon, optional
        Pre-trained reconstruction network for Stage 2 feature supervision.
        If ``None``, only Stage 1 is available.
    """

    def __init__(
        self,
        config: ParallelGANConfig,
        recon_net: Resrecon | None = None,
    ) -> None:
        self.config = config
        self.device = torch.device(config.device)

        norm_layer: type[nn.Module]
        if config.norm == "batch":
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)  # type: ignore[assignment]

        # Generator (translation)
        self.netG = ParaGAN(
            input_nc=config.input_nc,
            output_nc=config.output_nc,
            norm_layer=norm_layer,
            n_blocks=config.n_blocks,
        ).to(self.device)
        _init_weights(self.netG)

        # Discriminator
        self.netD = _NLayerDiscriminator(
            input_nc=config.input_nc + config.output_nc,
            ndf=config.ndf,
            n_layers=config.n_layers_D,
            norm_layer=norm_layer,
        ).to(self.device)
        _init_weights(self.netD)

        # Reconstruction network (frozen, for Stage 2)
        self.recon_net = recon_net
        if self.recon_net is not None:
            self.recon_net = self.recon_net.to(self.device)
            self.recon_net.eval()
            for p in self.recon_net.parameters():
                p.requires_grad = False

        # Losses
        self.criterion_GAN = _GANLoss(config.gan_mode).to(self.device)
        self.criterion_L1 = nn.L1Loss()
        self.criterion_VGG = VGGLoss().to(self.device)

        # Optimisers
        self.optimizer_G = torch.optim.Adam(
            self.netG.parameters(), lr=config.lr, betas=(config.beta1, 0.999)
        )
        self.optimizer_D = torch.optim.Adam(
            self.netD.parameters(), lr=config.lr, betas=(config.beta1, 0.999)
        )

    # ----- Stage 1: reconstruction -----------------------------------------

    def train_step_recon(
        self,
        real_B: torch.Tensor,
    ) -> dict[str, float]:
        """One reconstruction training step (Stage 1).

        The reconstruction network learns to reconstruct optical images
        from optical images.

        Parameters
        ----------
        real_B : Tensor ``[B, C, H, W]``
            Optical images.
        """
        real_B = real_B.to(self.device)
        features = self.netG(real_B)
        fake_B = features[-1]

        # Discriminator
        self._set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        loss_D = self._backward_D(real_B, fake_B, real_B)
        self.optimizer_D.step()

        # Generator
        self._set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        fake_AB = torch.cat([real_B, fake_B], dim=1)
        loss_G_GAN = self.criterion_GAN(self.netD(fake_AB), target_is_real=True)
        loss_G_L1 = self.criterion_L1(fake_B, real_B) * self.config.lambda_L1
        loss_VGG = self.criterion_VGG(fake_B, real_B) * self.config.lambda_vgg
        loss_G = loss_G_GAN + loss_G_L1 + loss_VGG
        loss_G.backward()
        self.optimizer_G.step()

        return {
            "D": loss_D,
            "G_GAN": loss_G_GAN.item(),
            "G_L1": loss_G_L1.item(),
            "VGG": loss_VGG.item(),
        }

    # ----- Stage 2: translation --------------------------------------------

    def train_step_trans(
        self,
        real_A: torch.Tensor,
        real_B: torch.Tensor,
    ) -> dict[str, float]:
        """One translation training step (Stage 2).

        Parameters
        ----------
        real_A : Tensor ``[B, C_in, H, W]``
            SAR images.
        real_B : Tensor ``[B, C_out, H, W]``
            Optical images.
        """
        if self.recon_net is None:
            raise RuntimeError(
                "Stage 2 requires a pre-trained reconstruction network. "
                "Pass recon_net= to ParallelGANTrainer."
            )

        real_A = real_A.to(self.device)
        real_B = real_B.to(self.device)

        trans_features = self.netG(real_A)
        fake_B = trans_features[-1]

        with torch.no_grad():
            recon_features = self.recon_net(real_B)

        # Discriminator
        self._set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        loss_D = self._backward_D(real_A, fake_B, real_B)
        self.optimizer_D.step()

        # Generator
        self._set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        fake_AB = torch.cat([real_A, fake_B], dim=1)
        loss_G_GAN = self.criterion_GAN(self.netD(fake_AB), target_is_real=True)
        loss_G_L1 = self.criterion_L1(fake_B, real_B) * self.config.lambda_L1

        loss_feat = torch.tensor(0.0, device=self.device)
        weights = self.config.feature_weights
        for i in range(len(trans_features) - 1):
            loss_feat = loss_feat + self.criterion_L1(
                trans_features[i], recon_features[i]
            ) * weights[i] * self.config.lambda_feat

        loss_G = loss_G_GAN + loss_G_L1 + loss_feat
        loss_G.backward()
        self.optimizer_G.step()

        return {
            "D": loss_D,
            "G_GAN": loss_G_GAN.item(),
            "G_L1": loss_G_L1.item(),
            "feat": loss_feat.item(),
        }

    # ----- Convenience: single step (auto-selects stage) -------------------

    def train_step(
        self,
        real_A: torch.Tensor,
        real_B: torch.Tensor,
    ) -> dict[str, float]:
        """Run one training step.

        If a reconstruction network was provided, runs Stage 2 (translation).
        Otherwise, runs Stage 1 (reconstruction) using *real_B* only.
        """
        if self.recon_net is not None:
            return self.train_step_trans(real_A, real_B)
        return self.train_step_recon(real_B)

    # ----- Helpers ----------------------------------------------------------

    def _backward_D(
        self,
        real_A: torch.Tensor,
        fake_B: torch.Tensor,
        real_B: torch.Tensor,
    ) -> float:
        fake_AB = torch.cat([real_A, fake_B.detach()], dim=1)
        real_AB = torch.cat([real_A, real_B], dim=1)
        loss_fake = self.criterion_GAN(self.netD(fake_AB), target_is_real=False)
        loss_real = self.criterion_GAN(self.netD(real_AB), target_is_real=True)
        loss_D = (loss_fake + loss_real) * 0.5
        loss_D.backward()
        return loss_D.item()

    @staticmethod
    def _set_requires_grad(net: nn.Module, requires_grad: bool) -> None:
        for p in net.parameters():
            p.requires_grad = requires_grad
