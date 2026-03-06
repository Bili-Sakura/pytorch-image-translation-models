# Copyright (c) 2026 EarthBridge Team.
# Credits: Adapted from Parallel-GAN (Wang et al., TGRS 2022).
# Original: https://github.com/ZZG-Z/Parallel-GAN

"""Parallel-GAN – SAR-to-Optical image translation with hierarchical latent features.

This is a **community pipeline**: a self-contained, single-file module that
bundles all network definitions, losses, and a training helper so it can be
used without importing any other project code.

Paper
-----
    Wang H., Zhang Z., Hu Z., Dong Q.
    *SAR-to-Optical Image Translation with Hierarchical Latent Features*
    IEEE Trans. Geoscience and Remote Sensing (TGRS), 2022.
    https://ieeexplore.ieee.org/document/9864654

Architecture
------------
The Parallel-GAN uses a two-stage training approach:

* **Stage 1 – Reconstruction network** (``Resrecon``):
  A ResNet-50 encoder + transposed-conv decoder trained to reconstruct
  optical images from optical images, learning a rich feature hierarchy.

* **Stage 2 – Translation network** (``ParaGAN``):
  A ResNet-50 encoder (accepts SAR input) with residual bottleneck blocks
  and a decoder that fuses encoder skip connections (like U-Net) plus a
  *hierarchical feature loss* that aligns intermediate features with
  the pre-trained reconstruction network.

Usage
-----
.. code-block:: python

    from examples.community.parallel_gan import (
        ParaGAN,
        Resrecon,
        ParallelGANTrainer,
        ParallelGANConfig,
    )

    cfg = ParallelGANConfig(input_nc=3, output_nc=3, device="cpu")
    trainer = ParallelGANTrainer(cfg)
    losses = trainer.train_step(sar_batch, optical_batch)

Citation
--------
.. code-block:: bibtex

    @ARTICLE{9864654,
      author={Wang, Haixia and Zhang, Zhigang and Hu, Zhanyi and Dong, Qiulei},
      journal={IEEE Trans. Geoscience and Remote Sensing},
      title={SAR-to-Optical Image Translation with Hierarchical Latent Features},
      year={2022},
      volume={60},
      pages={1-12},
      doi={10.1109/TGRS.2022.3200996}}
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models


# ---------------------------------------------------------------------------
# Initialisation helpers
# ---------------------------------------------------------------------------


def _init_weights(net: nn.Module, init_type: str = "normal", init_gain: float = 0.02) -> None:
    """Apply weight initialisation to *net* in-place."""

    def _init_func(m: nn.Module) -> None:
        cls = m.__class__.__name__
        if hasattr(m, "weight") and ("Conv" in cls or "Linear" in cls):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(f"init method [{init_type}] not implemented")
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif "BatchNorm2d" in cls:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(_init_func)


# ---------------------------------------------------------------------------
# ResNet block
# ---------------------------------------------------------------------------


class _ResnetBlock(nn.Module):
    """ResNet block with configurable padding, normalisation, and dropout."""

    def __init__(
        self,
        dim: int,
        padding_type: str = "reflect",
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
        use_dropout: bool = False,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        p = 0
        if padding_type == "reflect":
            layers.append(nn.ReflectionPad2d(1))
        elif padding_type == "replicate":
            layers.append(nn.ReplicationPad2d(1))
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError(f"padding [{padding_type}] not implemented")

        layers += [nn.Conv2d(dim, dim, 3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            layers.append(nn.Dropout(0.5))

        p = 0
        if padding_type == "reflect":
            layers.append(nn.ReflectionPad2d(1))
        elif padding_type == "replicate":
            layers.append(nn.ReplicationPad2d(1))
        elif padding_type == "zero":
            p = 1

        layers += [nn.Conv2d(dim, dim, 3, padding=p, bias=use_bias), norm_layer(dim)]
        self.conv_block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv_block(x)


# ---------------------------------------------------------------------------
# Reconstruction network  (Stage 1)
# ---------------------------------------------------------------------------


class Resrecon(nn.Module):
    """Reconstruction network: ResNet-50 encoder → transposed-conv decoder.

    Trained in Stage 1 to reconstruct optical images from optical images,
    learning a rich feature hierarchy.  At inference / Stage 2, the
    intermediate features serve as supervision targets.

    The forward pass returns a list of six feature tensors (five
    intermediate decoder features + the final RGB output).
    """

    def __init__(self) -> None:
        super().__init__()
        norm_layer = nn.BatchNorm2d
        use_bias = False

        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, 4, stride=2, padding=1, bias=use_bias),
            norm_layer(1024),
            nn.ReLU(False),
            nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1, bias=use_bias),
            norm_layer(512),
            nn.ReLU(False),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=use_bias),
            norm_layer(256),
            nn.ReLU(False),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=use_bias),
            norm_layer(128),
            nn.ReLU(False),
            nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1, bias=use_bias),
            norm_layer(64),
            nn.ReLU(False),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1, bias=use_bias),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        out_features: list[torch.Tensor] = []
        feat = self.encoder(x)
        for i in range(5):
            feat = self.decoder[3 * i : 3 * (i + 1)](feat)
            out_features.append(feat)
        out_features.append(self.decoder[15:17](feat))
        return out_features


# ---------------------------------------------------------------------------
# Translation network  (Stage 2)
# ---------------------------------------------------------------------------


class ParaGAN(nn.Module):
    """Parallel-GAN translation generator.

    Uses a ResNet-50 encoder, six residual bottleneck blocks, and a decoder
    with encoder skip connections (concatenation).  The forward pass returns
    a list of six feature tensors for hierarchical feature supervision.

    Parameters
    ----------
    input_nc : int
        Number of input channels (e.g. 3 for RGB SAR, 4 for SpaceNet).
    output_nc : int
        Number of output channels (typically 3 for RGB optical).
    channel : int
        Channel width inside the residual blocks (default 2048).
    norm_layer
        Normalisation layer constructor.
    use_dropout : bool
        Use dropout inside residual blocks.
    n_blocks : int
        Number of residual blocks between encoder and decoder.
    """

    def __init__(
        self,
        input_nc: int = 3,
        output_nc: int = 3,
        channel: int = 2048,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
        use_dropout: bool = False,
        n_blocks: int = 6,
        padding_type: str = "reflect",
    ) -> None:
        super().__init__()
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])

        # Handle 4-channel SAR input
        if input_nc == 4:
            with torch.no_grad():
                pretrained_w = self.encoder[0].weight.clone()
                self.encoder[0] = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
                self.encoder[0].weight[:, :3] = pretrained_w
                self.encoder[0].weight[:, 3:] = pretrained_w[:, :1]

        self.resnet_part = nn.Sequential(
            *[
                _ResnetBlock(channel, padding_type=padding_type, norm_layer=norm_layer,
                             use_dropout=use_dropout, use_bias=use_bias)
                for _ in range(n_blocks)
            ]
        )

        self.decoder = nn.Sequential(
            # Stage 0: 2048 → 1024
            nn.ConvTranspose2d(2048, 1024, 4, stride=2, padding=1, bias=use_bias),
            norm_layer(1024),
            nn.ReLU(False),
            # Stage 1: 2048 (cat) → 512
            nn.ConvTranspose2d(2048, 512, 4, stride=2, padding=1, bias=use_bias),
            norm_layer(512),
            nn.ReLU(False),
            # Stage 2: 1024 (cat) → 256
            nn.ConvTranspose2d(1024, 256, 4, stride=2, padding=1, bias=use_bias),
            norm_layer(256),
            nn.ReLU(False),
            # Stage 3: 512 (cat) → 128
            nn.ConvTranspose2d(512, 128, 4, stride=2, padding=1, bias=use_bias),
            norm_layer(128),
            nn.ReLU(False),
            # Stage 4: 128 → 64
            nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1, bias=use_bias),
            norm_layer(64),
            nn.ReLU(False),
            # Final: 128 (cat) → output_nc
            nn.ConvTranspose2d(128, output_nc, 4, stride=2, padding=1, bias=use_bias),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        encoder_features: list[torch.Tensor] = []
        out_features: list[torch.Tensor] = []

        feat = x
        for block in self.encoder:
            feat = block(feat)
            encoder_features.append(feat)

        decoder_feat = self.resnet_part(encoder_features[-1])

        for i in range(5):
            decoder_feat = self.decoder[3 * i : 3 * (i + 1)](decoder_feat)
            out_features.append(decoder_feat)
            if i < 3:
                decoder_feat = torch.cat([decoder_feat, encoder_features[6 - i]], dim=1)

        decoder_feat = torch.cat([decoder_feat, encoder_features[0]], dim=1)
        out_features.append(self.decoder[15:17](decoder_feat))
        return out_features


# ---------------------------------------------------------------------------
# PatchGAN discriminator
# ---------------------------------------------------------------------------


class _NLayerDiscriminator(nn.Module):
    """PatchGAN discriminator."""

    def __init__(
        self,
        input_nc: int,
        ndf: int = 64,
        n_layers: int = 3,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
    ) -> None:
        super().__init__()
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw, padw = 4, 1
        sequence: list[nn.Module] = [
            nn.Conv2d(input_nc, ndf, kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * nf_mult, 1, kw, stride=1, padding=padw),
        ]
        self.model = nn.Sequential(*sequence)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ---------------------------------------------------------------------------
# VGG perceptual loss
# ---------------------------------------------------------------------------


class _VGGFeatures(nn.Module):
    """VGG-19 multi-layer feature extractor (frozen)."""

    def __init__(self) -> None:
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        slices = [2, 7, 12, 21, 30]
        self.slices = nn.ModuleList()
        prev = 0
        for s in slices:
            self.slices.append(nn.Sequential(*[vgg[i] for i in range(prev, s)]))
            prev = s
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats: list[torch.Tensor] = []
        h = x
        for s in self.slices:
            h = s(h)
            feats.append(h)
        return feats


class VGGLoss(nn.Module):
    """Multi-layer VGG perceptual loss used by the reconstruction network."""

    def __init__(self) -> None:
        super().__init__()
        self.vgg = _VGGFeatures()
        self.criterion = nn.L1Loss()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_feats = self.vgg(x)
        y_feats = self.vgg(y)
        loss = torch.tensor(0.0, device=x.device)
        for xf, yf in zip(x_feats, y_feats):
            loss = loss + self.criterion(xf, yf.detach())
        return loss


# ---------------------------------------------------------------------------
# GAN loss
# ---------------------------------------------------------------------------


class _GANLoss(nn.Module):
    """GAN loss (vanilla / lsgan)."""

    def __init__(self, gan_mode: str = "vanilla") -> None:
        super().__init__()
        self.register_buffer("real_label", torch.tensor(1.0))
        self.register_buffer("fake_label", torch.tensor(0.0))
        self.gan_mode = gan_mode
        if gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        elif gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError(f"GAN mode [{gan_mode}] not implemented")

    def _target(self, pred: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        return (self.real_label if target_is_real else self.fake_label).expand_as(pred)

    def forward(self, pred: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        return self.loss(pred, self._target(pred, target_is_real))


# ---------------------------------------------------------------------------
# Trainer
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
