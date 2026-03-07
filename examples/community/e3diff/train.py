# Copyright (c) 2026 EarthBridge Team.
# Credits: Adapted from E3Diff (Qin et al., IEEE GRSL 2024).

"""Training configuration and harness for E3Diff."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from examples.community.e3diff.model import (
    E3DiffUNet,
    FocalFrequencyLoss,
    _GANLoss,
    _NLayerDiscriminator,
    _init_weights,
)
from examples.community.e3diff.pipeline import GaussianDiffusion


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class E3DiffConfig:
    """Configuration for :class:`E3DiffTrainer`.

    Parameters
    ----------
    stage : int
        Training stage: ``1`` for diffusion pre-training, ``2`` for one-step
        adversarial refinement.
    condition_ch : int
        Number of channels of the SAR conditioning image (CPEN input).
    out_ch : int
        Number of channels of the optical output image.
    image_size : int
        Spatial resolution used during training.
    inner_channel : int
        Base channel width of the U-Net (default: 64 to match CPEN).
    channel_mults : tuple[int, ...]
        Per-scale channel multipliers. Must have exactly 5 entries.
    attn_res : tuple[int, ...]
        Spatial resolutions at which to add self-attention.
    res_blocks : int
        Number of residual blocks per scale.
    dropout : float
        Dropout probability.
    n_timestep : int
        Number of diffusion timesteps for the noise schedule.
    beta_schedule : str
        Beta schedule type: ``"linear"`` or ``"cosine"``.
    linear_start : float
        Starting beta value for the linear schedule.
    linear_end : float
        Ending beta value for the linear schedule.
    xT_noise_r : float
        Mixing ratio for DDPM-inversion initialisation during sampling.
    fft_weight : float
        Weight for the Focal Frequency Loss (0 to disable).
    lambda_gan : float
        Weight for the GAN loss in Stage 2 (0 to disable).
    ndf : int
        Discriminator base feature width (Stage 2 only).
    n_layers_D : int
        Number of discriminator layers (Stage 2 only).
    gan_mode : str
        GAN loss mode: ``"lsgan"`` or ``"vanilla"``.
    lr : float
        Learning rate for the generator.
    lr_D : float
        Learning rate for the discriminator (Stage 2 only).
    beta1 : float
        Adam beta1.
    device : str
        Device string, e.g. ``"cpu"`` or ``"cuda"``.
    """

    stage: int = 1
    condition_ch: int = 3
    out_ch: int = 3
    image_size: int = 256
    inner_channel: int = 64
    channel_mults: tuple[int, ...] = field(default_factory=lambda: (1, 2, 4, 8, 16))
    attn_res: tuple[int, ...] = field(default_factory=tuple)
    res_blocks: int = 1
    dropout: float = 0.0
    n_timestep: int = 1000
    beta_schedule: str = "linear"
    linear_start: float = 1e-6
    linear_end: float = 1e-2
    xT_noise_r: float = 0.1
    fft_weight: float = 0.0
    lambda_gan: float = 0.1
    ndf: int = 64
    n_layers_D: int = 3
    gan_mode: str = "lsgan"
    lr: float = 5e-5
    lr_D: float = 5e-5
    beta1: float = 0.9
    device: str = "cpu"


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class E3DiffTrainer:
    """Two-stage training harness for E3Diff.

    Instantiate with an :class:`E3DiffConfig`, then call
    :meth:`train_step` to run one gradient update.

    Stage 1 uses diffusion noise prediction loss (L1) with an optional
    Focal Frequency Loss.

    Stage 2 uses a one-step DDIM prediction with an adversarial GAN loss
    (PatchGAN discriminator) in addition to the pixel and frequency losses.

    Parameters
    ----------
    config : E3DiffConfig
        Training configuration.
    """

    def __init__(self, config: E3DiffConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)

        # Build UNet + Diffusion
        unet = E3DiffUNet(
            out_channel=config.out_ch,
            inner_channel=config.inner_channel,
            norm_groups=min(config.inner_channel, 32),
            channel_mults=config.channel_mults,
            attn_res=config.attn_res,
            res_blocks=config.res_blocks,
            dropout=config.dropout,
            image_size=config.image_size,
            condition_ch=config.condition_ch,
        )

        self.diffusion = GaussianDiffusion(
            denoise_fn=unet,
            image_size=config.image_size,
            channels=config.out_ch,
            loss_type="l1",
            conditional=True,
            xT_noise_r=config.xT_noise_r,
        ).to(self.device)

        self.diffusion.set_noise_schedule(
            n_timestep=config.n_timestep,
            schedule=config.beta_schedule,
            linear_start=config.linear_start,
            linear_end=config.linear_end,
            device=self.device,
        )
        _init_weights(self.diffusion.denoise_fn)

        # Frequency loss
        self.freq_loss: FocalFrequencyLoss | None = None
        if config.fft_weight > 0:
            self.freq_loss = FocalFrequencyLoss(loss_weight=config.fft_weight, alpha=1.0).to(self.device)

        # Discriminator (Stage 2 only)
        self.netD: _NLayerDiscriminator | None = None
        self.optimizer_D: torch.optim.Optimizer | None = None
        if config.stage == 2:
            self.netD = _NLayerDiscriminator(
                input_nc=config.condition_ch + config.out_ch,
                ndf=config.ndf,
                n_layers=config.n_layers_D,
            ).to(self.device)
            _init_weights(self.netD)
            self.criterion_GAN = _GANLoss(config.gan_mode).to(self.device)
            self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(), lr=config.lr_D, betas=(config.beta1, 0.999)
            )

        # Generator optimiser
        self.optimizer_G = torch.optim.Adam(
            self.diffusion.parameters(), lr=config.lr, betas=(config.beta1, 0.999)
        )

    # ------------------------------------------------------------------
    # Stage 1: Diffusion noise-prediction training
    # ------------------------------------------------------------------

    def train_step_stage1(
        self,
        sar: torch.Tensor,
        optical: torch.Tensor,
    ) -> dict[str, float]:
        """One Stage-1 gradient update.

        Parameters
        ----------
        sar : Tensor ``[B, condition_ch, H, W]``
            SAR input (conditioning image).
        optical : Tensor ``[B, out_ch, H, W]``
            Ground-truth optical image.

        Returns
        -------
        dict with keys ``"l_pix"`` and (optionally) ``"l_freq"``.
        """
        self.diffusion.train()
        sar = sar.to(self.device)
        optical = optical.to(self.device)
        data = {"HR": optical, "SR": sar}

        self.optimizer_G.zero_grad()
        l_pix, x_start, x_pred = self.diffusion(data, stage=1)

        b, c, h, w = x_start.shape
        l_pix = l_pix / (b * c * h * w)
        loss = l_pix

        log: dict[str, float] = {"l_pix": l_pix.item()}

        if self.freq_loss is not None:
            l_freq = self.freq_loss(x_pred, x_start)
            loss = loss + l_freq
            log["l_freq"] = l_freq.item()

        if not torch.isnan(loss):
            loss.backward()
            self.optimizer_G.step()

        return log

    # ------------------------------------------------------------------
    # Stage 2: One-step DDIM + GAN adversarial fine-tuning
    # ------------------------------------------------------------------

    def train_step_stage2(
        self,
        sar: torch.Tensor,
        optical: torch.Tensor,
    ) -> dict[str, float]:
        """One Stage-2 gradient update (GAN fine-tuning).

        Parameters
        ----------
        sar : Tensor ``[B, condition_ch, H, W]``
            SAR input (conditioning image).
        optical : Tensor ``[B, out_ch, H, W]``
            Ground-truth optical image.

        Returns
        -------
        dict with keys ``"l_pix"``, ``"l_G"`` (GAN generator loss), ``"l_D"``
        (GAN discriminator loss), and optionally ``"l_freq"``.
        """
        if self.netD is None or self.optimizer_D is None:
            raise RuntimeError(
                "Stage-2 training requires a discriminator. "
                "Set stage=2 in E3DiffConfig."
            )
        self.diffusion.train()
        self.netD.train()
        sar = sar.to(self.device)
        optical = optical.to(self.device)
        data = {"HR": optical, "SR": sar}

        # ---- Generator step ----
        self.optimizer_G.zero_grad()
        l_pix, x_start, x_pred = self.diffusion(data, stage=2)

        b, c, h, w = x_start.shape
        l_pix = l_pix / (b * c * h * w)

        fake_input = torch.cat([sar, x_pred], dim=1)
        l_G = self.criterion_GAN(self.netD(fake_input), target_is_real=True) * self.config.lambda_gan
        loss_G = l_pix + l_G

        log: dict[str, float] = {"l_pix": l_pix.item(), "l_G": l_G.item()}

        if self.freq_loss is not None:
            l_freq = self.freq_loss(x_pred, x_start)
            loss_G = loss_G + l_freq
            log["l_freq"] = l_freq.item()

        if not torch.isnan(loss_G):
            loss_G.backward()
            self.optimizer_G.step()

        # ---- Discriminator step ----
        self.optimizer_D.zero_grad()
        real_input = torch.cat([sar, x_start], dim=1)
        fake_input_d = torch.cat([sar, x_pred.detach()], dim=1)
        l_D_real = self.criterion_GAN(self.netD(real_input), target_is_real=True)
        l_D_fake = self.criterion_GAN(self.netD(fake_input_d), target_is_real=False)
        l_D = (l_D_real + l_D_fake) * 0.5
        l_D.backward()
        self.optimizer_D.step()
        log["l_D"] = l_D.item()

        return log

    # ------------------------------------------------------------------
    # Unified training step
    # ------------------------------------------------------------------

    def train_step(
        self,
        sar: torch.Tensor,
        optical: torch.Tensor,
    ) -> dict[str, float]:
        """Run one training step for the configured stage.

        Parameters
        ----------
        sar : Tensor ``[B, condition_ch, H, W]``
        optical : Tensor ``[B, out_ch, H, W]``
        """
        if self.config.stage == 2:
            return self.train_step_stage2(sar, optical)
        return self.train_step_stage1(sar, optical)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        sar: torch.Tensor,
        n_ddim_steps: int = 50,
        eta: float = 0.8,
    ) -> torch.Tensor:
        """Translate a SAR image to optical.

        Parameters
        ----------
        sar : Tensor ``[B, condition_ch, H, W]``
            SAR conditioning image(s) in [-1, 1].
        n_ddim_steps : int
            Number of DDIM denoising steps (1 for the one-step Stage-2 model).
        eta : float
            DDIM stochasticity parameter.

        Returns
        -------
        Tensor ``[B, out_ch, H, W]``
            Predicted optical image(s) in [-1, 1].
        """
        self.diffusion.eval()
        sar = sar.to(self.device)
        result = self.diffusion.sample(sar, n_ddim_steps=n_ddim_steps, eta=eta)
        self.diffusion.train()
        return result
