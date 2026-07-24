# Copyright (c) 2026 EarthBridge Team.
# Credits: SPADE from Park et al. "Semantic Image Synthesis with Spatially-Adaptive Normalization" CVPR 2019.

"""SPADE trainer for semantic image synthesis."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.data.datasets import PairedImageDataset
from src.losses import GANLoss, PerceptualLoss
from src.losses.feature_matching import FeatureMatchingLoss, GaussianKLLoss
from src.models.spade import SPADEMultiscaleDiscriminator, SPADEGenerator, SPADEStyleEncoder
from src.utils.config_yaml import save_config_yaml

from .config import SPADEConfig

logger = logging.getLogger(__name__)


def _build_transform(resolution: int):
    return transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
    ])


def _gan_loss_on_multiscale(
    criterion: GANLoss,
    preds: list[list[torch.Tensor]],
    *,
    target_is_real: bool,
    for_discriminator: bool,
) -> torch.Tensor:
    loss = preds[0][-1].new_tensor(0.0)
    for scale_preds in preds:
        loss = loss + criterion(scale_preds[-1], target_is_real=target_is_real, for_discriminator=for_discriminator)
    return loss / len(preds)


class SPADETrainer:
    """Training harness for SPADE (semantic map -> image)."""

    def __init__(self, config: SPADEConfig) -> None:
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        self.generator = SPADEGenerator(
            label_nc=config.label_nc,
            output_nc=config.output_nc,
            ngf=config.ngf,
            crop_size=config.resolution,
            aspect_ratio=config.aspect_ratio,
            num_upsampling_layers=config.num_upsampling_layers,
            use_vae=config.use_vae,
            z_dim=config.z_dim,
            use_spectral_norm=config.use_spectral_norm,
            param_free_norm_type=config.param_free_norm_type,
        ).to(self.device)

        self.style_encoder = None
        if config.use_vae:
            self.style_encoder = SPADEStyleEncoder(
                input_nc=config.output_nc,
                ngf=config.ngf,
                z_dim=config.z_dim,
                crop_size=config.resolution,
                use_spectral_norm=config.use_spectral_norm,
            ).to(self.device)

        self.discriminator = SPADEMultiscaleDiscriminator(
            label_nc=config.label_nc,
            output_nc=config.output_nc,
            ndf=config.ndf,
            n_layers=config.n_layers_D,
            num_D=config.num_D,
            use_spectral_norm=config.use_spectral_norm,
        ).to(self.device)

        self.criterion_gan = GANLoss(config.gan_mode).to(self.device)
        self.criterion_feat = FeatureMatchingLoss().to(self.device)
        self.criterion_perceptual = PerceptualLoss().to(self.device) if config.lambda_perceptual > 0 else None
        self.criterion_kl = GaussianKLLoss().to(self.device) if config.use_vae else None

        g_params = list(self.generator.parameters())
        if self.style_encoder is not None:
            g_params += list(self.style_encoder.parameters())
        self.optimizer_g = torch.optim.Adam(g_params, lr=config.lr_g, betas=(config.beta1, config.beta2))
        self.optimizer_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=config.lr_d,
            betas=(config.beta1, config.beta2),
        )

    def build_dataset(
        self,
        root_source: str | Path,
        root_target: str | Path,
    ) -> PairedImageDataset:
        transform = _build_transform(self.config.resolution)
        return PairedImageDataset(
            root_source=root_source,
            root_target=root_target,
            transform_source=transform,
            transform_target=transform,
        )

    def _generate(self, segmap: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        cfg = self.config
        mu = logvar = None
        z = None
        if cfg.use_vae and self.style_encoder is not None:
            mu, logvar, z = self.style_encoder(target)
        fake = self.generator(segmap, z=z)
        return fake, mu, logvar

    def train_step(self, segmap: torch.Tensor, target: torch.Tensor) -> dict:
        cfg = self.config
        segmap = segmap.to(self.device)
        target = target.to(self.device) * 2 - 1

        # Update D
        self.optimizer_d.zero_grad()
        with torch.no_grad():
            fake, _, _ = self._generate(segmap, target)
        pred_real = self.discriminator(segmap, target)
        pred_fake = self.discriminator(segmap, fake.detach())
        loss_d_real = _gan_loss_on_multiscale(
            self.criterion_gan, pred_real, target_is_real=True, for_discriminator=True,
        )
        loss_d_fake = _gan_loss_on_multiscale(
            self.criterion_gan, pred_fake, target_is_real=False, for_discriminator=True,
        )
        loss_d = (loss_d_real + loss_d_fake) * 0.5 * cfg.lambda_gan
        loss_d.backward()
        self.optimizer_d.step()

        # Update G
        self.optimizer_g.zero_grad()
        fake, mu, logvar = self._generate(segmap, target)
        pred_fake = self.discriminator(segmap, fake)
        pred_real = self.discriminator(segmap, target)
        loss_gan = _gan_loss_on_multiscale(
            self.criterion_gan, pred_fake, target_is_real=True, for_discriminator=False,
        ) * cfg.lambda_gan
        loss_feat = self.criterion_feat(pred_fake, pred_real) * cfg.lambda_feat
        loss_g = loss_gan + loss_feat

        loss_perc = torch.tensor(0.0, device=self.device)
        if self.criterion_perceptual is not None:
            loss_perc = self.criterion_perceptual(fake, target) * cfg.lambda_perceptual
            loss_g = loss_g + loss_perc

        loss_kl = torch.tensor(0.0, device=self.device)
        if self.criterion_kl is not None and mu is not None and logvar is not None:
            loss_kl = self.criterion_kl(mu, logvar) * cfg.lambda_kl
            loss_g = loss_g + loss_kl

        loss_g.backward()
        self.optimizer_g.step()

        return {
            "loss_d": loss_d.item(),
            "loss_gan": loss_gan.item(),
            "loss_feat": loss_feat.item(),
            "loss_perc": loss_perc.item() if torch.is_tensor(loss_perc) else loss_perc,
            "loss_kl": loss_kl.item() if torch.is_tensor(loss_kl) else loss_kl,
        }

    def train(
        self,
        root_source: str | Path,
        root_target: str | Path,
    ) -> None:
        cfg = self.config
        os.makedirs(cfg.save_dir, exist_ok=True)

        dataset = self.build_dataset(root_source, root_target)
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )

        global_step = 0
        for epoch in range(cfg.epochs):
            self.generator.train()
            self.discriminator.train()
            if self.style_encoder is not None:
                self.style_encoder.train()
            pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{cfg.epochs}")

            for batch in pbar:
                logs = self.train_step(batch["source"], batch["target"])
                global_step += 1
                if global_step % cfg.log_every == 0:
                    pbar.set_postfix(logs)
                    logger.info(
                        "step %d | loss_d=%.4f loss_gan=%.4f loss_feat=%.4f",
                        global_step,
                        logs["loss_d"],
                        logs["loss_gan"],
                        logs["loss_feat"],
                    )

            if (epoch + 1) % cfg.save_every == 0:
                self.save_checkpoint(cfg.save_dir, epoch + 1, global_step=global_step)

        logger.info("SPADE training complete. Checkpoints saved to %s", cfg.save_dir)

    def save_checkpoint(
        self,
        save_dir: str,
        epoch: int,
        *,
        global_step: int | None = None,
    ) -> None:
        from safetensors.torch import save_file

        path = Path(save_dir) / f"checkpoint-epoch-{epoch}"
        path.mkdir(parents=True, exist_ok=True)

        gen_path = path / "generator"
        gen_path.mkdir(exist_ok=True)
        self.generator.save_pretrained(gen_path)
        disc_path = path / "discriminator"
        disc_path.mkdir(exist_ok=True)
        save_file(self.discriminator.state_dict(), disc_path / "diffusion_pytorch_model.safetensors")

        if self.style_encoder is not None:
            self.style_encoder.save_pretrained(path / "style_encoder")

        save_config_yaml(
            self.config,
            path / "config.yaml",
            extra={"epoch": epoch, "global_step": global_step if global_step is not None else epoch},
        )

        training_state = {
            "optimizer_g": self.optimizer_g.state_dict(),
            "optimizer_d": self.optimizer_d.state_dict(),
            "epoch": epoch,
            "global_step": global_step if global_step is not None else epoch,
        }
        torch.save(training_state, path / "training_state.pt")
        logger.info("Saved checkpoint to %s (with optimizer state)", path)

    def load_checkpoint(self, path: str | Path) -> dict:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        from safetensors.torch import load_file

        self.generator = SPADEGenerator.from_pretrained(path / "generator").to(self.device)
        disc_sd = load_file(str(path / "discriminator" / "diffusion_pytorch_model.safetensors"), device=str(self.device))
        self.discriminator.load_state_dict(disc_sd, strict=True)

        style_encoder_path = path / "style_encoder"
        if self.style_encoder is not None and style_encoder_path.exists():
            self.style_encoder = SPADEStyleEncoder.from_pretrained(style_encoder_path).to(self.device)

        train_state_path = path / "training_state.pt"
        if train_state_path.exists():
            ckpt = torch.load(train_state_path, map_location=self.device, weights_only=False)
            self.optimizer_g.load_state_dict(ckpt["optimizer_g"])
            self.optimizer_d.load_state_dict(ckpt["optimizer_d"])
            return {"epoch": ckpt.get("epoch", 0), "global_step": ckpt.get("global_step", 0)}
        return {"epoch": 0, "global_step": 0}


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="Semantic map directory")
    parser.add_argument("--target", type=str, required=True, help="Real image directory")
    parser.add_argument("--save-dir", type=str, default="./checkpoints/spade")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--label-nc", type=int, default=35)
    parser.add_argument("--use-vae", action="store_true")
    args = parser.parse_args()
    config = SPADEConfig(
        save_dir=args.save_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        label_nc=args.label_nc,
        use_vae=args.use_vae,
    )
    trainer = SPADETrainer(config)
    trainer.train(args.source, args.target)
