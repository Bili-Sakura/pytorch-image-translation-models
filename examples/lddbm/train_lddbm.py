# Copyright (c) 2026 EarthBridge Team.
# Credits: Berman et al., NeurIPS 2025 - Bosch Research LDDBM.
# "Towards General Modality Translation with Contrastive and Predictive Latent Diffusion Bridge"

"""LDDBM trainer for latent diffusion bridge (e.g. super-resolution 16→128)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.data.datasets import PairedImageDataset
from src.utils.config_yaml import save_config_yaml
from src.models.lddbm import (
    ModalityTranslationBridge,
    create_bridge,
    create_decoder,
    create_encoder,
)
from src.models.lddbm.names import (
    BridgeModelsTyps,
    Decoders,
    Encoders,
    ReconstructionLoss,
    TrainingStrategy,
)

from .config import LDDBMConfig

logger = logging.getLogger(__name__)


def _make_sr_args() -> SimpleNamespace:
    """Build args for SR task (16→128)."""
    return SimpleNamespace(
        encoder_x_type=Encoders.KlVaePreTrainedEncoder128.value,
        encoder_y_type=Encoders.KlVaePreTrainedEncoder16.value,
        decoder_x_type=Decoders.KlVaePreTrainedDecoder128.value,
        decoder_y_type=Decoders.NoDecoder.value,
        denoiser_type=BridgeModelsTyps.BridgeTransformer.value,
        latent_image_size=8,
        in_channels=64,
        num_of_views=4,
        num_channels_x=1,
        dropout=0.0,
        schedule_sampler="real-uniform",
        pred_mode="ve",
        sigma_data=0.5,
        cov_xy=0.0,
        beta_min=0.1,
        beta_d=2.0,
        sigma_max=80.0,
        sigma_min=0.002,
        weight_schedule="karras",
    )


class LDDBMTrainer:
    """Training harness for LDDBM latent diffusion bridge."""

    def __init__(self, config: LDDBMConfig) -> None:
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        args = _make_sr_args()
        encoder_x = create_encoder(args.encoder_x_type, args)
        encoder_y = create_encoder(args.encoder_y_type, args)
        decoder_x = create_decoder(args.decoder_x_type, args)
        decoder_y = create_decoder(args.decoder_y_type, args)
        bridge = create_bridge(args)

        self.mtb = ModalityTranslationBridge(
            bridge_model=bridge,
            encoder_x=encoder_x,
            encoder_y=encoder_y,
            decoder_x=decoder_x,
            decoder_y=decoder_y,
            rec_loss_type=ReconstructionLoss.Predictive.value,
            clip_loss_w=0.0,
            training_strategy=TrainingStrategy.WholeSystemTraining.value,
            distance_measure_loss="LPIPS",
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.mtb.parameters(),
            lr=config.lr,
            betas=(config.beta1, config.beta2),
        )

    def build_dataset(
        self,
        root_source: str | Path,
        root_target: str | Path,
    ) -> PairedImageDataset:
        cfg = self.config
        tr_src = transforms.Compose([
            transforms.Resize((cfg.resolution_source, cfg.resolution_source)),
            transforms.ToTensor(),
        ])
        tr_tgt = transforms.Compose([
            transforms.Resize((cfg.resolution_target, cfg.resolution_target)),
            transforms.ToTensor(),
        ])
        return PairedImageDataset(
            root_source=root_source,
            root_target=root_target,
            transform_source=tr_src,
            transform_target=tr_tgt,
        )

    def train_step(self, source: torch.Tensor, target: torch.Tensor, step: int) -> dict:
        """Single training step. source=LQ, target=HQ."""
        source = source.to(self.device) * 2 - 1
        target = target.to(self.device) * 2 - 1

        loss_comp = self.mtb(target, source)
        loss_comp["x"] = target

        total_loss, losses = self.mtb.loss(loss_comp, step)

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.mtb.parameters(), 1.0)
        self.optimizer.step()

        return {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()}

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
            self.mtb.train()
            pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{cfg.epochs}")

            for batch in pbar:
                source = batch["source"]
                target = batch["target"]
                logs = self.train_step(source, target, global_step)
                global_step += 1

                if global_step % cfg.log_every == 0:
                    pbar.set_postfix(logs)
                    logger.info("step %d | loss=%.4f diff=%.4f rec=%.4f",
                        global_step, logs["total_loss"], logs["diffusion_loss"],
                        logs.get("reconstruction_loss", 0.0))

            if (epoch + 1) % cfg.save_every == 0:
                self.save_checkpoint(cfg.save_dir, epoch + 1, global_step=global_step)

        logger.info("LDDBM training complete. Checkpoints saved to %s", cfg.save_dir)

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

        for name, comp in [
            ("encoder_x", self.mtb.encoder_x),
            ("encoder_y", self.mtb.encoder_y),
            ("decoder_x", self.mtb.decoder_x),
            ("bridge", self.mtb.bridge_model),
        ]:
            subdir = path / name
            subdir.mkdir(exist_ok=True)
            save_file(comp.state_dict(), subdir / "diffusion_pytorch_model.safetensors")
        save_config_yaml(
            self.config,
            path / "config.yaml",
            extra={"epoch": epoch, "global_step": global_step if global_step is not None else epoch},
        )

        training_state = {
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step if global_step is not None else epoch,
        }
        torch.save(training_state, path / "training_state.pt")
        logger.info("Saved checkpoint to %s (with optimizer state)", path)

    def load_checkpoint(self, path: str | Path) -> dict:
        """Load checkpoint and restore model + optimizer for resume."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        from safetensors.torch import load_file
        for name, comp in [
            ("encoder_x", self.mtb.encoder_x),
            ("encoder_y", self.mtb.encoder_y),
            ("decoder_x", self.mtb.decoder_x),
            ("bridge", self.mtb.bridge_model),
        ]:
            sd_path = path / name / "diffusion_pytorch_model.safetensors"
            if sd_path.exists():
                comp.load_state_dict(load_file(str(sd_path), device=str(self.device)), strict=True)
        train_state_path = path / "training_state.pt"
        if train_state_path.exists():
            ckpt = torch.load(train_state_path, map_location=self.device, weights_only=False)
            self.optimizer.load_state_dict(ckpt["optimizer"])
            return {"epoch": ckpt.get("epoch", 0), "global_step": ckpt.get("global_step", 0)}
        return {"epoch": 0, "global_step": 0}


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="Root dir for source (LQ) images")
    parser.add_argument("--target", type=str, required=True, help="Root dir for target (HQ) images")
    parser.add_argument("--save-dir", type=str, default="./checkpoints/lddbm")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    config = LDDBMConfig(
        save_dir=args.save_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    trainer = LDDBMTrainer(config)
    trainer.train(args.source, args.target)
