# Copyright (c) 2025 Copyright holder of the paper "Towards General Modality Translation with Contrastive and Predictive Latent Diffusion Bridge" submitted to NeurIPS 2025 for review.
# All rights reserved.

"""Multimodal Distribution Translator"""

import torch

from src.models.lddbm.names import (
    DistanceMetric,
    ReconstructionLoss,
    TrainingStrategy,
)


def _calculate_clip_loss_stub(z_x, z_y):
    """Optional CLIP loss stub - returns 0 (clip_loss_w is 0 for SR)."""
    return torch.tensor(0.0, device=z_x.device)


def _lpips_loss(x1, x2, lpips_model):
    """LPIPS with [0,1] input - piq expects [0,1]."""
    return lpips_model((x1 + 1) / 2, (x2 + 1) / 2).mean()


class ModalityTranslationBridge(torch.nn.Module):
    def __init__(
        self,
        bridge_model,
        encoder_x,
        encoder_y,
        decoder_x,
        rec_loss_type,
        clip_loss_w,
        training_strategy,
        distance_measure_loss,
        decoder_y=None,
    ):
        super().__init__()
        self.bridge_model = bridge_model
        self.encoder_x = encoder_x
        self.encoder_y = encoder_y
        self.decoder_x = decoder_x
        self.decoder_y = decoder_y
        self.rec_loss_type = rec_loss_type
        self.clip_loss_w = clip_loss_w
        self.training_strategy = training_strategy
        self.rec_distance_loss = self.get_loss(distance_measure_loss)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        z_x = self.encoder_x(x)
        z_y = self.encoder_y(y)

        z_x_bridge = self.bridge_model(z_x, z_y)

        x_hat = self.decoder_x(z_x)
        x_bridge_hat = self.decoder_x(z_x_bridge)
        y_hat = self.decoder_y(z_y) if self.decoder_y is not None else None

        return {
            "z_x": z_x,
            "z_y": z_y,
            "z_x_bridge": z_x_bridge,
            "x_hat": x_hat,
            "x_bridge_hat": x_bridge_hat,
            "y": y,
            "y_hat": y_hat,
        }

    def sample(self, y: torch.Tensor, sampling_steps=40):
        z_y = self.encoder_y(y)
        z_x_hat = self.bridge_model.sample(z_y, sampling_steps)
        x_hat = self.decoder_x(z_x_hat)

        return x_hat

    def loss(self, loss_comp, step):

        # --- bridge loss --- #
        losses = {
            "diffusion_loss": ((loss_comp["z_x"] - loss_comp["z_x_bridge"]) ** 2).mean()
        }

        if self.iterative_training_coordinator(step):
            # --- reconstruction loss --- #
            if self.rec_loss_type == ReconstructionLoss.Reconstruction.value:
                losses["reconstruction_loss"] = self.rec_distance_loss(
                    loss_comp["x_hat"], loss_comp["x"]
                )

            elif self.rec_loss_type == ReconstructionLoss.Predictive.value:
                losses["reconstruction_loss"] = self.rec_distance_loss(
                    loss_comp["x_bridge_hat"], loss_comp["x"]
                )

            # --- clip loss --- #
            if self.clip_loss_w > 0:
                losses["clip_loss"] = self.clip_loss_w * _calculate_clip_loss_stub(
                    loss_comp["z_x"], loss_comp["z_y"]
                )

        # --- final loss --- #
        total_loss = 0
        for v in losses.values():
            total_loss = total_loss + v

        losses["total_loss"] = total_loss

        return total_loss, losses

    def iterative_training_coordinator(self, step):
        if self.training_strategy == TrainingStrategy.WholeSystemTraining.value:
            return True  # if its regular training it always should enter
        elif self.training_strategy == TrainingStrategy.IterativeTraining.value:
            return step % 4 == 0
        else:
            raise NotImplementedError(
                f"No such training strategy {self.training_strategy}"
            )

    def get_loss(self, distance_metric):
        if distance_metric == DistanceMetric.MSE.value:
            return lambda x1, x2: ((x1 - x2) ** 2).mean()
        elif distance_metric == DistanceMetric.LPIPS.value:
            try:
                from piq import LPIPS
            except ImportError:
                raise ImportError(
                    "piq is required for LPIPS. Install with: pip install piq"
                )
            self._lpips = LPIPS(replace_pooling=True)
            return lambda x1, x2: _lpips_loss(x1, x2, self._lpips)
        else:
            raise NotImplementedError(f"No such distance metrics {distance_metric}")
