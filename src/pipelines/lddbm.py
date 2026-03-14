# Copyright (c) 2026 EarthBridge Team.
# Credits: Berman et al., NeurIPS 2025 - https://github.com/boschresearch/Multimodal-Distribution-Translation-MDT
# "Towards General Modality Translation with Contrastive and Predictive Latent Diffusion Bridge"

"""LDDBM (Latent Diffusion Bridge Model) inference pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import List, Literal, Union

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

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

logger = logging.getLogger(__name__)

LDDBM_REPO_URL = "https://github.com/boschresearch/Multimodal-Distribution-Translation-MDT"


@dataclass
class LDDBMPipelineOutput:
    """Output container for LDDBM pipeline inference."""

    images: Union[List[Image.Image], np.ndarray, torch.Tensor]


class LDDBMPipeline:
    """Pipeline for LDDBM (Latent Diffusion Bridge Model) inference.

    Uses built-in ModalityTranslationBridge. Translates source (e.g. LR) to target (e.g. HR).
    """

    def __init__(
        self,
        mtb: ModalityTranslationBridge,
        device: str | torch.device = "cpu",
    ) -> None:
        self.mtb = mtb
        self._device = torch.device(device) if isinstance(device, str) else device
        self.mtb.to(self._device)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.mtb.parameters()).dtype

    def _prepare_input(
        self,
        image: Union[torch.Tensor, Image.Image, List[Image.Image], np.ndarray],
    ) -> torch.Tensor:
        """Convert input images to tensor in [-1, 1], BCHW."""
        if isinstance(image, Image.Image):
            image = [image]
        if isinstance(image, list) and image and isinstance(image[0], Image.Image):
            tensors = []
            for img in image:
                arr = np.array(img, dtype=np.float32)
                if arr.max() > 1.0:
                    arr = arr / 255.0
                if arr.ndim == 2:
                    arr = arr[:, :, np.newaxis]
                tensors.append(torch.from_numpy(arr).permute(2, 0, 1))
            image = torch.stack(tensors)
        elif isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim == 4 and image.shape[-1] in (1, 3, 4) and image.shape[1] not in (1, 3, 4):
            image = image.permute(0, 3, 1, 2)
        if image.min() >= 0 and image.max() <= 1.0:
            image = image * 2 - 1
        elif image.max() > 1.0:
            image = image.float() / 255.0 * 2 - 1
        return image.to(device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def __call__(
        self,
        source_image: Union[torch.Tensor, Image.Image, List[Image.Image], np.ndarray],
        num_inference_steps: int = 40,
        output_type: str = "pil",
        return_dict: bool = True,
    ) -> Union[LDDBMPipelineOutput, tuple]:
        """Translate source image to target domain.

        For super-resolution: source_image is the low-resolution input (e.g. 16×16).
        """
        x = self._prepare_input(source_image)
        self.mtb.eval()
        out = self.mtb.sample(x, sampling_steps=num_inference_steps)
        out = out.clamp(-1, 1)
        if output_type == "pil":
            out = self._to_pil(out)
        elif output_type == "np":
            out = ((out + 1) / 2).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
        if not return_dict:
            return (out,)
        return LDDBMPipelineOutput(images=out)

    @staticmethod
    def _to_pil(images: torch.Tensor) -> List[Image.Image]:
        images = (images + 1) / 2
        images = images.clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).round().astype(np.uint8)
        return [Image.fromarray(img) for img in images]


def _make_sr_args(_checkpoint_dir: Path) -> SimpleNamespace:
    """Build a minimal args object for SR task (16→128)."""
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


_COMPONENT_SUBFOLDERS = ("encoder_x", "encoder_y", "decoder_x", "bridge")


def _load_weights(mtb: ModalityTranslationBridge, checkpoint_dir: Path, device: str) -> None:
    """Load weights from safetensors per component subfolder."""
    for name in _COMPONENT_SUBFOLDERS:
        subdir = checkpoint_dir / name
        config_path = subdir / "config.json"
        weights_path = subdir / "diffusion_pytorch_model.safetensors"
        if not (config_path.exists() and weights_path.exists()):
            raise FileNotFoundError(
                f"LDDBM checkpoint missing {name}/config.json or {name}/diffusion_pytorch_model.safetensors in {checkpoint_dir}"
            )
        state_dict = load_file(str(weights_path), device="cpu")
        if name == "bridge":
            mtb.bridge_model.load_state_dict(state_dict, strict=True)
        elif name == "encoder_x":
            mtb.encoder_x.load_state_dict(state_dict, strict=True)
        elif name == "encoder_y":
            mtb.encoder_y.load_state_dict(state_dict, strict=True)
        elif name == "decoder_x":
            mtb.decoder_x.load_state_dict(state_dict, strict=True)
        logger.info("Loaded LDDBM %s from %s", name, weights_path)


def load_lddbm_pipeline(
    checkpoint_dir: str | Path,
    task: Literal["sr_16_to_128"] = "sr_16_to_128",
    *,
    device: str = "cuda",
) -> LDDBMPipeline:
    """Load an LDDBM pipeline from checkpoints.

    Expects project-style layout: encoder_x/, encoder_y/, decoder_x/, bridge/
    each with config.json and diffusion_pytorch_model.safetensors.
    Use ``convert_pt_to_lddbm`` to convert raw .pt checkpoints.

    Parameters
    ----------
    checkpoint_dir : str or Path
        Directory containing subfolders per component.
    task : str
        Currently only ``"sr_16_to_128"`` (super-resolution 16×16 → 128×128).
    device : str
        Device to run inference on.

    Returns
    -------
    LDDBMPipeline
        Pipeline ready for inference.
    """
    ckpt_dir = Path(checkpoint_dir)
    args = _make_sr_args(ckpt_dir)

    encoder_x = create_encoder(args.encoder_x_type, args)
    encoder_y = create_encoder(args.encoder_y_type, args)
    decoder_x = create_decoder(args.decoder_x_type, args)
    decoder_y = create_decoder(args.decoder_y_type, args)
    bridge = create_bridge(args)

    mtb = ModalityTranslationBridge(
        bridge_model=bridge,
        encoder_x=encoder_x,
        encoder_y=encoder_y,
        decoder_x=decoder_x,
        decoder_y=decoder_y,
        rec_loss_type=ReconstructionLoss.Predictive.value,
        clip_loss_w=0.0,
        training_strategy=TrainingStrategy.WholeSystemTraining.value,
        distance_measure_loss="LPIPS",
    )

    _load_weights(mtb, ckpt_dir, device)
    mtb.to(device)
    return LDDBMPipeline(mtb, device=device)
