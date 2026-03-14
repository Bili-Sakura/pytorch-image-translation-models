# Copyright (c) 2026 EarthBridge Team.
# Credits: SelfRDB (Arslan et al., Medical Image Analysis 2024) - https://github.com/icon-lab/SelfRDB

"""SelfRDB community pipeline for medical image translation.

Self-Consistent Recursive Diffusion Bridge. Self-contained; no external repo required.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image

from diffusers import DiffusionPipeline
from diffusers.utils import BaseOutput

from .model import NCSNpp
from .diffusion import DiffusionBridge


@dataclass
class SelfRDBPipelineOutput(BaseOutput):
    """Output of the SelfRDB community pipeline."""

    images: Union[List[Image.Image], np.ndarray, torch.Tensor]
    nfe: int = 0


class SelfRDBPipeline(DiffusionPipeline):
    """Image-to-image translation pipeline using SelfRDB (Medical Image Analysis 2024).

    Self-Consistent Recursive Diffusion Bridge for multi-modal medical image synthesis.
    """

    model_cpu_offload_seq = "generator"

    def __init__(self, generator, diffusion, device: torch.device) -> None:
        super().__init__()
        self.register_modules(generator=generator)
        self._diffusion = diffusion
        self._device = device

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.generator.parameters()).dtype

    def prepare_inputs(
        self,
        image: Union[torch.Tensor, Image.Image, List[Image.Image], np.ndarray],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Convert input images to tensor in [0, 1] (SelfRDB uses [0,1] internally)."""
        if isinstance(image, Image.Image):
            image = [image]

        if isinstance(image, list) and image and isinstance(image[0], Image.Image):
            images = []
            for img in image:
                img_array = np.array(img, dtype=np.float32)
                if img_array.max() > 1.0:
                    img_array = img_array / 255.0
                if img_array.ndim == 2:
                    img_array = img_array[:, :, np.newaxis]
                images.append(torch.from_numpy(img_array).permute(2, 0, 1))
            image = torch.stack(images)

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)

        if image.ndim == 3:
            image = image.unsqueeze(0)

        if image.shape[1] not in (1, 2, 3) and image.shape[-1] in (1, 2, 3):
            image = image.permute(0, 3, 1, 2)

        if image.max() > 1.0:
            image = image / 255.0
        image = image.clamp(0, 1)

        return image.to(device=device, dtype=dtype)

    @torch.no_grad()
    def __call__(
        self,
        source_image: Union[torch.Tensor, Image.Image, List[Image.Image], np.ndarray],
        output_type: str = "pil",
        return_dict: bool = True,
    ) -> Union[SelfRDBPipelineOutput, tuple]:
        """Translate source image to target modality.

        Parameters
        ----------
        source_image : Tensor | PIL.Image | list[PIL.Image] | np.ndarray
            Source modality image(s). Values in [0, 1] or [0, 255].
        output_type : str
            "pil", "np", or "pt".
        return_dict : bool
            If True, return SelfRDBPipelineOutput.

        Returns
        -------
        SelfRDBPipelineOutput or tuple
        """
        device = self.device
        dtype = self.dtype

        source = self.prepare_inputs(source_image, device, dtype)
        b, c, h, w = source.shape
        if c == 3:
            source = source.mean(dim=1, keepdim=True)
        elif c > 2:
            source = source[:, :1]
        # SelfRDB expects y (source) as [B, 1, H, W] for grayscale medical images

        self.generator.eval()
        images = self._diffusion.sample_x0(source, self.generator)
        nfe = self._diffusion.n_steps

        if output_type == "pil":
            images = self._convert_to_pil(images)
        elif output_type == "np":
            images = self._convert_to_numpy(images)

        if not return_dict:
            return (images, nfe)
        return SelfRDBPipelineOutput(images=images, nfe=nfe)

    @staticmethod
    def _convert_to_pil(images: torch.Tensor) -> List[Image.Image]:
        images = images.clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).round().astype(np.uint8)
        pil_images = []
        for img in images:
            if img.shape[2] == 1:
                pil_images.append(Image.fromarray(img.squeeze(2), mode="L"))
            else:
                pil_images.append(Image.fromarray(img))
        return pil_images

    @staticmethod
    def _convert_to_numpy(images: torch.Tensor) -> np.ndarray:
        return images.clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()


def load_selfrdb_community_pipeline(
    checkpoint_path: str | Path,
    *,
    device: str = "cuda",
) -> SelfRDBPipeline:
    """Load SelfRDB pipeline from Lightning checkpoint.

    Checkpoint is a .ckpt file from SelfRDB training or from the
    [Model Zoo](https://github.com/icon-lab/SelfRDB#-model-zoo).

    Parameters
    ----------
    checkpoint_path : str | Path
        Path to .ckpt file (e.g. ixi_t1_t2.ckpt).
    device : str
        Device to load the model on.

    Returns
    -------
    SelfRDBPipeline
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"SelfRDB checkpoint not found: {ckpt_path}. "
            "Download from https://github.com/icon-lab/SelfRDB/releases"
        )

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    hparams = ckpt.get("hyper_parameters", {})
    if isinstance(hparams, dict):
        gen_params = hparams.get("generator_params", {})
        diff_params = hparams.get("diffusion_params", {})
    else:
        gen_params = getattr(hparams, "generator_params", {}) or {}
        diff_params = getattr(hparams, "diffusion_params", {}) or {}

    if not gen_params:
        if "model" in ckpt.get("hyper_parameters", {}):
            hp = ckpt["hyper_parameters"]["model"]
            gen_params = hp.get("generator_params", {}) or hp.get("model", {}).get("generator_params", {})
            diff_params = hp.get("diffusion_params", {}) or hp.get("model", {}).get("diffusion_params", {})

    defaults = {
        "self_recursion": True,
        "image_size": 256,
        "z_emb_dim": 256,
        "ch_mult": [1, 1, 2, 2, 4, 4],
        "num_res_blocks": 2,
        "attn_resolutions": [16],
        "dropout": 0.0,
        "resamp_with_conv": True,
        "conditional": True,
        "fir": True,
        "fir_kernel": [1, 3, 3, 1],
        "skip_rescale": True,
        "resblock_type": "biggan",
        "progressive": "none",
        "progressive_input": "residual",
        "embedding_type": "positional",
        "combine_method": "sum",
        "fourier_scale": 16,
        "nf": 64,
        "num_channels": 2,
        "nz": 100,
        "n_mlp": 3,
        "centered": True,
        "not_use_tanh": False,
    }
    for k, v in defaults.items():
        gen_params.setdefault(k, v)

    diff_defaults = {
        "n_steps": 10,
        "beta_start": 0.1,
        "beta_end": 3.0,
        "gamma": 1,
        "n_recursions": 2,
        "consistency_threshold": 0.01,
    }
    for k, v in diff_defaults.items():
        diff_params.setdefault(k, v)

    generator = NCSNpp(**gen_params).to(device)
    diffusion = DiffusionBridge(**diff_params)

    state = ckpt.get("state_dict", ckpt)
    if isinstance(state, dict):
        gen_state = {k[10:]: v for k, v in state.items() if k.startswith("generator.")}
        if not gen_state:
            gen_state = state
        generator.load_state_dict(gen_state, strict=False)
    generator.eval()

    return SelfRDBPipeline(generator, diffusion, torch.device(device))
