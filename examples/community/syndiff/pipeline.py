# Copyright (c) 2026 EarthBridge Team.
# Credits: SynDiff (Özbey et al., IEEE TMI 2023) - https://github.com/icon-lab/SynDiff

"""SynDiff community pipeline for unsupervised medical image translation.

Uses adversarial diffusion models. Self-contained; no external repo required.
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

from .model import NCSNpp, Posterior_Coefficients, get_time_schedule, sample_from_model


@dataclass
class SynDiffPipelineOutput(BaseOutput):
    """Output of the SynDiff community pipeline."""

    images: Union[List[Image.Image], np.ndarray, torch.Tensor]
    nfe: int = 0


class SynDiffPipeline(DiffusionPipeline):
    """Image-to-image translation pipeline using SynDiff (IEEE TMI 2023).

    Unsupervised medical image translation with adversarial diffusion models.
    """

    model_cpu_offload_seq = "generator"

    def __init__(self, generator, coefficients, args, device: torch.device) -> None:
        super().__init__()
        self.register_modules(generator=generator)
        self._coefficients = coefficients
        self._args = args
        self._device = device
        self._n_timesteps = args.num_timesteps

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
        """Convert input images to tensor in [-1, 1]."""
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

        if image.min() >= 0 and image.max() <= 1.0:
            image = (image - 0.5) / 0.5
        elif image.max() > 1.0:
            image = (image / 255.0 - 0.5) / 0.5

        return image.to(device=device, dtype=dtype)

    @torch.no_grad()
    def __call__(
        self,
        source_image: Union[torch.Tensor, Image.Image, List[Image.Image], np.ndarray],
        num_inference_steps: Optional[int] = None,
        output_type: str = "pil",
        return_dict: bool = True,
    ) -> Union[SynDiffPipelineOutput, tuple]:
        """Translate source image to target modality.

        Parameters
        ----------
        source_image : Tensor | PIL.Image | list[PIL.Image] | np.ndarray
            Source modality image(s). Values in [0, 1] or [0, 255].
        num_inference_steps : int, optional
            Number of diffusion steps. Defaults to model's num_timesteps.
        output_type : str
            "pil", "np", or "pt".
        return_dict : bool
            If True, return SynDiffPipelineOutput.

        Returns
        -------
        SynDiffPipelineOutput or tuple
        """
        n_steps = num_inference_steps if num_inference_steps is not None else self._n_timesteps
        device = self.device
        dtype = self.dtype

        source = self.prepare_inputs(source_image, device, dtype)
        b, c, h, w = source.shape
        if c == 3:
            source = source.mean(dim=1, keepdim=True)
        elif c > 2:
            source = source[:, :1]

        x_init = torch.cat(
            (torch.randn(b, 1, h, w, device=device, dtype=dtype), source),
            dim=1,
        )

        sample_from_model = self._sample_fn
        with torch.no_grad():
            images = sample_from_model(
                self._coefficients,
                self.generator,
                n_steps,
                x_init,
                self._args,
            )

        to_range_0_1 = lambda x: (x + 1.0) / 2.0
        images = to_range_0_1(images).clamp(0, 1)

        if output_type == "pil":
            images = self._convert_to_pil(images)
        elif output_type == "np":
            images = self._convert_to_numpy(images)

        if not return_dict:
            return (images, n_steps)
        return SynDiffPipelineOutput(images=images, nfe=n_steps)

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


def load_syndiff_community_pipeline(
    checkpoint_dir: str | Path,
    *,
    exp_name: str = "exp_syndiff",
    which_epoch: int | str = 50,
    direction: str = "contrast1_to_contrast2",
    image_size: int = 256,
    num_channels: int = 2,
    num_channels_dae: int = 64,
    ch_mult: tuple = (1, 1, 2, 2, 4, 4),
    num_timesteps: int = 4,
    num_res_blocks: int = 2,
    nz: int = 100,
    z_emb_dim: int = 256,
    embedding_type: str = "positional",
    use_geometric: bool = False,
    beta_min: float = 0.1,
    beta_max: float = 20.0,
    device: str = "cuda",
) -> SynDiffPipeline:
    """Load SynDiff pipeline from original checkpoint layout.

    Checkpoint layout (from SynDiff training):
    ``<checkpoint_dir>/<exp_name>/gen_diffusive_1_<epoch>.pth`` (contrast2→contrast1)
    ``<checkpoint_dir>/<exp_name>/gen_diffusive_2_<epoch>.pth`` (contrast1→contrast2)

    Parameters
    ----------
    checkpoint_dir : str | Path
        Root directory containing ``<exp_name>/`` with generator checkpoints.
    exp_name : str
        Experiment folder name under checkpoint_dir.
    which_epoch : int | str
        Epoch to load (e.g. 50 or "latest").
    direction : str
        "contrast1_to_contrast2" uses gen_diffusive_1, "contrast2_to_contrast1" uses gen_diffusive_2.
    image_size, num_channels, ... : various
        Architecture args passed to NCSNpp. Must match training config.

    Returns
    -------
    SynDiffPipeline
    """
    class Args:
        pass

    args = Args()
    args.image_size = image_size
    args.num_channels = num_channels
    args.num_channels_dae = num_channels_dae
    args.ch_mult = ch_mult
    args.num_res_blocks = num_res_blocks
    args.nz = nz
    args.z_emb_dim = z_emb_dim
    args.t_emb_dim = 256
    args.n_mlp = 3
    args.attn_resolutions = (16,)
    args.dropout = 0.0
    args.resamp_with_conv = True
    args.conditional = True
    args.fir = True
    args.fir_kernel = [1, 3, 3, 1]
    args.skip_rescale = True
    args.resblock_type = "biggan"
    args.progressive = "none"
    args.progressive_input = "residual"
    args.progressive_combine = "sum"
    args.embedding_type = embedding_type
    args.fourier_scale = 16.0
    args.not_use_tanh = False
    args.centered = True
    args.num_timesteps = num_timesteps
    args.use_geometric = use_geometric
    args.beta_min = beta_min
    args.beta_max = beta_max

    gen = NCSNpp(args).to(device)

    exp_path = Path(checkpoint_dir) / exp_name
    gen_name = (
        "gen_diffusive_2" if direction == "contrast1_to_contrast2" else "gen_diffusive_1"
    )
    ep = str(which_epoch)
    ckpt_file = exp_path / f"{gen_name}_{ep}.pth"
    if not ckpt_file.exists():
        raise FileNotFoundError(
            f"SynDiff checkpoint not found: {ckpt_file}. "
            "Train with original SynDiff or download pretrained from "
            "https://github.com/icon-lab/SynDiff"
        )

    ckpt = torch.load(ckpt_file, map_location=device, weights_only=True)
    if isinstance(ckpt, dict):
        state = {k[7:] if k.startswith("module.") else k: v for k, v in ckpt.items()}
        gen.load_state_dict(state, strict=True)
    gen.eval()

    T = get_time_schedule(args, device)
    pos_coeff = Posterior_Coefficients(args, device)
    sample_fn = lambda coef, g, n, x_init, opt: sample_from_model(coef, g, n, x_init, T, opt)

    pipeline = SynDiffPipeline(gen, pos_coeff, args, torch.device(device))
    pipeline._sample_fn = sample_fn
    return pipeline
