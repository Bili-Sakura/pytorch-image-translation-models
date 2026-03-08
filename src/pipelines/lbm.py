# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""LBM pixel-space inference pipeline for image-to-image translation.

Implements the LBM (Latent Bridge Matching) flow-matching sampling loop
for single-step or few-step image-to-image translation.

Reference: https://arxiv.org/abs/2503.07535
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from diffusers import DiffusionPipeline
from diffusers.utils import BaseOutput

from src.models.unet.diffusers_wrappers import LBMUNet
from src.schedulers.lbm import LBMScheduler


@dataclass
class LBMPipelineOutput(BaseOutput):
    """Output of the LBM pipeline.

    Attributes
    ----------
    images : list[PIL.Image.Image] | np.ndarray | torch.Tensor
        Translated images.
    nfe : int
        Number of function evaluations used.
    """

    images: Any
    nfe: int = 0


class LBMPipeline(DiffusionPipeline):
    """Image-to-image translation pipeline using LBM flow-matching.

    Parameters
    ----------
    unet : torch.nn.Module
        A UNet (or DiT) model for denoising.
    scheduler : LBMScheduler
        An LBM scheduler for the bridge flow-matching process.
    """

    model_cpu_offload_seq = "unet"

    def __init__(self, unet: torch.nn.Module, scheduler: LBMScheduler) -> None:
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        subfolder: str = "unet",
        scheduler_subfolder: str = "scheduler",
        device: str | torch.device = "cpu",
        torch_dtype: torch.dtype | None = None,
        **kwargs,
    ) -> "LBMPipeline":
        """Load LBM pipeline from local checkpoint directories."""
        unet = LBMUNet.from_pretrained(pretrained_model_name_or_path, subfolder=subfolder)
        try:
            scheduler = LBMScheduler.from_pretrained(
                pretrained_model_name_or_path, subfolder=scheduler_subfolder
            )
        except Exception:
            scheduler = LBMScheduler()
        unet = unet.eval().to(device=device)
        if torch_dtype is not None:
            unet = unet.to(dtype=torch_dtype)
        return cls(unet=unet, scheduler=scheduler)

    @property
    def device(self) -> torch.device:
        return next(self.unet.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.unet.parameters()).dtype

    @staticmethod
    def prepare_inputs(image, device, dtype):
        """Convert inputs to ``[-1, 1]`` BCHW tensors."""
        if isinstance(image, Image.Image):
            image = [image]
        if isinstance(image, list) and isinstance(image[0], Image.Image):
            images = []
            for img in image:
                img = img.convert("RGB")
                arr = np.array(img).astype(np.float32) / 255.0
                images.append(torch.from_numpy(arr).permute(2, 0, 1))
            image = torch.stack(images)
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        if image.max() > 1.0:
            image = image / 255.0
        if image.min() >= 0:
            image = image * 2 - 1
        return image.to(device=device, dtype=dtype)

    @torch.no_grad()
    def __call__(
        self,
        source_image: Union[torch.Tensor, Image.Image, List[Image.Image]],
        num_inference_steps: int = 1,
        cfg_scale: float = 1.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
    ):
        """Generate images using LBM bridge flow-matching.

        Parameters
        ----------
        source_image : Tensor | PIL.Image | list[PIL.Image]
            Source/condition image(s).
        num_inference_steps : int
            Number of Euler sampling steps (default: 1).
        cfg_scale : float
            Classifier-free guidance scale (default: 1.0, disables CFG).
        output_type : str
            ``"pil"`` | ``"np"`` | ``"pt"``.
        """
        device = self.device
        dtype = self.dtype

        x_source = self.prepare_inputs(source_image, device, dtype)
        batch_size = x_source.shape[0]

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        sample = x_source.clone()

        has_condition = hasattr(self.unet, "condition_mode") and self.unet.condition_mode == "concat"
        cond = x_source if has_condition else None
        use_cfg = has_condition and abs(float(cfg_scale) - 1.0) > 1e-6
        null_condition = torch.zeros_like(cond) if use_cfg else None
        nfe_per_step = 2 if use_cfg else 1
        nfe_count = 0

        progress_bar = tqdm(
            enumerate(self.scheduler.timesteps),
            total=len(self.scheduler.timesteps),
            desc="LBM Sampling",
        )

        for i, t in progress_bar:
            t_batch = t.to(device).repeat(batch_size) if isinstance(t, torch.Tensor) else torch.full(
                (batch_size,), t, device=device, dtype=torch.long,
            )

            if use_cfg:
                model_input = torch.cat([sample, sample], dim=0)
                timestep_input = torch.cat([t_batch, t_batch], dim=0)
                cond_input = torch.cat([cond, null_condition], dim=0)
                pred_batched = self.unet(model_input, timestep_input, cond=cond_input)
                pred_cond, pred_uncond = pred_batched.chunk(2, dim=0)
                pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
            else:
                pred = self.unet(sample, t_batch, cond=cond if cond is not None else None)
            nfe_count += nfe_per_step

            result = self.scheduler.step(pred, t, sample, return_dict=True)
            sample = result.prev_sample

            if i < len(self.scheduler.timesteps) - 1:
                next_t = self.scheduler.timesteps[i + 1]
                next_t_batch = next_t.to(device).repeat(batch_size)
                next_sigmas = self.scheduler.get_sigmas(next_t_batch, n_dim=sample.ndim, device=device, dtype=dtype)
                bridge_noise = self.scheduler.bridge_noise_sigma
                sample = sample + bridge_noise * (next_sigmas * (1.0 - next_sigmas)) ** 0.5 * torch.randn_like(sample)

            if callback is not None and i % callback_steps == 0:
                callback(i, len(self.scheduler.timesteps), sample)

        images = sample.clamp(-1, 1)
        if output_type == "pil":
            images = self._convert_to_pil(images)
        elif output_type == "np":
            images = self._convert_to_numpy(images)

        if not return_dict:
            return (images, nfe_count)
        return LBMPipelineOutput(images=images, nfe=nfe_count)

    @staticmethod
    def _convert_to_pil(images: torch.Tensor) -> List[Image.Image]:
        images = (images + 1) / 2
        images = images.clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).round().astype(np.uint8)
        pil_images = []
        for img in images:
            if img.shape[2] == 1:
                img = img.squeeze(2)
            pil_images.append(Image.fromarray(img))
        return pil_images

    @staticmethod
    def _convert_to_numpy(images: torch.Tensor) -> np.ndarray:
        images = (images + 1) / 2
        return images.clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
