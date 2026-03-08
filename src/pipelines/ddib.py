# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""DDIB inference pipeline for image-to-image translation.

DDIB (Dual Diffusion Implicit Bridges) translates between two domains by
concatenating a source-to-latent DDIM reverse ODE with a latent-to-target
DDIM forward ODE, using two independently trained diffusion models.

Reference
---------
Su, Xuan, et al. "Dual Diffusion Implicit Bridges for Image-to-Image
Translation." ICLR 2023.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
import torch
from PIL import Image

from diffusers import DiffusionPipeline
from diffusers.utils import BaseOutput

from src.models.unet.diffusers_wrappers import DDIBUNet
from src.schedulers.ddib import DDIBScheduler


@dataclass
class DDIBPipelineOutput(BaseOutput):
    """Output of the DDIB pipeline.

    Attributes
    ----------
    images : list[PIL.Image.Image] | np.ndarray | torch.Tensor
        Translated images.
    latent : torch.Tensor or None
        Shared latent representation (optional).
    """

    images: Any
    latent: Optional[torch.Tensor] = None


class DDIBPipeline(DiffusionPipeline):
    """Image-to-image translation pipeline using DDIB.

    Uses two independently trained diffusion models: one for the source
    domain and one for the target domain.

    Parameters
    ----------
    source_unet : torch.nn.Module
        Diffusion model trained on the **source** domain.
    target_unet : torch.nn.Module
        Diffusion model trained on the **target** domain.
    scheduler : DDIBScheduler
        Shared DDIB scheduler.
    """

    model_cpu_offload_seq = "source_unet->target_unet"

    def __init__(
        self,
        source_unet: torch.nn.Module,
        target_unet: torch.nn.Module,
        scheduler: DDIBScheduler,
    ) -> None:
        super().__init__()
        self.register_modules(
            source_unet=source_unet,
            target_unet=target_unet,
            scheduler=scheduler,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        source_subfolder: str = "source_unet",
        target_subfolder: str = "target_unet",
        scheduler_subfolder: str = "scheduler",
        device: str | torch.device = "cpu",
        torch_dtype: torch.dtype | None = None,
        **kwargs,
    ) -> "DDIBPipeline":
        """Load DDIB source/target UNets and scheduler from local folders."""
        source_unet = DDIBUNet.from_pretrained(
            pretrained_model_name_or_path, subfolder=source_subfolder
        )
        target_unet = DDIBUNet.from_pretrained(
            pretrained_model_name_or_path, subfolder=target_subfolder
        )
        try:
            scheduler = DDIBScheduler.from_pretrained(
                pretrained_model_name_or_path, subfolder=scheduler_subfolder
            )
        except Exception:
            scheduler = DDIBScheduler()

        source_unet = source_unet.eval().to(device=device)
        target_unet = target_unet.eval().to(device=device)
        if torch_dtype is not None:
            source_unet = source_unet.to(dtype=torch_dtype)
            target_unet = target_unet.to(dtype=torch_dtype)
        return cls(source_unet=source_unet, target_unet=target_unet, scheduler=scheduler)

    @property
    def device(self) -> torch.device:
        return next(self.target_unet.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.target_unet.parameters()).dtype

    # ------------------------------------------------------------------
    # DDIM reverse loop (encode: x_0 → x_T using source model)
    # ------------------------------------------------------------------

    def _ddim_reverse_sample_loop(
        self,
        model: torch.nn.Module,
        x_0: torch.Tensor,
        timesteps: torch.Tensor,
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        """Encode ``x_0`` into latent ``x_T`` via DDIM reverse."""
        x = x_0
        for i in range(len(timesteps) - 1):
            t_cur = timesteps[i]
            t_next = timesteps[i + 1]
            t_batch = t_cur.expand(x.shape[0])
            scaled_t = self.scheduler._scale_timesteps(t_batch)
            model_output = model(x, scaled_t)
            out = self.scheduler.ddim_reverse_step(
                model_output=model_output,
                timestep=t_batch,
                timestep_next=t_next.expand(x.shape[0]),
                sample=x,
                clip_denoised=clip_denoised,
            )
            x = out.prev_sample
        return x

    # ------------------------------------------------------------------
    # DDIM forward loop (decode: x_T → x_0 using target model)
    # ------------------------------------------------------------------

    def _ddim_sample_loop(
        self,
        model: torch.nn.Module,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        clip_denoised: bool = True,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """Decode latent ``x_T`` into ``x_0`` via DDIM forward."""
        x = noise
        reversed_timesteps = timesteps.flip(0)
        for i in range(len(reversed_timesteps) - 1):
            t_cur = reversed_timesteps[i]
            t_prev = reversed_timesteps[i + 1]
            t_batch = t_cur.expand(x.shape[0])
            scaled_t = self.scheduler._scale_timesteps(t_batch)
            model_output = model(x, scaled_t)
            out = self.scheduler.ddim_step(
                model_output=model_output,
                timestep=t_batch,
                timestep_prev=t_prev.expand(x.shape[0]),
                sample=x,
                eta=eta,
                clip_denoised=clip_denoised,
            )
            x = out.prev_sample
        return x

    # ------------------------------------------------------------------
    # Input helpers
    # ------------------------------------------------------------------

    @staticmethod
    def prepare_inputs(image, device, dtype):
        """Convert inputs to ``[-1, 1]`` tensors."""
        if isinstance(image, Image.Image):
            image = [image]
        if isinstance(image, list) and isinstance(image[0], Image.Image):
            images = []
            for img in image:
                arr = np.array(img).astype(np.float32) / 255.0
                if arr.ndim == 2:
                    t = torch.from_numpy(arr).unsqueeze(0)
                else:
                    t = torch.from_numpy(arr).permute(2, 0, 1)
                images.append(t)
            image = torch.stack(images)
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        if image.max() > 1.0:
            image = image / 255.0
        if image.min() >= 0:
            image = image * 2 - 1
        return image.to(device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # __call__
    # ------------------------------------------------------------------

    @torch.no_grad()
    def __call__(
        self,
        source_image: Union[torch.Tensor, Image.Image, List[Image.Image]],
        num_inference_steps: int = 250,
        clip_denoised: bool = True,
        eta: float = 0.0,
        output_type: str = "pil",
        return_dict: bool = True,
        return_latent: bool = False,
    ):
        """Translate *source_image* from source domain to target domain.

        Parameters
        ----------
        source_image : Tensor | PIL.Image | list[PIL.Image]
            Source image(s).
        num_inference_steps : int
            Number of DDIM steps for encode and decode.
        output_type : str
            ``"pil"`` | ``"np"`` | ``"pt"``.
        """
        device = self.device
        dtype = self.dtype
        x_source = self.prepare_inputs(source_image, device, dtype)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 1. Encode: source → latent
        latent = self._ddim_reverse_sample_loop(
            self.source_unet, x_source, timesteps, clip_denoised=clip_denoised,
        )

        # 2. Decode: latent → target
        images = self._ddim_sample_loop(
            self.target_unet, latent, timesteps, clip_denoised=clip_denoised, eta=eta,
        )
        images = images.clamp(-1, 1)

        if output_type == "pil":
            images = self._convert_to_pil(images)
        elif output_type == "np":
            images = self._convert_to_numpy(images)

        if not return_dict:
            return (images,)

        return DDIBPipelineOutput(
            images=images,
            latent=latent if return_latent else None,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_to_pil(images: torch.Tensor) -> List[Image.Image]:
        images = (images + 1) / 2
        images = images.clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).round().astype(np.uint8)
        return [Image.fromarray(img) for img in images]

    @staticmethod
    def _convert_to_numpy(images: torch.Tensor) -> np.ndarray:
        images = (images + 1) / 2
        return images.clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
