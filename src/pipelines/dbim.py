# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""DBIM inference pipeline for image-to-image translation.

Implements the Diffusion Bridge Implicit Models (DBIM) sampling loop,
which uses the same bridge model family as DDBM but introduces faster
samplers.

Reference
---------
Zheng, K., et al. "Diffusion Bridge Implicit Models." 2024.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from diffusers import DiffusionPipeline
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor

from src.models.unet import DBIMUNet
from src.schedulers.dbim import DBIMScheduler


@dataclass
class DBIMPipelineOutput(BaseOutput):
    """Output of the DBIM pipeline.

    Attributes
    ----------
    images : list[PIL.Image.Image] | np.ndarray | torch.Tensor
        Translated images.
    nfe : int
        Number of function evaluations used.
    sampler : str
        Sampler used (``"dbim"`` or ``"heun"``).
    """

    images: Any
    nfe: int = 0
    sampler: str = "dbim"


class DBIMPipeline(DiffusionPipeline):
    """Image-to-image translation pipeline using DBIM.

    Parameters
    ----------
    unet : torch.nn.Module
        Trained denoising model.
    scheduler : DBIMScheduler
        DBIM noise scheduler.
    """

    model_cpu_offload_seq = "unet"

    def __init__(self, unet: torch.nn.Module, scheduler: DBIMScheduler) -> None:
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
    ) -> "DBIMPipeline":
        """Load DBIM pipeline from local checkpoint directories."""
        unet = DBIMUNet.from_pretrained(pretrained_model_name_or_path, subfolder=subfolder)
        try:
            scheduler = DBIMScheduler.from_pretrained(
                pretrained_model_name_or_path, subfolder=scheduler_subfolder
            )
        except Exception:
            scheduler = DBIMScheduler()
        unet = unet.eval().to(device=device)
        if torch_dtype is not None:
            unet = unet.to(dtype=torch_dtype)
        return cls(unet=unet, scheduler=scheduler)

    @property
    def device(self) -> torch.device:
        unet = self.unet.module if hasattr(self.unet, "module") else self.unet
        return next(unet.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        unet = self.unet.module if hasattr(self.unet, "module") else self.unet
        return next(unet.parameters()).dtype

    @staticmethod
    def _append_dims(x: torch.Tensor, target_dims: int) -> torch.Tensor:
        dims = target_dims - x.ndim
        return x[(...,) + (None,) * dims]

    def _bridge_scalings(
        self, t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """DBIM bridge preconditioning ``(c_skip, c_out, c_in, c_noise)``."""
        a_t, b_t, c_t = self.scheduler.get_abc(t)
        sigma_data = self.scheduler.config.sigma_data

        A = a_t**2 * sigma_data**2 + b_t**2 * sigma_data**2 + c_t**2
        c_in = torch.rsqrt(torch.clamp(A, min=1e-20))
        c_skip = (b_t * sigma_data**2) / torch.clamp(A, min=1e-20)
        c_out = (
            torch.sqrt(torch.clamp(
                a_t**2 * sigma_data**4 + sigma_data**2 * c_t**2, min=1e-20,
            )) * c_in
        )
        c_noise = 1000.0 * 0.25 * torch.log(torch.clamp(t, min=1e-44))
        return c_skip, c_out, c_in, c_noise

    def denoise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        x_T: torch.Tensor,
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        """Predict denoised bridge state from noisy sample at time ``t``."""
        c_skip, c_out, c_in, c_noise = self._bridge_scalings(t)
        c_skip = self._append_dims(c_skip, x_t.ndim).to(dtype=x_t.dtype, device=x_t.device)
        c_out = self._append_dims(c_out, x_t.ndim).to(dtype=x_t.dtype, device=x_t.device)
        c_in = self._append_dims(c_in, x_t.ndim).to(dtype=x_t.dtype, device=x_t.device)

        model_output = self.unet(
            (c_in * x_t).to(dtype=self.dtype, device=self.device),
            c_noise.to(dtype=self.dtype, device=self.device),
            xT=x_T.to(dtype=self.dtype, device=self.device),
        )
        model_output = model_output.to(dtype=x_t.dtype, device=x_t.device)
        denoised = c_out * model_output + c_skip * x_t
        if clip_denoised:
            denoised = denoised.clamp(-1, 1)
        return denoised

    @staticmethod
    def prepare_inputs(image, device, dtype):
        """Convert inputs to ``[-1, 1]`` tensors."""
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
        num_inference_steps: int = 20,
        output_type: str = "pil",
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
        callback: Optional[Callable] = None,
        callback_steps: int = 1,
    ):
        """Run DBIM bridge diffusion sampling.

        Parameters
        ----------
        source_image : Tensor | PIL.Image | list[PIL.Image]
            Source image(s).
        num_inference_steps : int
            Number of diffusion steps.
        output_type : str
            ``"pil"`` | ``"np"`` | ``"pt"``.
        """
        device = self.device
        dtype = self.dtype
        x_T = self.prepare_inputs(source_image, device, dtype)
        batch_size = x_T.shape[0]

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        sigmas = self.scheduler.sigmas
        s_in = x_T.new_ones([batch_size])
        nfe = 0

        # Initialise from bridge marginal at t=sigma_max
        a_0, b_0, c_0 = [
            self._append_dims(v, x_T.ndim) for v in self.scheduler.get_abc(sigmas[0] * s_in)
        ]
        noise = randn_tensor(x_T.shape, generator=generator, device=device, dtype=dtype)
        x = a_0 * x_T + c_0 * noise

        for i in tqdm(range(len(sigmas) - 1), desc="DBIM Sampling"):
            sigma = sigmas[i]
            denoised = self.denoise(x, sigma * s_in, x_T)
            nfe += 1

            result = self.scheduler.step(
                model_output=denoised,
                timestep=i,
                sample=x,
                x_T=x_T,
                generator=generator,
            )
            x = result.prev_sample

            if callback is not None and i % callback_steps == 0:
                callback(i, num_inference_steps, x)

        images = x.clamp(-1, 1)
        if output_type == "pil":
            images = self._convert_to_pil(images)
        elif output_type == "np":
            images = self._convert_to_numpy(images)

        if not return_dict:
            return (images, nfe)
        return DBIMPipelineOutput(images=images, nfe=nfe, sampler="dbim")

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
