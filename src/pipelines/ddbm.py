# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""DDBM inference pipeline for image-to-image translation.

Implements the Denoising Diffusion Bridge Models (DDBM) sampling loop
using Heun's method for high-quality image-to-image translation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Union

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from diffusers import DiffusionPipeline
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor

from src.schedulers.ddbm import DDBMScheduler


@dataclass
class DDBMPipelineOutput(BaseOutput):
    """Output of the DDBM pipeline.

    Attributes
    ----------
    images : list[PIL.Image.Image] | np.ndarray | torch.Tensor
        Translated images in the requested format.
    nfe : int
        Number of function evaluations used.
    """

    images: Any
    nfe: int = 0


class DDBMPipeline(DiffusionPipeline):
    """Image-to-image translation pipeline using DDBM.

    Parameters
    ----------
    unet : torch.nn.Module
        Trained denoising model.
    scheduler : DDBMScheduler
        DDBM noise scheduler.
    """

    model_cpu_offload_seq = "unet"

    def __init__(self, unet: torch.nn.Module, scheduler: DDBMScheduler) -> None:
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)
        self.sigma_data = scheduler.config.sigma_data
        self.sigma_max = scheduler.config.sigma_max
        self.pred_mode = scheduler.config.pred_mode

    @property
    def device(self) -> torch.device:
        return next(self.unet.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.unet.parameters()).dtype

    @staticmethod
    def _append_dims(x: torch.Tensor, target_dims: int) -> torch.Tensor:
        dims = target_dims - x.ndim
        return x[(...,) + (None,) * dims]

    def _get_bridge_scalings(self, sigma: torch.Tensor):
        """Bridge preconditioning scalings ``(c_skip, c_out, c_in)``."""
        sigma_data = self.sigma_data
        if self.pred_mode == "ve":
            A = (sigma**4 / self.sigma_max**4 * sigma_data**2
                 + (1 - sigma**2 / self.sigma_max**2)**2 * sigma_data**2
                 + sigma**2 * (1 - sigma**2 / self.sigma_max**2))
            c_in = 1 / A**0.5
            c_skip = ((1 - sigma**2 / self.sigma_max**2) * sigma_data**2) / A
            c_out = ((sigma / self.sigma_max)**4 * sigma_data**4
                     + sigma_data**2 * sigma**2 * (1 - sigma**2 / self.sigma_max**2))**0.5 * c_in
        elif self.pred_mode.startswith("vp"):
            logsnr_t = self.scheduler._vp_logsnr(sigma)
            logsnr_T = self.scheduler._vp_logsnr(torch.as_tensor(self.sigma_max))
            logs_t = self.scheduler._vp_logs(sigma)
            logs_T = self.scheduler._vp_logs(torch.as_tensor(self.sigma_max))
            a_t = (logsnr_T - logsnr_t + logs_t - logs_T).exp()
            b_t = -torch.expm1(logsnr_T - logsnr_t) * logs_t.exp()
            c_t = -torch.expm1(logsnr_T - logsnr_t) * (2 * logs_t - logsnr_t).exp()
            A = a_t**2 * sigma_data**2 + b_t**2 * sigma_data**2 + c_t
            c_in = 1 / A**0.5
            c_skip = (b_t * sigma_data**2) / A
            c_out = (a_t**2 * sigma_data**4 + sigma_data**2 * c_t)**0.5 * c_in
        else:
            c_in = torch.ones_like(sigma)
            c_out = torch.ones_like(sigma)
            c_skip = torch.zeros_like(sigma)
        return c_skip, c_out, c_in

    def denoise(self, x_t, sigmas, x_T, clip_denoised=True):
        """Denoise using bridge preconditioning."""
        c_skip, c_out, c_in = [
            self._append_dims(x, x_t.ndim) for x in self._get_bridge_scalings(sigmas)
        ]
        rescaled_t = 1000 * 0.25 * torch.log(sigmas + 1e-44)
        model_output = self.unet(
            (c_in * x_t).to(device=self.device, dtype=self.dtype),
            rescaled_t.to(device=self.device, dtype=self.dtype),
            xT=x_T.to(device=self.device, dtype=self.dtype),
        )
        model_output = model_output.to(device=x_t.device, dtype=x_t.dtype)
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
        num_inference_steps: int = 40,
        guidance: float = 1.0,
        churn_step_ratio: float = 0.33,
        generator: Optional[torch.Generator] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback: Optional[Callable] = None,
        callback_steps: int = 1,
    ):
        """Run DDBM bridge diffusion sampling.

        Parameters
        ----------
        source_image : Tensor | PIL.Image | list[PIL.Image]
            Source image(s).
        num_inference_steps : int
            Number of diffusion steps.
        output_type : str
            ``"pil"``, ``"np"``, or ``"pt"``.
        """
        device = self.device
        dtype = self.dtype
        x_T = self.prepare_inputs(source_image, device, dtype)
        batch_size = x_T.shape[0]

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        sigmas = self.scheduler.sigmas
        x = x_T.clone()
        s_in = x.new_ones([batch_size])
        nfe = 0

        for i in tqdm(range(len(sigmas) - 1), desc="DDBM Sampling"):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            sigma_hat = sigma

            denoised = self.denoise(x, sigma_hat * s_in, x_T)
            nfe += 1

            result = self.scheduler.step(denoised, i, x, x_T, guidance=guidance)
            if sigma_next == 0:
                x = result.prev_sample
            else:
                x_2 = result.prev_sample
                denoised_2 = self.denoise(x_2, sigma_next * s_in, x_T)
                nfe += 1
                result2 = self.scheduler.step_heun(denoised, denoised_2, i, x, x_T, guidance=guidance)
                x = result2.prev_sample

            if callback is not None and i % callback_steps == 0:
                callback(i, num_inference_steps, x)

        images = x.clamp(-1, 1)
        if output_type == "pil":
            images = self._convert_to_pil(images)
        elif output_type == "np":
            images = self._convert_to_numpy(images)

        if not return_dict:
            return (images, nfe)
        return DDBMPipelineOutput(images=images, nfe=nfe)

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
