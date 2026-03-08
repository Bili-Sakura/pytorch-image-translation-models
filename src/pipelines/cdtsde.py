# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""CDTSDE inference pipeline for image-to-image translation.

Implements the CDTSDE (Adaptive Domain Shift Diffusion Bridge) sampling loop
for image-to-image translation with dynamic domain-shift scheduling.
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
from diffusers.utils.torch_utils import randn_tensor

from src.models.unet.diffusers_wrappers import CDTSDEUNet
from src.schedulers.cdtsde import CDTSDEScheduler


@dataclass
class CDTSDEPipelineOutput(BaseOutput):
    """Output of the CDTSDE pipeline.

    Attributes
    ----------
    images : list[PIL.Image.Image] | np.ndarray | torch.Tensor
        Translated images.
    nfe : int
        Number of function evaluations used.
    """

    images: Any
    nfe: int = 0


class CDTSDEPipeline(DiffusionPipeline):
    """Image-to-image translation pipeline using CDTSDE.

    Parameters
    ----------
    unet : torch.nn.Module
        Trained denoising model (must have ``predict_lambda`` method
        for domain-shift mode).
    scheduler : CDTSDEScheduler
        CDTSDE noise scheduler.
    """

    model_cpu_offload_seq = "unet"

    def __init__(self, unet: torch.nn.Module, scheduler: CDTSDEScheduler) -> None:
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
    ) -> "CDTSDEPipeline":
        """Load CDTSDE pipeline from local checkpoint directories."""
        unet = CDTSDEUNet.from_pretrained(pretrained_model_name_or_path, subfolder=subfolder)
        try:
            scheduler = CDTSDEScheduler.from_pretrained(
                pretrained_model_name_or_path, subfolder=scheduler_subfolder
            )
        except Exception:
            scheduler = CDTSDEScheduler()
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
        if isinstance(image, list) and image and isinstance(image[0], Image.Image):
            images = []
            for img in image:
                if img.mode not in ("L", "RGB"):
                    img = img.convert("RGB")
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
        if image.min() >= 0.0:
            image = image * 2.0 - 1.0
        return image.to(device=device, dtype=dtype)

    @torch.no_grad()
    def __call__(
        self,
        source_image: Union[torch.Tensor, Image.Image, List[Image.Image]],
        num_inference_steps: int = 50,
        stochastic: bool = True,
        apply_domain_shift: bool = True,
        generator: Optional[torch.Generator] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback: Optional[Callable] = None,
        callback_steps: int = 1,
    ):
        """Run CDTSDE sampling.

        Parameters
        ----------
        source_image : Tensor | PIL.Image | list[PIL.Image]
            Source image(s).
        num_inference_steps : int
            Number of diffusion steps.
        stochastic : bool
            Use stochastic sampling.
        apply_domain_shift : bool
            Apply the CDTSDE domain-shift mechanism.
        output_type : str
            ``"pil"`` | ``"np"`` | ``"pt"``.
        """
        device = self.device
        dtype = self.dtype
        x_T = self.prepare_inputs(source_image, device=device, dtype=dtype)
        batch_size = x_T.shape[0]

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        if self.scheduler.timesteps is None:
            raise ValueError("Scheduler timesteps not initialized.")

        sqrt_alpha_last = self.scheduler.sqrt_alphas_cumprod[-1].to(device=device, dtype=dtype)
        sigma_last = self.scheduler.sigmas[-1].to(device=device, dtype=dtype)
        noise = randn_tensor(x_T.shape, generator=generator, device=device, dtype=dtype)
        x = sqrt_alpha_last * x_T + sigma_last * noise

        nfe = 0
        total_steps = len(self.scheduler.timesteps) - 1
        progress = tqdm(range(total_steps), desc="CDTSDE Sampling")

        for i in progress:
            j = total_steps - i
            t_model = self.scheduler.timesteps[j]
            t_batch = torch.full((batch_size,), t_model, device=device, dtype=torch.long)

            pred_noise = self.unet(
                x.to(device=device, dtype=dtype),
                t_batch,
                xT=x_T.to(device=device, dtype=dtype),
            )
            pred_noise = pred_noise.to(device=x.device, dtype=x.dtype)
            nfe += 1

            idx_batch = torch.full((batch_size,), j, device=device, dtype=torch.long)
            pred_x0 = self.scheduler.predict_start_from_noise(
                sample=x, timesteps=idx_batch, noise=pred_noise,
                use_inference_schedule=True,
            )

            if apply_domain_shift and hasattr(self.unet, "predict_lambda"):
                lam_linear = self.scheduler.etas[j].to(device=device, dtype=dtype)
                lam_batch = torch.full(
                    (batch_size,), lam_linear, device=device, dtype=dtype,
                )
                lam_hat = self.unet.predict_lambda(lam_batch, pred_x0.shape)
                pred_x0 = lam_hat * x_T + (1.0 - lam_hat) * pred_x0

            step_out = self.scheduler.step(
                pred_original_sample=pred_x0,
                step_index=j,
                sample=x,
                reference_sample=x_T,
                generator=generator if isinstance(generator, torch.Generator) else None,
                stochastic=stochastic,
                return_dict=True,
            )
            x = step_out.prev_sample

            if callback is not None and i % callback_steps == 0:
                callback(i, total_steps, x)

        images = x.clamp(-1.0, 1.0)
        if output_type == "pil":
            images = self._convert_to_pil(images)
        elif output_type == "np":
            images = self._convert_to_numpy(images)

        if not return_dict:
            return (images, nfe)
        return CDTSDEPipelineOutput(images=images, nfe=nfe)

    @staticmethod
    def _convert_to_pil(images: torch.Tensor) -> List[Image.Image]:
        images = (images + 1.0) / 2.0
        images = images.clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).round().astype(np.uint8)
        return [
            Image.fromarray(img.squeeze(-1) if img.shape[-1] == 1 else img)
            for img in images
        ]

    @staticmethod
    def _convert_to_numpy(images: torch.Tensor) -> np.ndarray:
        images = (images + 1.0) / 2.0
        return images.clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
