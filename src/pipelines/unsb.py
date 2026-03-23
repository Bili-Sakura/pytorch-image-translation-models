# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""UNSB multi-step inference pipeline.

Provides :class:`UNSBPipeline` for running inference with a trained UNSB
generator, performing multi-step stochastic refinement through the
Schrödinger Bridge dynamics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from PIL import Image

from diffusers.utils import pt_to_pil

from src.models.unsb import UNSBGenerator, create_generator
from src.schedulers.unsb import UNSBScheduler


@dataclass
class UNSBPipelineOutput:
    """Output class for UNSB pipeline.

    Attributes
    ----------
    images : list of PIL.Image.Image, np.ndarray, or torch.Tensor
        Generated images after multi-step translation.
    nfe : int
        Number of function evaluations (generator calls).
    """

    images: Union[List[Image.Image], np.ndarray, torch.Tensor]
    nfe: int


class UNSBPipeline:
    """Multi-step inference pipeline for Unpaired Neural Schrödinger Bridge.

    Unlike single-pass GAN pipelines, UNSB performs image translation over
    multiple stochastic refinement steps.  At each step the generator maps
    the current noisy state to a refined prediction, then the scheduler
    applies stochastic interpolation to prepare the input for the next step.

    Example
    -------
    ::

        from src.models.unsb import create_generator
        from src.schedulers.unsb import UNSBScheduler
        from src.pipelines.unsb import UNSBPipeline

        generator = create_generator(input_nc=3, output_nc=3, ngf=64)
        scheduler = UNSBScheduler(num_timesteps=5, tau=0.01)
        pipeline = UNSBPipeline(generator=generator, scheduler=scheduler)
        output = pipeline(source_image, output_type="pt")
    """

    def __init__(self, generator: UNSBGenerator, scheduler: UNSBScheduler) -> None:
        self.generator = generator
        self.scheduler = scheduler

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        subfolder: str = "generator",
        device: str | torch.device = "cpu",
        torch_dtype: torch.dtype | None = None,
        scheduler_num_timesteps: int = 5,
        scheduler_tau: float = 0.01,
    ) -> "UNSBPipeline":
        """Load UNSB pipeline from local config + safetensors."""
        model_dir = Path(pretrained_model_name_or_path)
        if subfolder:
            model_dir = model_dir / subfolder

        config_path = model_dir / "config.json"
        weights_path = model_dir / "diffusion_pytorch_model.safetensors"
        if not (config_path.exists() and weights_path.exists()):
            raise FileNotFoundError("Expected UNSB generator config.json and diffusion_pytorch_model.safetensors")

        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)

        generator = create_generator(
            input_nc=cfg.get("input_nc", 3),
            output_nc=cfg.get("output_nc", 3),
            ngf=cfg.get("ngf", 64),
            n_blocks=cfg.get("n_blocks", 9),
            n_mlp=cfg.get("n_mlp", 3),
            norm_type=cfg.get("norm_type", "instance"),
            use_dropout=cfg.get("use_dropout", False),
            no_antialias=cfg.get("no_antialias", False),
            no_antialias_up=cfg.get("no_antialias_up", False),
            init_type=cfg.get("init_type", "normal"),
            init_gain=cfg.get("init_gain", 0.02),
        )
        from safetensors.torch import load_file

        state_dict = load_file(str(weights_path), device="cpu")
        generator.load_state_dict(state_dict, strict=True)
        generator = generator.eval().to(device=device)
        if torch_dtype is not None:
            generator = generator.to(dtype=torch_dtype)

        scheduler = UNSBScheduler(num_timesteps=scheduler_num_timesteps, tau=scheduler_tau)
        return cls(generator=generator, scheduler=scheduler)

    @property
    def device(self) -> torch.device:
        """Get the device of the pipeline."""
        return next(self.generator.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Get the dtype of the pipeline."""
        return next(self.generator.parameters()).dtype

    def to(self, device: torch.device | str) -> "UNSBPipeline":
        """Move the pipeline to a device."""
        self.generator = self.generator.to(device)
        return self

    def prepare_inputs(
        self,
        image: Union[torch.Tensor, Image.Image, List[Image.Image]],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Prepare input images for the pipeline.

        Converts PIL images or numpy arrays to normalised tensors in
        ``[-1, 1]`` range.
        """
        if isinstance(image, Image.Image):
            image = [image]

        if isinstance(image, list) and isinstance(image[0], Image.Image):
            images = []
            for img in image:
                img_array = np.array(img, dtype=np.float32)
                if img_array.max() > 1.0:
                    img_array = img_array / 255.0
                if img_array.ndim == 2:
                    img_array = img_array[:, :, np.newaxis]
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
                images.append(img_tensor)
            image = torch.stack(images)

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)

        # Ensure image is in [-1, 1] range
        if image.min() >= 0 and image.max() <= 1.0:
            image = image * 2 - 1  # [0, 1] → [-1, 1]
        elif image.max() > 1.0:
            image = image / 255.0 * 2 - 1  # [0, 255] → [-1, 1]

        return image.to(device=device, dtype=dtype)

    @torch.no_grad()
    def __call__(
        self,
        source_image: Union[torch.Tensor, Image.Image, List[Image.Image]],
        num_timesteps: int | None = None,
        output_type: str = "pil",
        return_dict: bool = True,
    ) -> Union[UNSBPipelineOutput, tuple]:
        """Generate translated images via multi-step stochastic refinement.

        Parameters
        ----------
        source_image : torch.Tensor or PIL.Image.Image or list of PIL.Image.Image
            Source images for translation.  Tensors should be ``(B, C, H, W)``
            in ``[0, 1]`` or ``[-1, 1]`` range.
        num_timesteps : int, optional
            Number of refinement steps.  Defaults to the scheduler setting.
        output_type : str
            ``"pil"``, ``"np"``, or ``"pt"`` (default ``"pil"``).
        return_dict : bool
            If ``True``, return a :class:`UNSBPipelineOutput`.

        Returns
        -------
        UNSBPipelineOutput or tuple
            Translated images and the number of function evaluations.
        """
        device = self.device
        dtype = self.dtype

        x = self.prepare_inputs(source_image, device, dtype)
        T = num_timesteps if num_timesteps is not None else self.scheduler.num_timesteps
        times = self.scheduler.times.to(device)
        ngf = self.generator.ngf
        bs = x.shape[0]

        self.generator.eval()
        Xt = x
        nfe = 0

        for t in range(T):
            if t > 0:
                delta = times[t] - times[t - 1]
                denom = times[-1] - times[t - 1]
                inter = (delta / denom).reshape(-1, 1, 1, 1)
                scale = (delta * (1 - delta / denom)).reshape(-1, 1, 1, 1)
                Xt = (
                    (1 - inter) * Xt
                    + inter * Xt_1.detach()
                    + (scale * self.scheduler.tau).sqrt() * torch.randn_like(Xt)
                )

            time_idx = (t * torch.ones(bs, device=device)).long()
            z = torch.randn(bs, 4 * ngf, device=device, dtype=dtype)
            Xt_1 = self.generator(Xt, time_idx, z)
            nfe += 1

        images = Xt_1.clamp(-1, 1)

        if output_type == "pil":
            images = pt_to_pil(images)
        elif output_type == "np":
            images = self._convert_to_numpy(images)
        # else: output_type == "pt", return tensor as-is

        if not return_dict:
            return (images, nfe)

        return UNSBPipelineOutput(images=images, nfe=nfe)

    @staticmethod
    def _convert_to_numpy(images: torch.Tensor) -> np.ndarray:
        """Convert tensor in [-1, 1] to numpy array in [0, 1]."""
        images = (images + 1) / 2
        images = images.clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        return images
