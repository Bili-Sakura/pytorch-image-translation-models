# Copyright (c) 2026 EarthBridge Team.
# Credits: FCDM (Kwon et al., CVPR 2026) - https://github.com/star-kwon/FCDM

"""FCDM inference pipeline for generative sampling and image-conditioned translation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from src.models.fcdm import FCDM, FCDMImageCond, FCDM_MODELS
from src.schedulers.fcdm import FCDMScheduler


@dataclass
class FCDMPipelineOutput:
    """Output of the FCDM pipeline."""

    images: Any
    nfe: int = 0


class FCDMPipeline:
    """Pipeline for FCDM (class-conditional generative) sampling.

    Operates in latent space; use ``vae`` for decoding to pixel space.
    Supports DDPM and DDIM sampling with optional classifier-free guidance.
    """

    def __init__(
        self,
        model: FCDM,
        scheduler: FCDMScheduler,
        vae: Optional[Any] = None,
        vae_scaling_factor: float = 0.18215,
    ) -> None:
        self.model = model
        self.scheduler = scheduler
        self.vae = vae
        self.vae_scaling_factor = vae_scaling_factor

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str | Path,
        *,
        model_name: str = "FCDM-XL",
        num_classes: int = 1000,
        in_channels: int = 4,
        scheduler_timesteps: int = 1000,
        num_inference_steps: int = 250,
        use_ddpm: bool = False,
        vae_path: Optional[str] = None,
        vae_scaling_factor: float = 0.18215,
        device: str | torch.device = "cpu",
        torch_dtype: Optional[torch.dtype] = None,
        **kwargs: object,
    ) -> "FCDMPipeline":
        """Load FCDM pipeline from checkpoint.

        Expects a checkpoint dict with 'ema' key (as in the original FCDM repo).
        """
        path = Path(pretrained_model_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        model_cls = FCDM_MODELS.get(model_name)
        if model_cls is None:
            raise ValueError(f"Unknown model {model_name}. Use one of {list(FCDM_MODELS.keys())}")

        model = model_cls(num_classes=num_classes, in_channels=in_channels, **kwargs)
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(ckpt, dict) and "ema" in ckpt:
            model.load_state_dict(ckpt["ema"], strict=True)
        else:
            model.load_state_dict(ckpt, strict=True)

        model = model.eval().to(device=device)
        if torch_dtype is not None:
            model = model.to(dtype=torch_dtype)

        scheduler = FCDMScheduler(
            num_train_timesteps=scheduler_timesteps,
            use_ddpm=use_ddpm,
        )
        scheduler.set_timesteps(num_inference_steps)

        vae = None
        if vae_path is not None:
            try:
                from diffusers import AutoencoderKL
                vae = AutoencoderKL.from_pretrained(vae_path, local_files_only=True)
                vae = vae.eval().to(device=device)
            except Exception as e:
                raise RuntimeError(f"Failed to load VAE from {vae_path}: {e}") from e

        return cls(model=model, scheduler=scheduler, vae=vae, vae_scaling_factor=vae_scaling_factor)

    def _get_model_output(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward pass; with learn_sigma, use only epsilon (first in_channels)."""
        out = self.model(x, t, y)
        if self.model.learn_sigma:
            out = out[:, : self.model.in_channels]
        return out

    def _get_model_output_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        cfg_scale: float,
    ) -> torch.Tensor:
        """Classifier-free guidance forward."""
        out = self.model.forward_with_cfg(x, t, y, cfg_scale)
        if self.model.learn_sigma:
            out = out[:, : self.model.in_channels]
        return out

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        num_inference_steps: Optional[int] = None,
        labels: Optional[torch.Tensor] = None,
        num_classes: int = 1000,
        latent_size: Optional[int] = None,
        image_size: int = 256,
        use_cfg: bool = False,
        cfg_scale: float = 1.5,
        use_ddpm: Optional[bool] = None,
        generator: Optional[torch.Generator] = None,
        output_type: str = "pt",
    ) -> FCDMPipelineOutput:
        """Sample from FCDM (class-conditional).

        Parameters
        ----------
        batch_size : int
            Number of samples to generate.
        num_inference_steps : int, optional
            Override scheduler step count.
        labels : Tensor (B,), optional
            Class labels in [0, num_classes-1]. If None, random labels.
        num_classes : int
            Number of classes (for random labels).
        latent_size : int, optional
            Latent spatial size (default: image_size // 8).
        image_size : int
            Target pixel resolution (for latent_size default).
        use_cfg : bool
            Use classifier-free guidance.
        cfg_scale : float
            Guidance scale when use_cfg.
        use_ddpm : bool, optional
            Override scheduler mode (DDPM vs DDIM).
        generator : Generator, optional
            RNG for sampling.
        output_type : str
            ``"pt"`` | ``"np"`` | ``"pil"``.
        """
        if latent_size is None:
            latent_size = image_size // 8

        if num_inference_steps is not None:
            self.scheduler.set_timesteps(num_inference_steps)
        if self.scheduler.timesteps is None:
            self.scheduler.set_timesteps(250)

        if use_ddpm is not None:
            self.scheduler.use_ddpm = use_ddpm

        if labels is None:
            labels = torch.randint(0, num_classes, (batch_size,), device=self.device, dtype=torch.long)

        shape = (batch_size, self.model.in_channels, latent_size, latent_size)
        z = torch.randn(shape, device=self.device, dtype=self.dtype, generator=generator)

        if use_cfg:
            z = torch.cat([z, z], dim=0)
            y_null = torch.full((batch_size,), num_classes, device=self.device, dtype=torch.long)
            labels = torch.cat([labels, y_null], dim=0)

        steps = self.scheduler.timesteps
        for i in tqdm(range(len(steps)), desc="FCDM sampling", total=len(steps)):
            t_val = int(steps[i].item())
            t = torch.full((z.shape[0],), t_val, device=self.device, dtype=torch.long)

            if use_cfg:
                model_out = self._get_model_output_with_cfg(z, t, labels, cfg_scale)
            else:
                model_out = self._get_model_output(z, t, labels)

            if use_cfg:
                model_out, _ = model_out.chunk(2, dim=0)

            if self.scheduler.use_ddpm:
                out = self.scheduler.step(model_out, t, z, clip_min=-10.0, clip_max=10.0)
            else:
                t_next = int(steps[i + 1].item()) if i + 1 < len(steps) else -1
                t_next_t = torch.full(
                    (z.shape[0],),
                    t_next,
                    device=self.device,
                    dtype=torch.long,
                )
                out = self.scheduler.ddim_step(
                    model_out,
                    t,
                    t_next_t,
                    z,
                    clip_min=-10.0,
                    clip_max=10.0,
                )

            z = out.prev_sample

        if use_cfg:
            z = z[: batch_size]

        if self.vae is not None:
            z = z / self.vae_scaling_factor
            decoded = self.vae.decode(z).sample
            decoded = (127.5 * decoded + 128.0).clamp(0, 255).byte()
            z = decoded.permute(0, 2, 3, 1).cpu()
        else:
            z = z.cpu()

        return self._format_output(z, output_type, nfe=len(steps))

    def _format_output(self, data: torch.Tensor, output_type: str, nfe: int = 0) -> FCDMPipelineOutput:
        if output_type == "pt":
            return FCDMPipelineOutput(images=data, nfe=nfe)
        arr = data.numpy() if isinstance(data, torch.Tensor) else data
        if output_type == "np":
            return FCDMPipelineOutput(images=arr, nfe=nfe)
        imgs = [Image.fromarray(ar.astype(np.uint8)) for ar in arr]
        return FCDMPipelineOutput(images=imgs, nfe=nfe)


class FCDMImageCondPipeline:
    """Pipeline for FCDMImageCond (image-to-image translation)."""

    def __init__(
        self,
        model: FCDMImageCond,
        scheduler: FCDMScheduler,
        vae: Optional[Any] = None,
        vae_scaling_factor: float = 0.18215,
    ) -> None:
        self.model = model
        self.scheduler = scheduler
        self.vae = vae
        self.vae_scaling_factor = vae_scaling_factor

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    @torch.no_grad()
    def __call__(
        self,
        condition: torch.Tensor,
        num_inference_steps: Optional[int] = None,
        latent_size: Optional[int] = None,
        use_ddpm: Optional[bool] = None,
        generator: Optional[torch.Generator] = None,
        output_type: str = "pt",
    ) -> FCDMPipelineOutput:
        """Translate from condition latent to target.

        Parameters
        ----------
        condition : Tensor (B, C_cond, H, W)
            Condition latents (e.g. encoded source image).
        num_inference_steps : int, optional
        latent_size : int, optional
        use_ddpm : bool, optional
        generator : Generator, optional
        output_type : str
        """
        b = condition.shape[0]
        if latent_size is None:
            latent_size = condition.shape[2]

        if num_inference_steps is not None:
            self.scheduler.set_timesteps(num_inference_steps)
        if self.scheduler.timesteps is None:
            self.scheduler.set_timesteps(250)
        if use_ddpm is not None:
            self.scheduler.use_ddpm = use_ddpm

        shape = (b, self.model.in_channels, latent_size, latent_size)
        z = torch.randn(shape, device=self.device, dtype=self.dtype, generator=generator)

        steps = self.scheduler.timesteps
        for i in tqdm(range(len(steps)), desc="FCDM translation", total=len(steps)):
            t_val = int(steps[i].item())
            t = torch.full((b,), t_val, device=self.device, dtype=torch.long)

            model_out = self.model(z, t, condition)
            if self.model.learn_sigma:
                model_out = model_out[:, : self.model.in_channels]

            if self.scheduler.use_ddpm:
                out = self.scheduler.step(model_out, t, z, clip_min=-10.0, clip_max=10.0)
            else:
                t_next = int(steps[i + 1].item()) if i + 1 < len(steps) else -1
                t_next_t = torch.full((b,), t_next, device=self.device, dtype=torch.long)
                out = self.scheduler.ddim_step(
                    model_out,
                    t,
                    t_next_t,
                    z,
                    clip_min=-10.0,
                    clip_max=10.0,
                )

            z = out.prev_sample

        if self.vae is not None:
            z = z / self.vae_scaling_factor
            decoded = self.vae.decode(z).sample
            decoded = (127.5 * decoded + 128.0).clamp(0, 255).byte()
            z = decoded.permute(0, 2, 3, 1).cpu()
        else:
            z = z.cpu()

        if output_type == "pt":
            return FCDMPipelineOutput(images=z, nfe=len(steps))
        arr = z.numpy()
        if output_type == "np":
            return FCDMPipelineOutput(images=arr, nfe=len(steps))
        imgs = [Image.fromarray(ar.astype(np.uint8)) for ar in arr]
        return FCDMPipelineOutput(images=imgs, nfe=len(steps))
