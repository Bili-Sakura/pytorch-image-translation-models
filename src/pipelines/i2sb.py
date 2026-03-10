# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""I2SB inference pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from src.models.unet.diffusers_wrappers import I2SBDiffusersUNet
from src.models.unet.i2sb_unet import I2SBUNet
from src.schedulers.i2sb import I2SBScheduler


@dataclass
class I2SBPipelineOutput:
    """Container for pipeline results.

    Attributes
    ----------
    images : Tensor | list[PIL.Image.Image] | np.ndarray
        Translated images in the requested format.
    nfe : int
        Number of function evaluations used.
    """

    images: Any
    nfe: int


class I2SBPipeline:
    """End-to-end inference pipeline for an I2SB model.

    Parameters
    ----------
    unet : I2SBUNet
        Trained I2SB U-Net.
    scheduler : I2SBScheduler
        Noise scheduler.
    """

    def __init__(self, unet: I2SBUNet, scheduler: I2SBScheduler) -> None:
        self.unet = unet
        self.scheduler = scheduler

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        subfolder: str = "unet",
        scheduler_config: str = "scheduler_config.json",
        device: str | torch.device = "cpu",
        torch_dtype: torch.dtype | None = None,
    ) -> "I2SBPipeline":
        """Load I2SB pipeline from local checkpoint and scheduler config."""
        unet = I2SBDiffusersUNet.from_pretrained(pretrained_model_name_or_path, subfolder=subfolder)

        root = Path(pretrained_model_name_or_path)
        sched_cfg_path = root / scheduler_config
        if not sched_cfg_path.exists():
            sched_cfg_path = root / "scheduler" / scheduler_config
        if sched_cfg_path.exists():
            with open(sched_cfg_path, encoding="utf-8") as f:
                cfg = json.load(f)
            scheduler = I2SBScheduler(
                interval=cfg.get("interval", 1000),
                beta_max=cfg.get("beta_max", 0.3),
            )
        else:
            scheduler = I2SBScheduler()

        unet = unet.eval().to(device=device)
        if torch_dtype is not None:
            unet = unet.to(dtype=torch_dtype)
        return cls(unet=unet, scheduler=scheduler)

    def to(self, device: torch.device | str) -> "I2SBPipeline":
        self.unet = self.unet.to(device)
        return self

    @property
    def device(self) -> torch.device:
        return next(self.unet.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.unet.parameters()).dtype

    @torch.no_grad()
    def __call__(
        self,
        source: torch.Tensor,
        nfe: int = 20,
        ot_ode: bool = False,
        output_type: str = "pt",
    ) -> I2SBPipelineOutput:
        """Run the I2SB reverse bridge starting from *source*.

        Parameters
        ----------
        source : Tensor ``[B, C, H, W]``
            Source images.
        nfe : int
            Number of function evaluations (denoising steps).
        ot_ode : bool
            If *True*, use the deterministic OT-ODE sampler (no noise
            injected during reverse steps).
        output_type : ``"pt"`` | ``"pil"`` | ``"np"``
            Desired output format.
        """
        source = source.to(device=self.device, dtype=self.dtype)
        self.scheduler.set_timesteps(nfe=nfe)
        device = self.device

        # Corrupt source with maximum forward noise to initialise x_T
        noise = torch.randn_like(source)
        std_T = self.scheduler.std_fwd[-1].to(device)
        x = source + std_T * noise

        timesteps = self.scheduler.timesteps
        for i in range(len(timesteps) - 1):
            t = timesteps[i].item()
            t_prev = timesteps[i + 1].item()

            t_batch = torch.full((source.shape[0],), t, device=device, dtype=torch.float32)

            if self.unet.condition_mode == "concat":
                model_output = self.unet(x, t_batch, cond=source)
            else:
                model_output = self.unet(x, t_batch)

            result = self.scheduler.step(model_output, t, t_prev, x, ot_ode=ot_ode)
            x = result.prev_sample

        return self._format_output(x, nfe, output_type)

    # ------------------------------------------------------------------
    # Output formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_output(
        tensor: torch.Tensor,
        nfe: int,
        output_type: str,
    ) -> I2SBPipelineOutput:
        if output_type == "pt":
            return I2SBPipelineOutput(images=tensor, nfe=nfe)

        if output_type == "np":
            return I2SBPipelineOutput(images=tensor.cpu().numpy(), nfe=nfe)

        if output_type == "pil":
            images: list[Image.Image] = []
            arr = tensor.clamp(0, 1).cpu()
            for i in range(arr.shape[0]):
                img_np = arr[i].permute(1, 2, 0).numpy()
                img_np = (img_np * 255).astype(np.uint8)
                if img_np.shape[2] == 1:
                    img_np = img_np[:, :, 0]
                images.append(Image.fromarray(img_np))
            return I2SBPipelineOutput(images=images, nfe=nfe)

        raise ValueError(f"Unknown output_type: {output_type!r}")
