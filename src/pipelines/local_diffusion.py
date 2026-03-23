# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Local Diffusion inference pipeline with branching and fusion.

Provides :class:`LocalDiffusionPipeline` for running hallucination-aware
inference.  The pipeline supports two modes:

1. **Standard mode** – standard DDPM/DDIM reverse-process sampling.
2. **Branch-and-fuse mode** – the input anomaly mask partitions the
   conditioning image into in-distribution (IND) and out-of-distribution
   (OOD) regions.  Two denoising branches run in parallel; at a fusion
   timestep they merge and a single unified denoising continues.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image

from diffusers.utils import pt_to_pil

from src.models.local_diffusion import LocalDiffusionUNet
from src.schedulers.local_diffusion import LocalDiffusionScheduler


@dataclass
class LocalDiffusionPipelineOutput:
    """Output class for Local Diffusion pipeline.

    Attributes
    ----------
    images : list of PIL.Image.Image, np.ndarray, or torch.Tensor
        Generated images after translation.
    nfe : int
        Number of function evaluations (model forward passes).
    """

    images: Union[List[Image.Image], np.ndarray, torch.Tensor]
    nfe: int


class LocalDiffusionPipeline:
    """Inference pipeline for Local Diffusion with optional branching and fusion.

    The pipeline wraps a :class:`LocalDiffusionUNet` and a
    :class:`LocalDiffusionScheduler` with a consistent API for loading,
    preprocessing, and postprocessing.

    Example
    -------
    ::

        from src.models.local_diffusion import create_unet
        from src.schedulers.local_diffusion import LocalDiffusionScheduler
        from src.pipelines.local_diffusion import LocalDiffusionPipeline

        unet = create_unet(dim=32, channels=1)
        scheduler = LocalDiffusionScheduler(num_train_timesteps=250)
        pipeline = LocalDiffusionPipeline(unet=unet, scheduler=scheduler)
        output = pipeline(cond_image=my_tensor, output_type="pt")
    """

    def __init__(
        self,
        unet: LocalDiffusionUNet,
        scheduler: LocalDiffusionScheduler,
    ) -> None:
        self.unet = unet
        self.scheduler = scheduler

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        unet_subfolder: str = "unet",
        scheduler_subfolder: str = "scheduler",
        device: str | torch.device = "cpu",
        torch_dtype: torch.dtype | None = None,
    ) -> "LocalDiffusionPipeline":
        """Load Local Diffusion UNet + scheduler from local config files."""
        root = Path(pretrained_model_name_or_path)
        unet_dir = root / unet_subfolder
        cfg_path = unet_dir / "config.json"
        weights_path = unet_dir / "diffusion_pytorch_model.safetensors"
        if not (cfg_path.exists() and weights_path.exists()):
            raise FileNotFoundError(
                f"Expected {cfg_path} and {weights_path} for LocalDiffusionPipeline.from_pretrained"
            )

        with open(cfg_path, encoding="utf-8") as f:
            unet_cfg = json.load(f)
        valid_keys = set(inspect.signature(LocalDiffusionUNet.__init__).parameters.keys()) - {"self"}
        unet_kwargs = {k: v for k, v in unet_cfg.items() if k in valid_keys}
        unet = LocalDiffusionUNet(**unet_kwargs)

        from safetensors.torch import load_file

        state_dict = load_file(str(weights_path), device="cpu")
        unet.load_state_dict(state_dict, strict=True)

        sched_cfg_path = root / scheduler_subfolder / "scheduler_config.json"
        if sched_cfg_path.exists():
            with open(sched_cfg_path, encoding="utf-8") as f:
                sched_cfg = json.load(f)
            valid_s_keys = set(inspect.signature(LocalDiffusionScheduler.__init__).parameters.keys()) - {"self"}
            sched_kwargs = {k: v for k, v in sched_cfg.items() if k in valid_s_keys}
            scheduler = LocalDiffusionScheduler(**sched_kwargs)
        else:
            scheduler = LocalDiffusionScheduler()

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

    def to(self, device: torch.device | str) -> "LocalDiffusionPipeline":
        self.unet = self.unet.to(device)
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

        if image.min() >= 0 and image.max() <= 1.0:
            image = image * 2 - 1
        elif image.max() > 1.0:
            image = image / 255.0 * 2 - 1

        return image.to(device=device, dtype=dtype)

    @torch.no_grad()
    def __call__(
        self,
        cond_image: Union[torch.Tensor, Image.Image, List[Image.Image]],
        anomaly_mask: Optional[torch.Tensor] = None,
        num_inference_steps: Optional[int] = None,
        use_ddim: bool = False,
        ddim_steps: Optional[int] = None,
        branch_out: bool = False,
        fusion_timestep: int = 2,
        clip_min: float = -1.0,
        clip_max: float = 1.0,
        output_type: str = "pil",
        return_dict: bool = True,
    ) -> Union[LocalDiffusionPipelineOutput, tuple]:
        """Generate translated images via diffusion sampling.

        Parameters
        ----------
        cond_image : torch.Tensor or PIL.Image.Image or list of PIL.Image.Image
            Conditioning image (e.g. degraded, noisy, or corrupted input).
        anomaly_mask : torch.Tensor, optional
            Binary mask indicating OOD regions (1 = anomalous, 0 = normal).
            Required when ``branch_out=True``.
        num_inference_steps : int, optional
            Number of DDPM steps.  Defaults to the scheduler's training steps.
        use_ddim : bool
            Use DDIM sampling instead of DDPM.
        ddim_steps : int, optional
            Number of DDIM steps (ignored when ``use_ddim=False``).
        branch_out : bool
            Enable branch-and-fuse mode for hallucination suppression.
        fusion_timestep : int
            Timestep at which to fuse the OOD and IND branches.
        clip_min, clip_max : float
            Clipping bounds for predicted x_0.
        output_type : str
            ``"pil"``, ``"np"``, or ``"pt"`` (default ``"pil"``).
        return_dict : bool
            If ``True``, return a :class:`LocalDiffusionPipelineOutput`.

        Returns
        -------
        LocalDiffusionPipelineOutput or tuple
            Translated images and the number of function evaluations.
        """
        device = self.device
        dtype = self.dtype

        cond = self.prepare_inputs(cond_image, device, dtype)
        bs, channels, h, w = cond.shape
        T = num_inference_steps if num_inference_steps is not None else self.scheduler.num_train_timesteps

        self.unet.eval()
        nfe = 0

        if use_ddim:
            images, nfe = self._ddim_sample(
                cond, anomaly_mask, ddim_steps or T, branch_out,
                fusion_timestep, clip_min, clip_max, device, dtype,
            )
        else:
            images, nfe = self._ddpm_sample(
                cond, anomaly_mask, T, branch_out,
                fusion_timestep, clip_min, clip_max, device, dtype,
            )

        images = images.clamp(clip_min, clip_max)

        if output_type == "pil":
            images = pt_to_pil(images)
        elif output_type == "np":
            images = self._convert_to_numpy(images)

        if not return_dict:
            return (images, nfe)
        return LocalDiffusionPipelineOutput(images=images, nfe=nfe)

    # ------------------------------------------------------------------
    # DDPM sampling loop
    # ------------------------------------------------------------------

    def _ddpm_sample(
        self,
        cond: torch.Tensor,
        anomaly_mask: Optional[torch.Tensor],
        T: int,
        branch_out: bool,
        fusion_timestep: int,
        clip_min: float,
        clip_max: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple:
        bs, channels, h, w = cond.shape
        img = torch.randn(bs, channels, h, w, device=device, dtype=dtype)
        nfe = 0
        branching = branch_out and anomaly_mask is not None

        if branching:
            mask = anomaly_mask.to(device).float()
            mask_binary = (mask >= 1.0).float()
            img_out = torch.randn_like(img)
            img_in = torch.randn_like(img)

        for t_idx in reversed(range(0, T)):
            t = torch.full((bs,), t_idx, device=device, dtype=torch.long)

            if branching and t_idx > fusion_timestep:
                # Branch mode: separate OOD and IND denoising
                cond_out = cond * mask_binary
                cond_in = cond * (1.0 - mask_binary)

                out_pred = self.unet(img_out, cond_out, t)
                in_pred = self.unet(img_in, cond_in, t)
                nfe += 2

                step_out = self.scheduler.step(out_pred, t, img_out, clip_min, clip_max)
                step_in = self.scheduler.step(in_pred, t, img_in, clip_min, clip_max)

                img_out = step_out.prev_sample
                img_in = step_in.prev_sample

                if t_idx == fusion_timestep + 1:
                    # Fuse at next step
                    x_start_fused = (
                        step_out.pred_x_start * mask_binary
                        + step_in.pred_x_start * (1.0 - mask_binary)
                    )
                    img = torch.where(mask_binary.bool(), img_out, img_in)
                    branching = False
            else:
                # Standard denoising
                model_output = self.unet(img, cond, t)
                nfe += 1
                step_result = self.scheduler.step(model_output, t, img, clip_min, clip_max)
                img = step_result.prev_sample

        return img, nfe

    # ------------------------------------------------------------------
    # DDIM sampling loop
    # ------------------------------------------------------------------

    def _ddim_sample(
        self,
        cond: torch.Tensor,
        anomaly_mask: Optional[torch.Tensor],
        ddim_steps: int,
        branch_out: bool,
        fusion_timestep: int,
        clip_min: float,
        clip_max: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple:
        bs, channels, h, w = cond.shape
        T = self.scheduler.num_train_timesteps

        # Build DDIM time schedule
        times = torch.linspace(-1, T - 1, steps=ddim_steps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        img = torch.randn(bs, channels, h, w, device=device, dtype=dtype)
        nfe = 0
        branching = branch_out and anomaly_mask is not None

        if branching:
            mask = anomaly_mask.to(device).float()
            mask_binary = (mask >= 1.0).float()
            img_out = torch.randn_like(img)
            img_in = torch.randn_like(img)

        for time_now, time_next in time_pairs:
            t = torch.full((bs,), time_now, device=device, dtype=torch.long)
            t_next = torch.full((bs,), time_next, device=device, dtype=torch.long)

            if branching and time_now > fusion_timestep:
                cond_out = cond * mask_binary
                cond_in = cond * (1.0 - mask_binary)

                out_pred = self.unet(img_out, cond_out, t)
                in_pred = self.unet(img_in, cond_in, t)
                nfe += 2

                step_out = self.scheduler.ddim_step(out_pred, t, t_next, img_out, clip_min, clip_max)
                step_in = self.scheduler.ddim_step(in_pred, t, t_next, img_in, clip_min, clip_max)

                img_out = step_out.prev_sample
                img_in = step_in.prev_sample

                if time_next <= fusion_timestep:
                    img = torch.where(mask_binary.bool(), img_out, img_in)
                    branching = False
            else:
                model_output = self.unet(img, cond, t)
                nfe += 1
                step_result = self.scheduler.ddim_step(model_output, t, t_next, img, clip_min, clip_max)
                img = step_result.prev_sample

        return img, nfe

    # ------------------------------------------------------------------
    # Output conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_to_numpy(images: torch.Tensor) -> np.ndarray:
        """Convert tensor in [-1, 1] to numpy array in [0, 1]."""
        images = (images + 1) / 2
        images = images.clamp(0, 1)
        return images.cpu().permute(0, 2, 3, 1).numpy()
