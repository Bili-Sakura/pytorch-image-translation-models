# Copyright (c) 2026 EarthBridge Team.
# Credits: ECSI (https://github.com/szhan311/ECSI) - diffusion bridge for image translation.

"""ECSI inference pipeline for image-to-image translation.

ECSI extends DDBM with improved bridge parametrization. Uses Heun or stochastic
sampling with Karras sigma schedule.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from diffusers.utils import pt_to_pil

from src.models.ecsi.diffusion import KarrasDenoiser
from src.models.ecsi.model import create_model
from src.models.ecsi.route import get_route
from src.models.ecsi.sampling import karras_sample


@dataclass
class ECSIPipelineOutput:
    """Output of the ECSI pipeline.

    Attributes
    ----------
    images : list[PIL.Image.Image] | np.ndarray | torch.Tensor
        Translated images.
    nfe : int
        Number of function evaluations.
    """

    images: Any
    nfe: int = 0


class ECSIPipeline:
    """Image-to-image translation pipeline using ECSI (DDBM-extended bridge model)."""

    def __init__(self, model: torch.nn.Module, diffusion: KarrasDenoiser) -> None:
        self.model = model
        self.diffusion = diffusion

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
        config: dict | None = None,
        device: str | torch.device = "cpu",
        torch_dtype: torch.dtype | None = None,
        pred_mode: str = "vp",
        **kwargs,
    ) -> "ECSIPipeline":
        """Load ECSI pipeline from a PyTorch checkpoint (.pt).

        Expects an OpenAI-style state dict. Pass config dict for non-default model/diffusion.
        """
        path = Path(pretrained_model_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        config = config or {}
        model_config = config.get("model", {})
        diffusion_config = config.get("diffusion", {})

        model = create_model(
            image_size=model_config.get("image_size", 256),
            in_channels=model_config.get("in_channels", 3),
            num_channels=model_config.get("num_channels", 128),
            num_res_blocks=model_config.get("num_res_blocks", 2),
            unet_type=model_config.get("unet_type", "adm"),
            channel_mult=model_config.get("channel_mult", ""),
            attention_resolutions=model_config.get("attention_resolutions", "32,16,8"),
            num_heads=model_config.get("num_heads", 4),
            num_head_channels=model_config.get("num_head_channels", 64),
            use_scale_shift_norm=model_config.get("use_scale_shift_norm", True),
            dropout=model_config.get("dropout", 0.1),
            resblock_updown=model_config.get("resblock_updown", True),
            use_fp16=model_config.get("use_fp16", False),
            attention_type=model_config.get("attention_type", "default"),
            condition_mode=model_config.get("condition_mode", "concat"),
        )

        state = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        model = model.eval().to(device=device)

        diffusion = KarrasDenoiser(
            sigma_data=diffusion_config.get("sigma_data", 0.5),
            sigma_min=diffusion_config.get("sigma_min", 1e-4),
            sigma_max=diffusion_config.get("sigma_max", 1.0),
            cov_xy=diffusion_config.get("cov_xy", 0.0),
            pred_mode=diffusion_config.get("pred_mode", pred_mode),
        )

        if torch_dtype is not None:
            model = model.to(dtype=torch_dtype)
        return cls(model=model, diffusion=diffusion)

    @staticmethod
    def prepare_inputs(image: Any, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Convert inputs to [-1, 1] tensors."""
        if isinstance(image, Image.Image):
            image = [image]
        if isinstance(image, list) and image and isinstance(image[0], Image.Image):
            images = []
            for img in image:
                img = img.convert("RGB")
                arr = np.array(img).astype(np.float32) / 255.0
                images.append(torch.from_numpy(arr).permute(2, 0, 1))
            image = torch.stack(images)
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        if hasattr(image, "max") and image.max() > 1.0:
            image = image / 255.0
        if hasattr(image, "min") and image.min() >= 0:
            image = image * 2 - 1
        return image.to(device=device, dtype=dtype)

    @torch.no_grad()
    def __call__(
        self,
        source_image: Union[torch.Tensor, Image.Image, List[Image.Image]],
        num_inference_steps: int = 40,
        sampler: str = "heun",
        churn_step_ratio: float = 0.33,
        pred_mode: str | None = None,
        sigma_min: float = 0.002,
        sigma_max: float = 1.0,
        rho: float = 7.0,
        smooth: float = 0.0,
        clip_denoised: bool = True,
        output_type: str = "pil",
        return_dict: bool = True,
        callback: Optional[Callable] = None,
        callback_steps: int = 1,
    ):
        """Run ECSI bridge diffusion sampling.

        Parameters
        ----------
        source_image : Tensor | PIL.Image | list[PIL.Image]
            Source image(s) in [0, 1] or [0, 255].
        num_inference_steps : int
            Number of denoising steps.
        sampler : str
            "heun" (deterministic) or "stoch" (stochastic).
        pred_mode : str or None
            Route mode: "vp", "ve", etc. Defaults to diffusion's pred_mode.
        output_type : str
            "pil", "np", or "pt".
        """
        device = self.device
        dtype = self.dtype
        x_T = self.prepare_inputs(source_image, device, dtype)
        route = get_route(pred_mode or self.diffusion.pred_mode)

        def cb(info):
            if callback is not None and info.get("i", 0) % callback_steps == 0:
                callback(info.get("i", 0), num_inference_steps, info.get("x"))

        result, path, x0_est = karras_sample(
            self.diffusion,
            self.model,
            x_T,
            None,
            route,
            steps=num_inference_steps,
            clip_denoised=clip_denoised,
            progress=True,
            callback=cb,
            model_kwargs={"xT": x_T},
            device=device,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            rho=rho,
            sampler=sampler,
            churn_step_ratio=churn_step_ratio,
            smooth=smooth,
        )

        images = result.clamp(-1, 1)
        if output_type == "pil":
            images = pt_to_pil(images)
        elif output_type == "np":
            images = self._convert_to_numpy(images)

        nfe = num_inference_steps * (2 if sampler == "heun" else 1)
        if not return_dict:
            return (images, nfe)
        return ECSIPipelineOutput(images=images, nfe=nfe)

    @staticmethod
    def _convert_to_numpy(images: torch.Tensor) -> np.ndarray:
        images = (images + 1) / 2
        return images.clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()


def load_ecsi_pipeline(
    checkpoint_path: str | Path,
    *,
    config: dict | None = None,
    device: str | torch.device = "cpu",
    torch_dtype: torch.dtype | None = None,
    pred_mode: str = "vp",
    **kwargs,
) -> ECSIPipeline:
    """Load ECSI pipeline from a checkpoint."""
    return ECSIPipeline.from_pretrained(
        checkpoint_path,
        config=config,
        device=device,
        torch_dtype=torch_dtype,
        pred_mode=pred_mode,
        **kwargs,
    )
