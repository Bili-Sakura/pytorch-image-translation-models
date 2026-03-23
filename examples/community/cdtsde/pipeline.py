# Copyright (c) 2026 EarthBridge Team.
# Credits: CDTSDE/PSCDE (solar defect identification, projects/CDTSDE).

"""CDTSDE community pipeline for image-to-image translation.

Diffusion bridge pipeline wrapped by pytorch-image-translation-models conventions.
Loads only from diffusers-style checkpoint layout (unet/, controlnet/, vae/, etc.).
Uses ControlLDM + dynamic shift sampling from projects/CDTSDE.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
import torch
from PIL import Image

from diffusers import DiffusionPipeline
from diffusers.utils import BaseOutput, numpy_to_pil


def _ensure_cdtsde_path(cdtsde_src_path: Optional[str | Path]) -> Path:
    """Resolve CDTSDE source path. Default: workspace/projects/CDTSDE."""
    if cdtsde_src_path is not None:
        path = Path(cdtsde_src_path)
        if path.exists():
            return path.resolve()
        raise FileNotFoundError(f"CDTSDE source not found: {path}")

    # Default relative to common workspace layout
    candidates = [
        Path(__file__).resolve().parents[4] / "CDTSDE",  # .../projects/CDTSDE
        Path.cwd() / "projects" / "CDTSDE",
        Path.cwd() / "CDTSDE",
    ]
    for p in candidates:
        if p.exists() and (p / "model" / "cldm.py").exists():
            return p.resolve()

    raise FileNotFoundError(
        "CDTSDE source not found. Set cdtsde_src_path or place CDTSDE at "
        "projects/CDTSDE, ./CDTSDE, or ./projects/CDTSDE"
    )


def _import_cdtsde(cdtsde_path: Path) -> tuple:
    """Add CDTSDE to path and import ControlLDM, utilities, config loader."""
    import sys
    cdtsde_str = str(cdtsde_path)
    if cdtsde_str not in sys.path:
        sys.path.insert(0, cdtsde_str)

    from model.cldm import ControlLDM
    from omegaconf import OmegaConf
    from utils.common import instantiate_from_config, load_state_dict

    return ControlLDM, OmegaConf, instantiate_from_config, load_state_dict


@dataclass
class CDTSDECommunityPipelineOutput(BaseOutput):
    """Output of the CDTSDE community pipeline.

    Attributes
    ----------
    images : list[PIL.Image.Image] | np.ndarray | torch.Tensor
        Translated images.
    nfe : int
        Number of function evaluations (sampling steps).
    """

    images: Any
    nfe: int = 0


class CDTSDECommunityPipeline(DiffusionPipeline):
    """Image-to-image translation pipeline using CDTSDE diffusion bridge.

    Wrapped by pytorch-image-translation-models conventions. Loads only from
    diffusers-style checkpoint layout (unet/, controlnet/, vae/, etc.).
    """

    model_cpu_offload_seq = "model"

    def __init__(self, model: Any) -> None:
        super().__init__()
        self.register_modules(model=model)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        cdtsde_src_path: Optional[str | Path] = None,
        model_config_path: Optional[str | Path] = None,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        **kwargs,
    ) -> "CDTSDECommunityPipeline":
        """Load CDTSDE pipeline from diffusers-style checkpoint only.

        Expects unet/, controlnet/, vae/, text_encoder/, cond_encoder/,
        nonlinear_lambda/ with diffusion_pytorch_model.safetensors.
        Run convert_ckpt_to_cdtsde first for raw .ckpt conversion.

        Parameters
        ----------
        pretrained_model_name_or_path : str | Path
            Path to checkpoint directory (diffusers layout).
        cdtsde_src_path : str | Path | None
            Path to projects/CDTSDE. Auto-detected if None.
        model_config_path : str | Path | None
            Override model config YAML.
        device : str | None
            Device (default: cuda if available).
        torch_dtype : torch.dtype | None
            Optional model dtype.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        path = Path(pretrained_model_name_or_path)
        if not path.is_dir() or not (path / "unet").is_dir():
            raise FileNotFoundError(
                f"Diffusers-style checkpoint not found at {path}. "
                "Expected unet/, controlnet/, vae/, etc. Run convert_ckpt_to_cdtsde first."
            )

        cdtsde_path = _ensure_cdtsde_path(cdtsde_src_path)
        ControlLDM, OmegaConf, instantiate_from_config, load_state_dict = _import_cdtsde(
            cdtsde_path
        )

        # Resolve model config
        config_path = None
        if model_config_path and Path(model_config_path).exists():
            config_path = Path(model_config_path)
        elif (path / "model_config.json").exists():
            with open(path / "model_config.json", encoding="utf-8") as f:
                mc = json.load(f)
            if isinstance(mc, dict) and "config_path" in mc:
                candidate = cdtsde_path / mc["config_path"]
                if candidate.exists():
                    config_path = candidate
        if config_path is None:
            config_path = cdtsde_path / "configs" / "model" / "cldm_v21_dynamic.yaml"

        if not config_path.exists():
            raise FileNotFoundError(
                f"Model config not found: {config_path}. "
                "Pass model_config_path."
            )

        model_cfg = OmegaConf.load(config_path)
        model: ControlLDM = instantiate_from_config(model_cfg)
        model.freeze()
        model.eval()

        # Load from diffusers layout only
        from safetensors.torch import load_file

        state = {}
        subfolders = ("unet", "controlnet", "vae", "text_encoder", "cond_encoder", "nonlinear_lambda")
        for sub in subfolders:
            sub_dir = path / sub
            if not sub_dir.is_dir():
                continue
            for w_name in ("diffusion_pytorch_model.safetensors", "model.safetensors"):
                w_path = sub_dir / w_name
                if w_path.exists():
                    state.update(load_file(str(w_path), device="cpu"))
                    break
            else:
                ckpt_path_sub = sub_dir / "model.ckpt"
                if ckpt_path_sub.exists():
                    raw = torch.load(str(ckpt_path_sub), map_location="cpu", weights_only=True)
                    sd = raw.get("state_dict", raw)
                    state.update({k: v for k, v in sd.items() if torch.is_tensor(v)})

        if not state:
            raise FileNotFoundError(
                f"No weights found in diffusers layout at {path}. "
                "Run convert_ckpt_to_cdtsde first."
            )

        load_state_dict(model, state, strict=True)
        model = model.to(device)
        if torch_dtype is not None:
            model = model.to(dtype=torch_dtype)

        return cls(model=model)

    @staticmethod
    def _prepare_inputs(
        image: Union[torch.Tensor, Image.Image, List[Image.Image]],
        device: torch.device,
        dtype: torch.dtype,
        size: int = 256,
    ) -> torch.Tensor:
        """Prepare input to [1, 3, H, W] in [0, 1], resized to size."""
        if isinstance(image, Image.Image):
            image = [image]
        if isinstance(image, list) and image and isinstance(image[0], Image.Image):
            images = []
            for img in image:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                arr = np.array(img).astype(np.float32) / 255.0
                if arr.ndim == 2:
                    arr = np.stack([arr] * 3, axis=-1)
                t = torch.from_numpy(arr).permute(2, 0, 1)
                images.append(t)
            image = torch.stack(images)
        elif isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
            if image.dim() == 3:
                image = image.permute(2, 0, 1).unsqueeze(0)
            elif image.dim() == 2:
                image = image.unsqueeze(0).unsqueeze(0).expand(-1, 3, -1, -1)
        elif isinstance(image, torch.Tensor):
            if image.dim() == 3:
                image = image.unsqueeze(0)
        else:
            raise TypeError(f"Unexpected image type: {type(image)}")

        if image.max() > 1.0:
            image = image / 255.0

        # Resize to size x size (center crop if needed)
        from torch.nn import functional as F
        b, c, h, w = image.shape
        if h != size or w != size:
            min_dim = min(h, w)
            top = (h - min_dim) // 2
            left = (w - min_dim) // 2
            image = image[:, :, top : top + min_dim, left : left + min_dim]
            image = F.interpolate(
                image, size=(size, size), mode="bilinear", align_corners=False
            )

        return image.to(device=device, dtype=dtype)

    @torch.no_grad()
    def __call__(
        self,
        source_image: Union[torch.Tensor, Image.Image, List[Image.Image]],
        num_inference_steps: int = 50,
        positive_prompt: str = "clean, high-resolution, 8k",
        size: int = 256,
        output_type: str = "pil",
        return_dict: bool = True,
    ) -> Union[CDTSDECommunityPipelineOutput, tuple]:
        """Run CDTSDE sampling.

        Parameters
        ----------
        source_image : Tensor | PIL.Image | list[PIL.Image]
            Source (electroluminescence) image(s).
        num_inference_steps : int
            Number of sampling steps (default 50).
        positive_prompt : str
            Text prompt for conditioning (default solar-defect style).
        size : int
            Input/output spatial size (default 256).
        output_type : str
            "pil" | "np" | "pt".
        return_dict : bool
            If True, return CDTSDECommunityPipelineOutput.

        Returns
        -------
        CDTSDECommunityPipelineOutput or (images, nfe)
        """
        device = self.device
        dtype = next(self.model.parameters()).dtype

        hint = self._prepare_inputs(source_image, device, dtype, size=size)

        # Build conditioning
        c_txt = self.model.get_learned_conditioning([positive_prompt] * hint.shape[0])
        c_latent = self.model.apply_condition_encoder(hint * 2 - 1)
        cond_dict = {
            "c_concat": [hint],
            "c_crossattn": [c_txt],
            "c_latent": [c_latent],
        }

        samples = self.model.sample_log(cond=cond_dict, steps=num_inference_steps)

        # samples: [0,1] pixel space
        images = samples.clamp(0, 1)

        if output_type == "pil":
            images_out = numpy_to_pil(images.cpu().permute(0, 2, 3, 1).numpy())
        elif output_type == "np":
            images_out = images.cpu().permute(0, 2, 3, 1).numpy()
        else:
            images_out = images

        if not return_dict:
            return (images_out, num_inference_steps)
        return CDTSDECommunityPipelineOutput(images=images_out, nfe=num_inference_steps)


def load_cdtsde_community_pipeline(
    pretrained_model_name_or_path: str | Path,
    *,
    cdtsde_src_path: Optional[str | Path] = None,
    model_config_path: Optional[str | Path] = None,
    device: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
) -> CDTSDECommunityPipeline:
    """Load CDTSDE community pipeline from diffusers-style checkpoint.

    Requires unet/, controlnet/, vae/, etc. Run convert_ckpt_to_cdtsde first.

    Parameters
    ----------
    pretrained_model_name_or_path : str | Path
        Path to checkpoint directory (diffusers layout).
    cdtsde_src_path : str | Path | None
        Path to projects/CDTSDE (auto-detected if None).
    model_config_path : str | Path | None
        Override model config YAML path.
    device : str | None
        Device (default: cuda if available).
    torch_dtype : torch.dtype | None
        Optional model dtype.

    Returns
    -------
    CDTSDECommunityPipeline
        Pipeline ready for inference.
    """
    return CDTSDECommunityPipeline.from_pretrained(
        pretrained_model_name_or_path,
        cdtsde_src_path=cdtsde_src_path,
        model_config_path=model_config_path,
        device=device,
        torch_dtype=torch_dtype,
    )
