# Copyright (c) 2026 EarthBridge Team.
# Credits: DiffusionRouter (kvmduc) - https://github.com/kvmduc/DiffusionRouter

"""DiffusionRouter community inference pipeline."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Union

import numpy as np
import torch
from PIL import Image

from diffusers import DiffusionPipeline
from diffusers.utils import BaseOutput, numpy_to_pil

from .model import DiffusionRouterConfig, compose_route, parse_class


def _ensure_diffusionrouter_path(diffusionrouter_src_path: Optional[str | Path]) -> Path:
    """Resolve DiffusionRouter source path."""
    if diffusionrouter_src_path is not None:
        p = Path(diffusionrouter_src_path)
        if p.exists() and (p / "guided_diffusion" / "script_util.py").exists():
            return p.resolve()
        raise FileNotFoundError(
            f"DiffusionRouter source not found at {p}. Expected guided_diffusion/script_util.py."
        )

    root = Path(__file__).resolve().parents[4]
    candidates = [
        root / "DiffusionRouter",
        root / "projects" / "DiffusionRouter",
        Path.cwd() / "DiffusionRouter",
        Path.cwd() / "projects" / "DiffusionRouter",
    ]
    for p in candidates:
        if p.exists() and (p / "guided_diffusion" / "script_util.py").exists():
            return p.resolve()

    raise FileNotFoundError(
        "DiffusionRouter source not found. Set diffusionrouter_src_path or place DiffusionRouter "
        "at ./DiffusionRouter or ./projects/DiffusionRouter."
    )


def _import_diffusionrouter(diffusionrouter_path: Path) -> tuple[Any, Any]:
    """Import DiffusionRouter model factory functions."""
    src = str(diffusionrouter_path)
    if src not in sys.path:
        sys.path.insert(0, src)

    from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults

    return create_model_and_diffusion, model_and_diffusion_defaults


@dataclass
class DiffusionRouterPipelineOutput(BaseOutput):
    """Output of the DiffusionRouter pipeline."""

    images: Union[List[Image.Image], np.ndarray, torch.Tensor]
    route: List[int]
    nfe: int = 0


class DiffusionRouterPipeline(DiffusionPipeline):
    """Image translation pipeline with optional multi-hop routing."""

    model_cpu_offload_seq = "model"

    def __init__(self, model: Any, diffusion: Any, config: DiffusionRouterConfig) -> None:
        super().__init__()
        self.register_modules(model=model)
        self._diffusion = diffusion
        self._config = config
        self._sample_fn = diffusion.ddim_sample_loop if config.use_ddim else diffusion.p_sample_loop

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    @staticmethod
    def _prepare_inputs(
        source_image: Union[torch.Tensor, Image.Image, List[Image.Image], np.ndarray],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if isinstance(source_image, Image.Image):
            source_image = [source_image]

        if isinstance(source_image, list) and source_image and isinstance(source_image[0], Image.Image):
            source_image = torch.stack(
                [torch.from_numpy(np.array(img.convert("RGB"), dtype=np.float32)).permute(2, 0, 1) for img in source_image]
            )

        if isinstance(source_image, np.ndarray):
            source_image = torch.from_numpy(source_image)

        if source_image.ndim == 3:
            source_image = source_image.unsqueeze(0)
        if source_image.shape[1] not in (1, 3, 4) and source_image.shape[-1] in (1, 3, 4):
            source_image = source_image.permute(0, 3, 1, 2)
        if source_image.shape[1] == 1:
            source_image = source_image.repeat(1, 3, 1, 1)
        if source_image.shape[1] > 3:
            source_image = source_image[:, :3]

        if source_image.max() > 1:
            source_image = source_image / 255.0
        source_image = source_image.clamp(0, 1).mul(2).sub(1)
        return source_image.to(device=device, dtype=dtype)

    def _convert_to_numpy(self, images: torch.Tensor) -> np.ndarray:
        return images.clamp(-1, 1).add(1).div(2).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()

    @torch.no_grad()
    def __call__(
        self,
        source_image: Union[torch.Tensor, Image.Image, List[Image.Image], np.ndarray],
        *,
        context_class: int | str,
        target_class: int | str,
        via_seq: None | str | Sequence[int | str] = None,
        output_type: str = "pil",
        clip_denoised: bool = True,
        return_dict: bool = True,
    ) -> Union[DiffusionRouterPipelineOutput, tuple]:
        """Translate source image with optional route hops."""
        class_names = self._config.class_names
        chain = self._config.chain
        src = parse_class(context_class, class_names)
        dst = parse_class(target_class, class_names)
        route = compose_route(src, dst, via_seq, class_names=class_names, chain=chain)

        x_cur = self._prepare_inputs(source_image, device=self.device, dtype=self.dtype)
        batch_size = x_cur.shape[0]
        nfe_per_hop = len(getattr(self._diffusion, "use_timesteps", []))

        for u, v in zip(route[:-1], route[1:]):
            model_kwargs = {
                "target_class": torch.full((batch_size,), int(v), dtype=torch.int64, device=self.device),
                "context_class": torch.full((batch_size,), int(u), dtype=torch.int64, device=self.device),
            }
            noise = torch.randn_like(x_cur)
            x_t = torch.cat([noise, x_cur], dim=1)
            x_cur = self._sample_fn(
                self.model,
                x_cur.shape,
                noise=x_t,
                clip_denoised=clip_denoised,
                model_kwargs=model_kwargs,
            )

        nfe = nfe_per_hop * max(0, len(route) - 1)
        if output_type == "pil":
            images: Union[List[Image.Image], np.ndarray, torch.Tensor] = numpy_to_pil(self._convert_to_numpy(x_cur))
        elif output_type == "np":
            images = self._convert_to_numpy(x_cur)
        else:
            images = x_cur

        if not return_dict:
            return (images, route, nfe)
        return DiffusionRouterPipelineOutput(images=images, route=route, nfe=nfe)


def load_diffusionrouter_community_pipeline(
    checkpoint_path: str | Path,
    *,
    diffusionrouter_src_path: Optional[str | Path] = None,
    config: Optional[DiffusionRouterConfig] = None,
    device: Optional[str] = None,
) -> DiffusionRouterPipeline:
    """Load DiffusionRouter pipeline from a `.pt` checkpoint."""
    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"DiffusionRouter checkpoint not found: {ckpt}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if config is None:
        config = DiffusionRouterConfig()

    src_path = _ensure_diffusionrouter_path(diffusionrouter_src_path)
    create_model_and_diffusion, model_and_diffusion_defaults = _import_diffusionrouter(src_path)

    args = model_and_diffusion_defaults()
    args.update(
        image_size=config.image_size,
        in_channels=config.in_channels,
        class_cond=config.class_cond,
        num_classes=config.num_classes,
        timestep_respacing=config.timestep_respacing,
    )

    model, diffusion = create_model_and_diffusion(**args)

    raw = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(raw, dict):
        if "state_dict" in raw and isinstance(raw["state_dict"], dict):
            state = raw["state_dict"]
        elif "model_state_dict" in raw and isinstance(raw["model_state_dict"], dict):
            state = raw["model_state_dict"]
        else:
            state = raw
    else:
        raise RuntimeError(f"Unsupported checkpoint format at {ckpt}")

    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items() if torch.is_tensor(v)}
    if not state:
        raise RuntimeError(f"No model tensor weights found in checkpoint: {ckpt}")

    model.load_state_dict(state, strict=False)
    model.eval().to(device)
    return DiffusionRouterPipeline(model=model, diffusion=diffusion, config=config)
