# Copyright (c) 2026 EarthBridge Team.
# Credits: CycleDiff (Zou et al., TIP 2026) — https://github.com/ZouShilong1024/CycleDiff

"""CycleDiff pipeline for unpaired image-to-image translation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
import torch
from PIL import Image

from diffusers.utils import pt_to_pil

from src.models.cyclediff.config_loader import load_cfg_node
from src.models.cyclediff.factory import build_all_models, load_checkpoint_weights
from src.models.cyclediff.inference import translate_batch, translate_for_task


@dataclass
class CycleDiffPipelineOutput:
    """Output of :class:`CycleDiffPipeline`."""

    images: Union[List[Image.Image], np.ndarray, torch.Tensor]


class CycleDiffPipeline:
    """Unpaired image translation with CycleDiff (dual LDM + cycle generators).

    Parameters
    ----------
    models : dict
        Keys ``ldm1``, ``ldm2``, ``net_G_A``, ``net_G_B`` (and optionally discriminators).
    task : str
        Translation task name from config (e.g. ``cat2dog``) used to pick direction.
    device : torch.device
        Inference device.
    """

    def __init__(
        self,
        models: dict,
        *,
        task: str = "cat2dog",
        device: torch.device | str = "cpu",
    ) -> None:
        self.models = models
        self.task = task
        self._device = torch.device(device)

        for key in ("ldm1", "ldm2", "net_G_A", "net_G_B"):
            self.models[key].to(self._device)
            self.models[key].eval()

    @property
    def device(self) -> torch.device:
        return self._device

    @classmethod
    def from_config(
        cls,
        config_path: Union[str, Path],
        *,
        ckpt_path: Optional[Union[str, Path]] = None,
        task: Optional[str] = None,
        use_ema: bool = True,
        device: Union[str, torch.device] = "cpu",
    ) -> "CycleDiffPipeline":
        """Build pipeline from YAML config and optional checkpoint."""
        cfg = load_cfg_node(config_path)
        models = build_all_models(cfg)
        if ckpt_path is None and hasattr(cfg, "sampler") and cfg.sampler.ckpt_path:
            ckpt_path = cfg.sampler.ckpt_path
        if ckpt_path is not None:
            load_checkpoint_weights(models, str(ckpt_path), use_ema=use_ema)
        if task is None and hasattr(cfg, "sampler"):
            task = cfg.sampler.task
        return cls(models, task=task or "cat2dog", device=device)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, Path],
        *,
        config_path: Optional[Union[str, Path]] = None,
        task: Optional[str] = None,
        use_ema: bool = True,
        device: Union[str, torch.device] = "cpu",
        **kwargs,
    ) -> "CycleDiffPipeline":
        """Load from a checkpoint directory or explicit config + ``.pt`` file."""
        path = Path(pretrained_model_name_or_path)
        if path.suffix == ".pt":
            ckpt_path = path
            if config_path is None:
                raise ValueError("config_path is required when pretrained_model_name_or_path is a .pt file")
            return cls.from_config(config_path, ckpt_path=ckpt_path, task=task, use_ema=use_ema, device=device)

        cfg = config_path or path / "config.yaml"
        if not Path(cfg).exists():
            raise FileNotFoundError(f"CycleDiff config not found: {cfg}")
        ckpt = None
        if (path / "model.pt").exists():
            ckpt = path / "model.pt"
        elif list(path.glob("model-*.pt")):
            ckpt = sorted(path.glob("model-*.pt"))[-1]
        return cls.from_config(cfg, ckpt_path=ckpt, task=task, use_ema=use_ema, device=device)

    def _prepare_input(
        self,
        image: Union[torch.Tensor, Image.Image, List[Image.Image], np.ndarray],
        size: Optional[tuple[int, int]] = None,
    ) -> torch.Tensor:
        if isinstance(image, Image.Image):
            image = [image]
        if isinstance(image, list) and image and isinstance(image[0], Image.Image):
            tensors = []
            for img in image:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                arr = np.array(img, dtype=np.float32) / 255.0
                t = torch.from_numpy(arr).permute(2, 0, 1)
                tensors.append(t)
            image = torch.stack(tensors)
        elif isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
            if image.ndim == 3:
                image = image.permute(2, 0, 1).unsqueeze(0)
        if isinstance(image, torch.Tensor) and image.ndim == 3:
            image = image.unsqueeze(0)
        if image.shape[-1] in (1, 3, 4) and image.shape[1] not in (1, 3, 4):
            image = image.permute(0, 3, 1, 2)
        if image.max() <= 1.0:
            image = image * 2 - 1
        if size is not None:
            image = torch.nn.functional.interpolate(image, size=size, mode="bilinear", align_corners=False)
        return image.to(self.device)

    @torch.no_grad()
    def __call__(
        self,
        source_image: Union[torch.Tensor, Image.Image, List[Image.Image], np.ndarray],
        *,
        task: Optional[str] = None,
        output_type: str = "pil",
        return_dict: bool = True,
    ) -> Union[CycleDiffPipelineOutput, tuple]:
        """Translate source image(s) to the target domain."""
        task = task or self.task
        x = self._prepare_input(source_image)
        out = translate_for_task(x, self.models, task)
        out = out.clamp(-1, 1)
        if output_type == "pil":
            out = pt_to_pil((out + 1) * 0.5)
        elif output_type == "np":
            out = ((out + 1) * 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
        if not return_dict:
            return (out,)
        return CycleDiffPipelineOutput(images=out)

    @torch.no_grad()
    def translate_a2b(self, source_image: torch.Tensor) -> torch.Tensor:
        """Explicit A→B translation (model1 encode, G_A, model2 decode)."""
        return translate_batch(
            source_image.to(self.device),
            self.models["ldm1"],
            self.models["ldm2"],
            self.models["net_G_A"],
        )

    @torch.no_grad()
    def translate_b2a(self, source_image: torch.Tensor) -> torch.Tensor:
        """Explicit B→A translation."""
        return translate_batch(
            source_image.to(self.device),
            self.models["ldm2"],
            self.models["ldm1"],
            self.models["net_G_B"],
        )


def load_cyclediff_pipeline(
    config_path: Union[str, Path],
    *,
    ckpt_path: Optional[Union[str, Path]] = None,
    task: Optional[str] = None,
    device: Union[str, torch.device] = "cpu",
) -> CycleDiffPipeline:
    """Load :class:`CycleDiffPipeline` from YAML and checkpoint."""
    return CycleDiffPipeline.from_config(
        config_path, ckpt_path=ckpt_path, task=task, device=device
    )
