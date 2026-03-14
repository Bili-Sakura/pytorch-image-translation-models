# Copyright (c) 2026 EarthBridge Team.
# Credits: AlignFlow (Grover et al., AAAI 2020) - https://github.com/ermongroup/alignflow

"""Inference pipeline for AlignFlow (CycleFlow, Flow2Flow)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Literal, Union

import numpy as np
import torch
from PIL import Image

from diffusers import DiffusionPipeline
from diffusers.utils import BaseOutput

from .config import AlignFlowConfig
from .models import CycleFlow, Flow2Flow


class AlignFlowPipelineOutput(BaseOutput):
    """Output of the AlignFlow pipeline."""

    images: Union[List[Image.Image], np.ndarray, torch.Tensor]


def load_alignflow_pipeline(
    checkpoint_dir: str | Path,
    *,
    model_name: Literal["CycleFlow", "Flow2Flow"] = "CycleFlow",
    device: str | torch.device = "cuda",
) -> "AlignFlowPipeline":
    """Load an AlignFlow pipeline from a checkpoint directory.

    The checkpoint directory should contain:
    - config.json: Model configuration
    - model.safetensors or model.pt: Model weights

    Parameters
    ----------
    checkpoint_dir : str | Path
        Path to the checkpoint directory.
    model_name : Literal["CycleFlow", "Flow2Flow"]
        Model variant to load.
    device : str | torch.device
        Device to run inference on.

    Returns
    -------
    AlignFlowPipeline
        Loaded pipeline for inference.

    Example
    -------
    ::

        from examples.community.alignflow import load_alignflow_pipeline

        pipe = load_alignflow_pipeline(
            "/path/to/alignflow-checkpoint",
            model_name="CycleFlow",
            device="cuda",
        )
        out = pipe(source_image=image, output_type="pil")
    """
    path = Path(checkpoint_dir)
    config_path = path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        cfg_dict = json.load(f)

    config = AlignFlowConfig.from_dict(cfg_dict)
    config.gpu_ids = [0] if str(device) != "cpu" else []
    config.is_training = False

    if model_name == "CycleFlow":
        model = CycleFlow(config)
    elif model_name == "Flow2Flow":
        model = Flow2Flow(config)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    weights_path = path / "model.safetensors"
    if not weights_path.exists():
        weights_path = path / "model.pt"
    if not weights_path.exists():
        ckpt_dir = next(path.glob("checkpoint-epoch-*"), None)
        if ckpt_dir:
            weights_path = ckpt_dir / "alignflow.pt"

    if weights_path.exists():
        loaded = torch.load(weights_path, map_location="cpu")
        state = loaded.get("model", loaded)
        if isinstance(state, dict):
            model.load_state_dict(state, strict=False)

    model.eval()
    model.to(device)
    return AlignFlowPipeline(model=model)


class AlignFlowPipeline(DiffusionPipeline):
    """Inference pipeline for AlignFlow (CycleFlow / Flow2Flow).

    Performs unpaired image-to-image translation using normalizing flows.
    """

    def __init__(self, model: CycleFlow | Flow2Flow) -> None:
        super().__init__()
        self.register_modules(model=model)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _prepare_image(
        self,
        image: torch.Tensor | Image.Image | List[Image.Image],
    ) -> torch.Tensor:
        """Convert input to tensor in [-1, 1] range."""
        if isinstance(image, Image.Image):
            image = [image]
        if isinstance(image, list) and image and isinstance(image[0], Image.Image):
            tensors = []
            for img in image:
                arr = np.array(img, dtype=np.float32)
                if arr.max() > 1.0:
                    arr = arr / 255.0
                if arr.ndim == 2:
                    arr = arr[:, :, np.newaxis]
                t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
                tensors.append(t)
            image = torch.cat(tensors, dim=0)
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if image.min() >= 0 and image.max() <= 1.0:
            image = image * 2 - 1
        elif image.max() > 1.0:
            image = image.float() / 255.0 * 2 - 1
        return image.to(device=self.device)

    @torch.no_grad()
    def __call__(
        self,
        source_image: torch.Tensor | Image.Image | List[Image.Image],
        output_type: str = "pil",
        return_dict: bool = True,
    ) -> AlignFlowPipelineOutput | tuple:
        """Translate source images to target domain.

        Parameters
        ----------
        source_image : torch.Tensor | Image.Image | list of Image.Image
            Source image(s).
        output_type : str
            "pil", "np", or "pt".
        return_dict : bool
            If True, return AlignFlowPipelineOutput.

        Returns
        -------
        AlignFlowPipelineOutput or tuple
            Translated images.
        """
        x = self._prepare_image(source_image)
        self.model.set_inputs(x)
        self.model.test()
        images = self.model.src2tgt.clamp(-1, 1)

        if output_type == "pil":
            images = self._to_pil(images)
        elif output_type == "np":
            images = (images + 1) / 2
            images = images.clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()

        if not return_dict:
            return (images,)
        return AlignFlowPipelineOutput(images=images)

    @staticmethod
    def _to_pil(images: torch.Tensor) -> List[Image.Image]:
        """Convert tensor in [-1, 1] to PIL."""
        images = (images + 1) / 2
        images = images.clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).round().astype(np.uint8)
        return [Image.fromarray(img) for img in images]
