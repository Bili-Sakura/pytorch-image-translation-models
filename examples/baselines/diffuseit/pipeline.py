# Copyright (c) 2026 EarthBridge Team.
# Credits: DiffuseIT (Kwon & Ye, ICLR 2023) - https://github.com/cyclomon/DiffuseIT

"""DiffuseIT baseline pipeline for diffusion-based image translation.

Wraps the DiffuseIT ImageEditor for text-guided and image-guided translation.
Expects DiffuseIT repo cloned at projects/DiffuseIT (or diffuseit_src_path).
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
import torch
from PIL import Image

from diffusers import DiffusionPipeline
from diffusers.utils import BaseOutput


def _ensure_diffuseit_path(diffuseit_src_path: Optional[str | Path]) -> Path:
    """Resolve DiffuseIT source path. Default: workspace/projects/DiffuseIT."""
    if diffuseit_src_path is not None:
        path = Path(diffuseit_src_path)
        if path.exists():
            return path.resolve()
        raise FileNotFoundError(f"DiffuseIT source not found: {path}")

    candidates = [
        Path(__file__).resolve().parents[4] / "DiffuseIT",  # .../projects/DiffuseIT
        Path.cwd().parent / "DiffuseIT",  # when cwd is projects/pytorch-image-translation-models
        Path.cwd() / "projects" / "DiffuseIT",
        Path.cwd() / "DiffuseIT",
    ]
    for p in candidates:
        if p.exists() and (p / "optimization" / "image_editor.py").exists():
            return p.resolve()

    raise FileNotFoundError(
        "DiffuseIT source not found. Clone from https://github.com/cyclomon/DiffuseIT "
        "and set diffuseit_src_path or place at projects/DiffuseIT"
    )


@dataclass
class DiffuseITPipelineOutput(BaseOutput):
    """Output of the DiffuseIT baseline pipeline.

    Attributes
    ----------
    images : list[PIL.Image.Image] | np.ndarray | torch.Tensor
        Translated images.
    nfe : int
        Effective sampling steps (diffusion iterations).
    """

    images: Any
    nfe: int = 0


class DiffuseITPipeline(DiffusionPipeline):
    """Image translation pipeline using DiffuseIT (ICLR 2023).

    Supports text-guided (prompt + source text) and image-guided (target_image)
    translation. Uses pre-trained diffusion models from the DiffuseIT checkpoint layout.
    """

    def __init__(
        self,
        editor: Any,
        diffuseit_path: Path,
    ) -> None:
        super().__init__()
        self._editor = editor
        self._diffuseit_path = diffuseit_path

    @property
    def device(self) -> torch.device:
        return self._editor.device

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        diffuseit_src_path: Optional[str | Path] = None,
        use_ffhq: bool = False,
        image_size: int = 256,
        timestep_respacing: str = "100",
        skip_timesteps: int = 40,
        device: Optional[str] = None,
        **kwargs,
    ) -> "DiffuseITPipeline":
        """Load DiffuseIT pipeline from DiffuseIT checkpoint layout.

        Expects checkpoint at ``pretrained_model_name_or_path``:
        - For ImageNet256: ``256x256_diffusion_uncond.pt``
        - For FFHQ: ``ffhq_10m.pt``
        - For ImageNet512: ``512x512_diffusion.pt`` (in checkpoints/)

        Parameters
        ----------
        pretrained_model_name_or_path : str | Path
            Path to DiffuseIT root (containing checkpoints/) or to checkpoint dir.
        diffuseit_src_path : str | Path, optional
            Path to DiffuseIT repo. Default: projects/DiffuseIT.
        use_ffhq : bool
            Use FFHQ face model.
        image_size : int
            Output resolution (256 or 512).
        timestep_respacing : str
            DDIM respacing (e.g. "100").
        skip_timesteps : int
            Timesteps to skip for inpainting start.
        device : str, optional
            Device to load onto.
        """
        # Resolve DiffuseIT root: prefer pretrained path if it's the repo
        p = Path(pretrained_model_name_or_path)
        if p.exists() and (p / "optimization" / "image_editor.py").exists():
            diffuseit_path = p.resolve()
        else:
            diffuseit_path = _ensure_diffuseit_path(diffuseit_src_path)
        import sys
        diffuseit_str = str(diffuseit_path)
        if diffuseit_str not in sys.path:
            sys.path.insert(0, diffuseit_str)

        from optimization.image_editor import ImageEditor

        # Build minimal args for ImageEditor
        class Args:
            pass
        args = Args()
        args.init_image = ""
        args.target_image = None
        args.prompt = ""
        args.source = ""
        args.skip_timesteps = skip_timesteps
        args.ddim = True
        args.timestep_respacing = timestep_respacing
        args.model_output_size = image_size
        args.clip_models = ["ViT-B/32"]
        args.aug_num = 8
        args.diff_iter = 50
        args.clip_guidance_lambda = 2000.0
        args.lambda_trg = 2000.0
        args.l2_trg_lambda = 3000.0
        args.range_lambda = 200.0
        args.vit_lambda = 1.0
        args.lambda_ssim = 1000.0
        args.lambda_dir_cls = 100.0
        args.lambda_contra_ssim = 200.0
        args.id_lambda = 100.0
        args.resample_num = 10
        args.seed = None
        args.gpu_id = 0
        if device:
            dev_str = str(device)
            if dev_str.startswith("cuda:") and dev_str[5:].isdigit():
                args.gpu_id = int(dev_str.split(":")[-1])
        args.output_path = tempfile.mkdtemp()
        args.output_file = "output.png"
        args.iterations_num = 1
        args.batch_size = 1
        args.use_ffhq = use_ffhq
        args.use_prog_contrast = False
        args.use_range_restart = True
        args.use_colormatch = True
        args.use_noise_aug_all = True
        args.regularize_content = False

        orig_cwd = os.getcwd()
        try:
            os.chdir(diffuseit_path)
            editor = ImageEditor(args)
        finally:
            os.chdir(orig_cwd)

        pipeline = cls(editor=editor, diffuseit_path=diffuseit_path)
        if device:
            pipeline.to(device)
        return pipeline

    def to(self, device) -> "DiffuseITPipeline":
        dev = torch.device(device) if isinstance(device, str) else device
        self._editor.device = dev
        self._editor.model.to(dev)
        if hasattr(self._editor, "VIT_LOSS") and self._editor.VIT_LOSS is not None:
            self._editor.VIT_LOSS.to(dev)
        return self

    def __call__(
        self,
        source_image: Union[Image.Image, np.ndarray, torch.Tensor],
        *,
        prompt: Optional[str] = None,
        source: Optional[str] = None,
        target_image: Optional[Union[Image.Image, np.ndarray, torch.Tensor]] = None,
        use_colormatch: bool = True,
        use_range_restart: bool = True,
        use_noise_aug_all: bool = True,
        iterations_num: int = 1,
        output_type: str = "pil",
    ) -> DiffuseITPipelineOutput:
        """Run DiffuseIT image translation.

        Parameters
        ----------
        source_image : PIL.Image | np.ndarray | torch.Tensor
            Input image to translate.
        prompt : str, optional
            Target text prompt (text-guided mode).
        source : str, optional
            Source domain text (text-guided mode).
        target_image : PIL.Image | np.ndarray | torch.Tensor, optional
            Target style image (image-guided mode).
        use_colormatch : bool
            Apply color matching to target (image-guided).
        use_range_restart : bool
            Use range restart for stability.
        use_noise_aug_all : bool
            Use noise augmentation for VIT losses.
        iterations_num : int
            Number of diffusion iterations.
        output_type : str
            "pil" or "pt".

        Returns
        -------
        DiffuseITPipelineOutput
        """
        if prompt is None and target_image is None:
            raise ValueError("Provide either prompt (text-guided) or target_image (image-guided)")

        pipe = self._editor
        pipe.args.use_colormatch = use_colormatch
        pipe.args.use_range_restart = use_range_restart
        pipe.args.use_noise_aug_all = use_noise_aug_all
        pipe.args.iterations_num = iterations_num

        if prompt is not None:
            pipe.args.prompt = prompt
        if source is not None:
            pipe.args.source = source
        if target_image is not None:
            pipe.args.target_image = None  # set path below

        # Convert input to PIL if needed
        if isinstance(source_image, torch.Tensor):
            from torchvision.transforms.functional import to_pil_image
            if source_image.dim() == 3:
                source_image = source_image.unsqueeze(0)
            source_pil = to_pil_image(source_image[0].add(1).div(2).clamp(0, 1))
        elif isinstance(source_image, np.ndarray):
            source_pil = Image.fromarray(source_image).convert("RGB")
        else:
            source_pil = source_image.convert("RGB") if hasattr(source_image, "convert") else source_image

        image_size = pipe.model_config.get("image_size", 256) or 256
        source_pil = source_pil.resize((image_size, image_size), Image.LANCZOS)

        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = Path(tmpdir) / "input.png"
            out_path = Path(tmpdir) / "output"
            out_path.mkdir(exist_ok=True)

            source_pil.save(in_path)
            pipe.args.init_image = str(in_path)
            pipe.args.output_path = str(out_path)
            pipe.args.output_file = "out.png"

            if target_image is not None:
                if isinstance(target_image, torch.Tensor):
                    from torchvision.transforms.functional import to_pil_image
                    t = target_image
                    if t.dim() == 3:
                        t = t.unsqueeze(0)
                    tg_pil = to_pil_image(t[0].add(1).div(2).clamp(0, 1))
                elif isinstance(target_image, np.ndarray):
                    tg_pil = Image.fromarray(target_image).convert("RGB")
                else:
                    tg_pil = target_image.convert("RGB") if hasattr(target_image, "convert") else target_image
                tg_pil = tg_pil.resize((image_size, image_size), Image.LANCZOS)
                tg_path = Path(tmpdir) / "target.png"
                tg_pil.save(tg_path)
                pipe.args.target_image = str(tg_path)
            else:
                pipe.args.target_image = None

            orig_cwd = os.getcwd()
            try:
                os.chdir(self._diffuseit_path)
                pipe.edit_image_by_prompt()
            finally:
                os.chdir(orig_cwd)

            # Find output (ImageEditor saves with iteration/batch in name)
            outs = list(out_path.glob("out_i_*_b_0.png"))
            if not outs:
                outs = list(out_path.glob("*.png"))
            if not outs:
                raise RuntimeError("DiffuseIT produced no output")

            result_path = sorted(outs)[-1]
            result_pil = Image.open(result_path).convert("RGB")

        total_t = getattr(pipe.diffusion, "num_timesteps", 1000)
        num_steps = total_t - pipe.args.skip_timesteps
        if pipe.args.ddim and hasattr(pipe.args, "timestep_respacing"):
            num_steps = len(
                timestep_respacing_to_steps(
                    pipe.args.timestep_respacing, total_t
                )
            )

        images: List[Any] = [result_pil]
        if output_type == "pt":
            from torchvision.transforms.functional import to_tensor
            images = [to_tensor(img).unsqueeze(0).mul(2).sub(1) for img in images]

        return DiffuseITPipelineOutput(images=images, nfe=num_steps)


def timestep_respacing_to_steps(respacing: str, total: int) -> list:
    """Parse timestep_respacing string to step list."""
    parts = respacing.replace(" ", "").split(",")
    if len(parts) == 1 and parts[0].isdigit():
        n = int(parts[0])
        return list(range(0, total, max(1, total // n)))
    return list(range(total))


def load_diffuseit_baseline_pipeline(
    checkpoint_path: str | Path,
    *,
    diffuseit_src_path: Optional[str | Path] = None,
    use_ffhq: bool = False,
    image_size: int = 256,
    device: str = "cuda",
    **kwargs,
) -> DiffuseITPipeline:
    """Load DiffuseIT baseline pipeline from checkpoint.

    Parameters
    ----------
    checkpoint_path : str | Path
        Path to DiffuseIT root (with checkpoints/) or checkpoint directory.
    diffuseit_src_path : str | Path, optional
        Path to DiffuseIT repo. Default: projects/DiffuseIT.
    use_ffhq : bool
        Use FFHQ model.
    image_size : int
        Output resolution.
    device : str
        Device to use.

    Returns
    -------
    DiffuseITPipeline
    """
    return DiffuseITPipeline.from_pretrained(
        checkpoint_path,
        diffuseit_src_path=diffuseit_src_path,
        use_ffhq=use_ffhq,
        image_size=image_size,
        device=device,
        **kwargs,
    )
