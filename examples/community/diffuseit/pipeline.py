# Copyright (c) 2026 EarthBridge Team.
# Credits: DiffuseIT (Kwon & Ye, ICLR 2023) - https://github.com/cyclomon/DiffuseIT

"""DiffuseIT community pipeline for diffusion-based image translation.

Loads from BiliSakura/DiffuseIT-ckpt layout (after convert_ckpt_to_diffuseit).
"""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
import torch
from PIL import Image

from diffusers import DiffusionPipeline
from diffusers.utils import BaseOutput

# Bundled DiffuseIT implementation (no external repo dependency)
_BUNDLED_DIFFUSEIT = Path(__file__).resolve().parent / "_vendor" / "DiffuseIT"


def _get_diffuseit_path() -> Path:
    """Return bundled DiffuseIT path. No external repo or clone needed."""
    path = _BUNDLED_DIFFUSEIT.resolve()
    if not (path / "optimization" / "image_editor.py").exists():
        raise FileNotFoundError(
            f"Bundled DiffuseIT not found at {path}. "
            "Ensure examples/community/diffuseit/_vendor/DiffuseIT contains the full DiffuseIT source."
        )
    return path


def _setup_id_model(ckpt_dir: Path, diffuseit_path: Path) -> None:
    """Copy ArcFace id_model from self-contained ckpt to DiffuseIT for use_ffhq."""
    id_src_dir = ckpt_dir / "id_model"
    id_dst_dir = diffuseit_path / "id_model"
    id_dst_dir.mkdir(exist_ok=True)
    dst = id_dst_dir / "model_ir_se50.pth"

    src_pt = id_src_dir / "model_ir_se50.pth"
    src_safetensors = id_src_dir / "model_ir_se50.safetensors"
    if src_pt.exists():
        if not dst.exists() or dst.resolve() != src_pt.resolve():
            try:
                dst.unlink(missing_ok=True)
                dst.symlink_to(src_pt.resolve())
            except OSError:
                shutil.copy2(src_pt, dst)
    elif src_safetensors.exists():
        from safetensors.torch import load_file
        state = load_file(str(src_safetensors))
        torch.save(state, dst)


def _setup_checkpoint_for_editor(
    ckpt_dir: Path,
    diffuseit_path: Path,
    use_ffhq: bool,
    image_size: int,
) -> None:
    """Copy BiliSakura checkpoint to DiffuseIT checkpoints/ for ImageEditor."""
    ckpt_dir = Path(ckpt_dir)
    checkpoints_dir = diffuseit_path / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    if use_ffhq:
        ckpt_name = "ffhq_10m.pt"
    elif image_size == 512:
        ckpt_name = "512x512_diffusion.pt"
    else:
        ckpt_name = "256x256_diffusion_uncond.pt"

    dst = checkpoints_dir / ckpt_name
    src = ckpt_dir / "diffusion_pytorch_model.pt"
    if src.exists():
        if not dst.exists() or dst.resolve() != src.resolve():
            try:
                dst.unlink(missing_ok=True)
                dst.symlink_to(src.resolve())
            except OSError:
                shutil.copy2(src, dst)
        return

    safetensors_path = ckpt_dir / "unet" / "diffusion_pytorch_model.safetensors"
    if safetensors_path.exists():
        from safetensors.torch import load_file
        state = load_file(str(safetensors_path))
        torch.save(state, dst)
        return

    raise FileNotFoundError(
        f"No checkpoint found in {ckpt_dir}. Run convert_ckpt_to_diffuseit first."
    )


@dataclass
class DiffuseITPipelineOutput(BaseOutput):
    """Output of the DiffuseIT community pipeline."""

    images: Any
    nfe: int = 0


class DiffuseITPipeline(DiffusionPipeline):
    """Image translation pipeline using DiffuseIT (ICLR 2023)."""

    def __init__(self, editor: Any, diffuseit_path: Path) -> None:
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
        use_ffhq: bool = False,
        image_size: int = 256,
        timestep_respacing: str = "100",
        skip_timesteps: int = 40,
        device: Optional[str] = None,
        **kwargs,
    ) -> "DiffuseITPipeline":
        """Load DiffuseIT pipeline from BiliSakura checkpoint.

        Expects path like ``/path/to/DiffuseIT-ckpt/imagenet256-uncond`` with
        ``unet/config.json``, ``diffusion_pytorch_model.pt`` (from convert_ckpt_to_diffuseit).
        Uses bundled DiffuseIT implementation; no external repo required.
        """
        ckpt_path = Path(pretrained_model_name_or_path)
        diffuseit_path = _get_diffuseit_path()

        # BiliSakura layout: setup checkpoint for the bundled editor
        if (ckpt_path / "unet" / "config.json").exists() or (
            ckpt_path / "diffusion_pytorch_model.pt"
        ).exists():
            _setup_checkpoint_for_editor(
                ckpt_path, diffuseit_path, use_ffhq, image_size
            )
            if use_ffhq:
                _setup_id_model(ckpt_path, diffuseit_path)

        import sys
        diffuseit_str = str(diffuseit_path)
        if diffuseit_str not in sys.path:
            sys.path.insert(0, diffuseit_str)

        from optimization.image_editor import ImageEditor

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
        """Run DiffuseIT image translation."""
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
            pipe.args.target_image = None

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

            outs = list(out_path.glob("out_i_*_b_0.png"))
            if not outs:
                outs = list(out_path.glob("*.png"))
            if not outs:
                raise RuntimeError("DiffuseIT produced no output")
            result_pil = Image.open(sorted(outs)[-1]).convert("RGB")

        total_t = getattr(pipe.diffusion, "num_timesteps", 1000)
        num_steps = total_t - pipe.args.skip_timesteps
        if pipe.args.ddim and hasattr(pipe.args, "timestep_respacing"):
            num_steps = len(
                _timestep_respacing_to_steps(pipe.args.timestep_respacing, total_t)
            )

        images: List[Any] = [result_pil]
        if output_type == "pt":
            from torchvision.transforms.functional import to_tensor
            images = [to_tensor(img).unsqueeze(0).mul(2).sub(1) for img in images]

        return DiffuseITPipelineOutput(images=images, nfe=num_steps)


def _timestep_respacing_to_steps(respacing: str, total: int) -> list:
    parts = respacing.replace(" ", "").split(",")
    if len(parts) == 1 and parts[0].isdigit():
        n = int(parts[0])
        return list(range(0, total, max(1, total // n)))
    return list(range(total))


def load_diffuseit_community_pipeline(
    checkpoint_path: str | Path,
    *,
    use_ffhq: bool = False,
    image_size: int = 256,
    device: str = "cuda",
    **kwargs,
) -> DiffuseITPipeline:
    """Load DiffuseIT community pipeline from BiliSakura checkpoint.
    Uses bundled implementation; no external DiffuseIT repo required.
    """
    return DiffuseITPipeline.from_pretrained(
        checkpoint_path,
        use_ffhq=use_ffhq,
        image_size=image_size,
        device=device,
        **kwargs,
    )
