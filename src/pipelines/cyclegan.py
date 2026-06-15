# Credits: CycleGAN (Zhu et al., ICCV 2017) — https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

"""CycleGAN single-pass inference pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Union

import numpy as np
import torch
from PIL import Image

from diffusers import DiffusionPipeline
from diffusers.utils import BaseOutput, pt_to_pil

from src.models.cyclegan_pix2pix import CycleGANGenerator, create_generator


@dataclass
class CycleGANPipelineOutput(BaseOutput):
    """Output class for CycleGAN pipeline."""

    images: Union[List[Image.Image], np.ndarray, torch.Tensor]


class CycleGANPipeline(DiffusionPipeline):
    """Inference pipeline for CycleGAN unpaired image translation.

    Supports translation in both directions via ``G_A`` (A→B) and ``G_B`` (B→A).
  """

    def __init__(
        self,
        generator_a: CycleGANGenerator,
        generator_b: CycleGANGenerator | None = None,
    ) -> None:
        super().__init__()
        self.register_modules(generator_a=generator_a, generator_b=generator_b)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        subfolder_a: str = "generator_a",
        subfolder_b: str = "generator_b",
        device: str | torch.device = "cpu",
        torch_dtype: torch.dtype | None = None,
        **kwargs,
    ) -> "CycleGANPipeline":
        """Load from HF-style checkpoint with ``generator_a/`` and optional ``generator_b/``."""
        root = Path(pretrained_model_name_or_path)
        gen_a = CycleGANGenerator.from_pretrained(root, subfolder=subfolder_a, **kwargs)
        gen_a = gen_a.eval().to(device=device)
        if torch_dtype is not None:
            gen_a = gen_a.to(dtype=torch_dtype)

        gen_b = None
        gen_b_path = root / subfolder_b
        if gen_b_path.exists():
            gen_b = CycleGANGenerator.from_pretrained(root, subfolder=subfolder_b, **kwargs)
            gen_b = gen_b.eval().to(device=device)
            if torch_dtype is not None:
                gen_b = gen_b.to(dtype=torch_dtype)

        return cls(generator_a=gen_a, generator_b=gen_b)

    @property
    def device(self) -> torch.device:
        return next(self.generator_a.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.generator_a.parameters()).dtype

    def prepare_inputs(
        self,
        image: Union[torch.Tensor, Image.Image, List[Image.Image]],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
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
        source_image: Union[torch.Tensor, Image.Image, List[Image.Image]],
        direction: Literal["a2b", "b2a"] = "a2b",
        output_type: str = "pil",
        return_dict: bool = True,
    ) -> Union[CycleGANPipelineOutput, tuple]:
        """Translate source images in the requested direction."""
        device = self.device
        dtype = self.dtype
        x = self.prepare_inputs(source_image, device, dtype)

        if direction == "a2b":
            generator = self.generator_a
        elif direction == "b2a":
            if self.generator_b is None:
                raise ValueError("generator_b is not loaded; cannot translate B→A")
            generator = self.generator_b
        else:
            raise ValueError(f"Unknown direction: {direction}")

        fake = generator(x).clamp(-1, 1)

        if output_type == "pil":
            images = pt_to_pil(fake)
        elif output_type == "np":
            images = self._convert_to_numpy(fake)
        else:
            images = fake

        if not return_dict:
            return (images,)
        return CycleGANPipelineOutput(images=images)

    @staticmethod
    def _convert_to_numpy(images: torch.Tensor) -> np.ndarray:
        images = (images + 1) / 2
        images = images.clamp(0, 1)
        return images.cpu().permute(0, 2, 3, 1).numpy()


def load_cyclegan_pipeline(
    checkpoint_dir: Union[str, Path],
    *,
    direction: Literal["a2b", "b2a"] = "a2b",
    netG: str = "resnet_9blocks",
    norm: str = "instance",
    input_nc: int = 3,
    output_nc: int = 3,
    ngf: int = 64,
    no_dropout: bool = True,
    device: str | torch.device = "cpu",
    torch_dtype: torch.dtype | None = None,
) -> CycleGANPipeline:
    """Load CycleGAN pipeline from HF checkpoint or upstream ``.pth`` weights.

    Upstream checkpoints from
    ``scripts/download_cyclegan_model.sh`` use ``latest_net_G.pth`` for
    single-direction test models, or ``latest_net_G_A.pth`` / ``latest_net_G_B.pth``
    for full CycleGAN training checkpoints.
    """
    from src.models.cyclegan_pix2pix import load_upstream_generator_state

    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    gen_a_dir = checkpoint_dir / "generator_a"
    if gen_a_dir.exists() and (gen_a_dir / "config.json").exists():
        return CycleGANPipeline.from_pretrained(
            checkpoint_dir,
            device=device,
            torch_dtype=torch_dtype,
        )

    gen_a = create_generator(
        input_nc=input_nc,
        output_nc=output_nc,
        ngf=ngf,
        netG=netG,
        norm=norm,
        use_dropout=not no_dropout,
    )
    gen_b = None

    g_a_path = checkpoint_dir / "latest_net_G_A.pth"
    g_b_path = checkpoint_dir / "latest_net_G_B.pth"
    single_g_path = checkpoint_dir / "latest_net_G.pth"

    if g_a_path.exists():
        load_upstream_generator_state(gen_a, g_a_path)
    elif single_g_path.exists() and direction == "a2b":
        load_upstream_generator_state(gen_a, single_g_path)
    else:
        raise FileNotFoundError(
            f"No CycleGAN generator checkpoint found in {checkpoint_dir}. "
            "Expected generator_a/, latest_net_G_A.pth, or latest_net_G.pth"
        )

    if g_b_path.exists():
        gen_b = create_generator(
            input_nc=output_nc,
            output_nc=input_nc,
            ngf=ngf,
            netG=netG,
            norm=norm,
            use_dropout=not no_dropout,
        )
        load_upstream_generator_state(gen_b, g_b_path)
    elif single_g_path.exists() and direction == "b2a":
        load_upstream_generator_state(gen_a, single_g_path)

    gen_a = gen_a.eval().to(device=device)
    if gen_b is not None:
        gen_b = gen_b.eval().to(device=device)
    if torch_dtype is not None:
        gen_a = gen_a.to(dtype=torch_dtype)
        if gen_b is not None:
            gen_b = gen_b.to(dtype=torch_dtype)

    return CycleGANPipeline(generator_a=gen_a, generator_b=gen_b)


__all__ = [
    "CycleGANPipeline",
    "CycleGANPipelineOutput",
    "load_cyclegan_pipeline",
]
