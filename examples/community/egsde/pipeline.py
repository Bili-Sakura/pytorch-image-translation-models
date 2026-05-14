# Copyright (c) 2026 EarthBridge Team.
# Credits: EGSDE (Zhao et al., NeurIPS 2022) — https://github.com/Bili-Sakura/EGSDE-diffusers

"""Diffusers-style wrapper for EGSDE (Energy-Guided Stochastic Differential Equations)."""

from __future__ import annotations

import copy
import importlib.util
import sys
from argparse import Namespace
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from diffusers import DiffusionPipeline
from diffusers.utils import BaseOutput, numpy_to_pil

from .model import EGSDE_TASKS


def _ensure_egsde_path(egsde_src_path: Optional[str | Path]) -> Path:
    """Resolve EGSDE-diffusers checkout root."""
    if egsde_src_path is not None:
        p = Path(egsde_src_path)
        if p.is_dir() and (p / "runners" / "egsde.py").is_file() and (p / "guided_diffusion" / "script_util.py").is_file():
            return p.resolve()
        raise FileNotFoundError(
            f"EGSDE-diffusers source not found at {p}. Expected runners/egsde.py and guided_diffusion/script_util.py."
        )

    root = Path(__file__).resolve().parents[4]
    candidates = [
        root / "EGSDE-diffusers",
        root / "projects" / "EGSDE-diffusers",
        Path.cwd() / "EGSDE-diffusers",
        Path.cwd() / "projects" / "EGSDE-diffusers",
    ]
    for p in candidates:
        if p.is_dir() and (p / "runners" / "egsde.py").is_file():
            return p.resolve()

    raise FileNotFoundError(
        "EGSDE-diffusers source not found. Clone https://github.com/Bili-Sakura/EGSDE-diffusers.git and pass "
        "egsde_src_path, or place it at ./EGSDE-diffusers or ./projects/EGSDE-diffusers."
    )


def _inject_egsde_path(egsde_root: Path) -> None:
    s = str(egsde_root.resolve())
    if s not in sys.path:
        sys.path.insert(0, s)


def _load_profile_args(egsde_root: Path, task: str) -> Namespace:
    if task not in EGSDE_TASKS:
        raise ValueError(f"Unknown EGSDE task {task!r}. Choose one of {EGSDE_TASKS}.")
    args_path = egsde_root / "profiles" / task / "args.py"
    if not args_path.is_file():
        raise FileNotFoundError(f"Missing profile args: {args_path}")
    spec = importlib.util.spec_from_file_location(f"egsde_profile_{task}", args_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load profile module from {args_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    args = copy.copy(mod.argsall)
    root = egsde_root.resolve()

    def _abs(p: str) -> str:
        pp = Path(p)
        return str(pp.resolve()) if pp.is_absolute() else str((root / pp).resolve())

    args.ckpt = _abs(args.ckpt)
    args.dsepath = _abs(args.dsepath)
    args.config_path = _abs(args.config_path)
    if hasattr(args, "testdata_path"):
        args.testdata_path = _abs(args.testdata_path)
    return args


def _load_config_namespace(config_path: str | Path) -> Namespace:
    from tool.utils import dict2namespace

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return dict2namespace(raw)


def _normalize_state_dict(states: object) -> object:
    if not isinstance(states, dict):
        return states
    keys = list(states.keys())
    if not keys:
        return states
    if all(isinstance(k, str) and k.startswith("module.") for k in keys):
        return {k[7:]: v for k, v in states.items()}
    return states


def _prepare_source_batch(
    source_image: Union[torch.Tensor, Image.Image, List[Image.Image], np.ndarray],
    *,
    image_size: int,
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

    if source_image.dtype != torch.float32:
        source_image = source_image.float()

    if source_image.max() > 1.0:
        source_image = source_image / 255.0
    source_image = source_image.clamp(0, 1).mul(2).sub(1)

    if source_image.shape[-1] != image_size or source_image.shape[-2] != image_size:
        source_image = F.interpolate(
            source_image, size=(image_size, image_size), mode="bicubic", align_corners=False
        ).clamp(-1.0, 1.0)

    return source_image.to(device=device, dtype=dtype)


class EGSDEPipelineOutput(BaseOutput):
    """Output of :class:`EGSDEPipeline`."""

    images: Union[List[Image.Image], np.ndarray, torch.Tensor]


class EGSDEPipeline(DiffusionPipeline):
    """Unpaired image-to-image translation with EGSDE energy guidance."""

    model_cpu_offload_seq = "score_model"

    def __init__(self, args: Namespace, config: Namespace, egsde_root: Path, device: str | torch.device) -> None:
        self.args = args
        self.config = config
        self._egsde_root = Path(egsde_root)
        self._device = torch.device(device)
        self._die_batch = -1
        self._down: Optional[torch.nn.Module] = None
        self._up: Optional[torch.nn.Module] = None

        from runners.egsde import EGSDE

        schedule = EGSDE(args, config, device=self._device)
        self.betas = schedule.betas
        self.logvar = schedule.logvar

        score_model, dse = self._build_score_and_dse(args, config, self._device)
        super().__init__()
        self.register_modules(score_model=score_model, dse=dse)

    @staticmethod
    def _build_score_and_dse(args: Namespace, config: Namespace, device: torch.device) -> tuple[torch.nn.Module, torch.nn.Module]:
        from guided_diffusion.script_util import create_dse, create_model

        if args.diffusionmodel == "ADM":
            score_model = create_model(
                image_size=config.data.image_size,
                num_class=config.model.num_class,
                num_channels=config.model.num_channels,
                num_res_blocks=config.model.num_res_blocks,
                learn_sigma=config.model.learn_sigma,
                class_cond=config.model.class_cond,
                attention_resolutions=config.model.attention_resolutions,
                num_heads=config.model.num_heads,
                num_head_channels=config.model.num_head_channels,
                num_heads_upsample=config.model.num_heads_upsample,
                use_scale_shift_norm=config.model.use_scale_shift_norm,
                dropout=config.model.dropout,
                resblock_updown=config.model.resblock_updown,
                use_fp16=config.model.use_fp16,
                use_new_attention_order=config.model.use_new_attention_order,
            )
        elif args.diffusionmodel == "DDPM":
            from models.ddpm import Model

            score_model = Model(config)
        else:
            raise ValueError(f"Unsupported diffusionmodel {args.diffusionmodel!r}; expected ADM or DDPM.")

        states = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        states = _normalize_state_dict(states)
        score_model.load_state_dict(states, strict=True)
        score_model.to(device).eval()

        dse = create_dse(
            image_size=config.data.image_size,
            num_class=config.dse.num_class,
            classifier_use_fp16=config.dse.classifier_use_fp16,
            classifier_width=config.dse.classifier_width,
            classifier_depth=config.dse.classifier_depth,
            classifier_attention_resolutions=config.dse.classifier_attention_resolutions,
            classifier_use_scale_shift_norm=config.dse.classifier_use_scale_shift_norm,
            classifier_resblock_updown=config.dse.classifier_resblock_updown,
            classifier_pool=config.dse.classifier_pool,
            phase=args.phase,
        )
        dse_states = torch.load(args.dsepath, map_location="cpu", weights_only=False)
        dse_states = _normalize_state_dict(dse_states)
        dse.load_state_dict(dse_states, strict=True)
        dse.to(device).eval()

        return score_model, dse

    def _ensure_die(self, batch_size: int) -> tuple[torch.nn.Module, torch.nn.Module]:
        from functions.resizer import Resizer

        if batch_size == self._die_batch and self._down is not None and self._up is not None:
            return self._down, self._up

        h = int(self.config.data.image_size)
        shape = (batch_size, 3, h, h)
        shape_d = (batch_size, 3, int(h / self.args.down_N), int(h / self.args.down_N))
        dev = self.device
        self._down = Resizer(shape, 1 / self.args.down_N).to(dev)
        self._up = Resizer(shape_d, self.args.down_N).to(dev)
        self._die_batch = batch_size
        return self._down, self._up

    @property
    def device(self) -> torch.device:
        return next(self.score_model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.score_model.parameters()).dtype

    def to(self, *args, **kwargs):  # noqa: ANN002
        out = super().to(*args, **kwargs)
        self.betas = self.betas.to(self.device)
        self.logvar = self.logvar.to(self.device)
        self._die_batch = -1
        self._down = None
        self._up = None
        return out

    def _tensor_to_output(self, y: torch.Tensor, output_type: str) -> Union[List[Image.Image], np.ndarray, torch.Tensor]:
        y = y.clamp(0.0, 1.0)
        if output_type == "pil":
            return numpy_to_pil(y.cpu().permute(0, 2, 3, 1).numpy())
        if output_type == "np":
            return y.cpu().permute(0, 2, 3, 1).numpy()
        if output_type in ("pt", "latent"):
            return y
        raise ValueError(f"Unsupported output_type {output_type!r}; expected pil, np, or pt.")

    def __call__(
        self,
        source_image: Union[torch.Tensor, Image.Image, List[Image.Image], np.ndarray],
        *,
        output_type: str = "pil",
        return_dict: bool = True,
    ) -> Union[EGSDEPipelineOutput, tuple]:
        """Run VP-EGSDE sampling (source domain → target domain).

        Autograd stays enabled because energy guidance computes gradients in ``egsde_sample``.
        """
        from datasets import inverse_rescale
        from functions.denoising import egsde_sample

        dev = self.device
        dt = self.dtype
        x0 = _prepare_source_batch(
            source_image, image_size=int(self.config.data.image_size), device=dev, dtype=dt
        )
        n = x0.shape[0]
        down, up = self._ensure_die(n)
        die = (down, up)

        y0 = x0
        model = self.score_model
        dse = self.dse
        args = self.args
        total_noise_levels = int(args.t)

        for _it in range(int(args.sample_step)):
            e = torch.randn_like(y0)
            a = (1 - self.betas).cumprod(dim=0)
            y = y0 * a[total_noise_levels - 1].sqrt() + e * (1.0 - a[total_noise_levels - 1]).sqrt()
            for i in reversed(range(total_noise_levels)):
                t = (torch.ones(n, device=dev, dtype=torch.long) * i).long()
                xt = x0 * a[i].sqrt() + e * (1.0 - a[i]).sqrt()
                y = egsde_sample(
                    y=y,
                    dse=dse,
                    ls=args.ls,
                    die=die,
                    li=args.li,
                    t=t,
                    model=model,
                    logvar=self.logvar,
                    betas=self.betas,
                    xt=xt,
                    s1=args.s1,
                    s2=args.s2,
                    model_name=args.diffusionmodel,
                )
            y0 = y

        y_out = inverse_rescale(y0)
        images = self._tensor_to_output(y_out, output_type)
        if not return_dict:
            return (images,)
        return EGSDEPipelineOutput(images=images)


def load_egsde_community_pipeline(
    egsde_src_path: Optional[str | Path] = None,
    *,
    task: Optional[str] = None,
    device: str | torch.device = "cuda",
    ckpt: Optional[str] = None,
    dsepath: Optional[str] = None,
    config_path: Optional[str | Path] = None,
    diffusionmodel: Optional[str] = None,
    t: Optional[int] = None,
    ls: Optional[float] = None,
    li: Optional[float] = None,
    s1: Optional[str] = None,
    s2: Optional[str] = None,
    down_N: Optional[int] = None,
    sample_step: Optional[int] = None,
    phase: Optional[str] = None,
    batch_size: Optional[int] = None,
) -> EGSDEPipeline:
    """Load EGSDE inference weights from a local EGSDE-diffusers checkout.

    Either pass ``task`` (loads ``profiles/<task>/args.py``) or supply full checkpoint paths
    and ``config_path`` plus ``diffusionmodel``.
    """
    root = _ensure_egsde_path(egsde_src_path)
    _inject_egsde_path(root)

    if task is not None:
        args = _load_profile_args(root, task)
    else:
        missing = []
        for name, val in (
            ("ckpt", ckpt),
            ("dsepath", dsepath),
            ("config_path", config_path),
            ("diffusionmodel", diffusionmodel),
        ):
            if val is None:
                missing.append(name)
        if missing:
            raise ValueError(
                "When task is None, the following arguments are required: " + ", ".join(missing) + ". "
                "Alternatively pass task='cat2dog', 'wild2dog', or 'male2female'."
            )
        args = Namespace(
            ckpt=str(Path(ckpt).resolve()),
            dsepath=str(Path(dsepath).resolve()),
            config_path=str(Path(config_path).resolve()),
            diffusionmodel=diffusionmodel,
            t=int(t) if t is not None else 500,
            ls=float(ls) if ls is not None else 500.0,
            li=float(li) if li is not None else 2.0,
            s1=s1 or "cosine",
            s2=s2 or "neg_l2",
            phase=phase or "test",
            sample_step=int(sample_step) if sample_step is not None else 1,
            batch_size=int(batch_size) if batch_size is not None else 4,
            down_N=int(down_N) if down_N is not None else 32,
        )

    if ckpt is not None:
        args.ckpt = str(Path(ckpt).resolve())
    if dsepath is not None:
        args.dsepath = str(Path(dsepath).resolve())
    if config_path is not None:
        args.config_path = str(Path(config_path).resolve())
    if diffusionmodel is not None:
        args.diffusionmodel = diffusionmodel
    if t is not None:
        args.t = int(t)
    if ls is not None:
        args.ls = float(ls)
    if li is not None:
        args.li = float(li)
    if s1 is not None:
        args.s1 = s1
    if s2 is not None:
        args.s2 = s2
    if down_N is not None:
        args.down_N = int(down_N)
    if sample_step is not None:
        args.sample_step = int(sample_step)
    if phase is not None:
        args.phase = phase
    if batch_size is not None:
        args.batch_size = int(batch_size)

    config = _load_config_namespace(args.config_path)
    return EGSDEPipeline(args, config, root, device)
