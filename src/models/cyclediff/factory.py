# Copyright (c) 2026 EarthBridge Team.
# Credits: CycleDiff (Zou et al., TIP 2026) — https://github.com/ZouShilong1024/CycleDiff

"""Construct CycleDiff models from YAML configuration."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from fvcore.common.config import CfgNode

from src.models.cyclediff.ddm.utils import construct_class_by_name, safe_torch_load


def build_latent_diffusion(model_cfg: CfgNode | Dict[str, Any]) -> Any:
    """Build one :class:`LatentDiffusion` model (domain A or B)."""
    if isinstance(model_cfg, dict):
        model_cfg = CfgNode(model_cfg)
    first_stage_model = construct_class_by_name(**model_cfg.first_stage)
    unet = construct_class_by_name(**model_cfg.unet)
    model_kwargs = {"model": unet, "auto_encoder": first_stage_model, "cfg": model_cfg}
    model_kwargs.update(dict(model_cfg))
    ldm = construct_class_by_name(**model_kwargs)
    return ldm


def build_cycle_nets(cfg: CfgNode) -> Tuple[Any, Any, Any, Any]:
    """Build cycle generators and discriminators."""
    net_g_a = construct_class_by_name(**cfg.net_G)
    net_g_b = construct_class_by_name(**cfg.net_G)
    net_d_a = construct_class_by_name(**cfg.net_D)
    net_d_b = construct_class_by_name(**cfg.net_D)
    return net_g_a, net_g_b, net_d_a, net_d_b


def build_all_models(cfg: CfgNode) -> Dict[str, Any]:
    """Build both LDMs and cycle GAN components from a full training/translation config."""
    return {
        "ldm1": build_latent_diffusion(cfg.model1),
        "ldm2": build_latent_diffusion(cfg.model2),
        "net_G_A": construct_class_by_name(**cfg.net_G),
        "net_G_B": construct_class_by_name(**cfg.net_G),
        "net_D_A": construct_class_by_name(**cfg.net_D),
        "net_D_B": construct_class_by_name(**cfg.net_D),
    }


def load_checkpoint_weights(
    models: Dict[str, Any],
    ckpt_path: str,
    *,
    use_ema: bool = True,
) -> None:
    """Load combined cycle checkpoint (``model-*.pt`` from training)."""
    data = safe_torch_load(ckpt_path, map_location="cpu")

    def _load_ldm(ldm: Any, key: str, ema_key: str) -> None:
        if use_ema and ema_key in data:
            sd = data[ema_key]
            new_sd = {}
            for k, v in sd.items():
                if k.startswith("ema_model."):
                    new_sd[k[10:]] = v
            ldm.load_state_dict(new_sd)
        else:
            ldm.load_state_dict(data[key])
        if "scale_factor" in data.get(key, {}):
            ldm.scale_factor = data[key]["scale_factor"]

    def _load_net(net: Any, key: str, ema_key: str) -> None:
        if use_ema and ema_key in data:
            sd = data[ema_key]
            new_sd = {k[10:]: v for k, v in sd.items() if k.startswith("ema_model.")}
            net.load_state_dict(new_sd)
        else:
            net.load_state_dict(data[key])

    _load_ldm(models["ldm1"], "model1", "ema_d1")
    _load_ldm(models["ldm2"], "model2", "ema_d2")
    _load_net(models["net_G_A"], "net_G_A", "ema_G_A")
    _load_net(models["net_G_B"], "net_G_B", "ema_G_B")
