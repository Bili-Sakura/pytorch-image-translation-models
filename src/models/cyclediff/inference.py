# Copyright (c) 2026 EarthBridge Team.
# Credits: CycleDiff (Zou et al., TIP 2026) — https://github.com/ZouShilong1024/CycleDiff

"""CycleDiff unpaired translation sampling (A→B and B→A)."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

# Tasks where the first domain test set is translated toward the second.
_A2B_TASK_PREFIXES = (
    "cat2dog",
    "wild2dog",
    "male2female",
    "sem2rgb",
    "depth2rgb",
    "edge2rgb",
    "summer2winter",
    "horse2zebra",
    "young2old",
    "map2satellite",
    "label2cityscape",
)


def is_a2b_task(task: str) -> bool:
    return any(task.startswith(p) for p in _A2B_TASK_PREFIXES)


def _build_target_c_list(
    src_img: torch.Tensor,
    ldm_encode: nn.Module,
    net_g: nn.Module,
) -> List[torch.Tensor]:
    """Reverse diffusion on source domain and map conditions through ``net_g``."""
    device = src_img.device
    c_list, noise = ldm_encode.reverse_q_sample_c_list_concat(src_img)
    target_input: List[torch.Tensor] = []

    step = 1.0 / ldm_encode.sampling_timesteps
    rho = 1.0
    step_indices = torch.arange(ldm_encode.sampling_timesteps, dtype=torch.float32, device=device)
    t_steps = (
        ldm_encode.sigma_max ** (1 / rho)
        + step_indices / (ldm_encode.sampling_timesteps - 1)
        * (step - ldm_encode.sigma_max ** (1 / rho))
    ) ** rho
    t_steps = reversed(torch.cat([t_steps, torch.zeros_like(t_steps[:1])]))
    t_list = list(t_steps)
    batch = src_img.shape[0]

    for i in range(len(c_list) - 1):
        target_input.append(net_g(c_list[i], t_list[i + 1].repeat(batch)))
    target_input.append(net_g(c_list[-1], t_list[-1].repeat(batch)))
    target_input.append(noise)
    return target_input


@torch.no_grad()
def translate_batch(
    source_image: torch.Tensor,
    ldm_encode: nn.Module,
    ldm_decode: nn.Module,
    net_g: nn.Module,
) -> torch.Tensor:
    """Translate a batch of RGB images in ``[-1, 1]`` through the cycle diffusion bridge."""
    target_input = _build_target_c_list(source_image, ldm_encode, net_g)
    return ldm_decode.sample_from_c_list(batch_size=source_image.shape[0], c_list=target_input)


@torch.no_grad()
def translate_for_task(
    source_image: torch.Tensor,
    models: dict,
    task: str,
) -> torch.Tensor:
    """Translate using task name to select model1→model2 or model2→model1."""
    if is_a2b_task(task):
        return translate_batch(
            source_image,
            models["ldm1"],
            models["ldm2"],
            models["net_G_A"],
        )
    return translate_batch(
        source_image,
        models["ldm2"],
        models["ldm1"],
        models["net_G_B"],
    )
