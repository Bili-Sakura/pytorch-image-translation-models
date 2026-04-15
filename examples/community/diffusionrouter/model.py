# Copyright (c) 2026 EarthBridge Team.
# Credits: DiffusionRouter (kvmduc) - https://github.com/kvmduc/DiffusionRouter

"""Routing utilities and config for the DiffusionRouter community pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

DIFFUSIONROUTER_CLASS_NAMES: dict[int, str] = {
    0: "color",
    1: "edge",
    2: "gray",
    3: "depth",
}
DIFFUSIONROUTER_DEFAULT_CHAIN: tuple[int, ...] = (2, 0, 1, 3)


@dataclass
class DiffusionRouterConfig:
    """Configuration for loading DiffusionRouter checkpoints."""

    image_size: int = 64
    in_channels: int = 6
    class_cond: bool = True
    num_classes: int = 4
    timestep_respacing: str = "1000"
    use_ddim: bool = True
    class_names: Mapping[int, str] = None  # type: ignore[assignment]
    chain: Sequence[int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.class_names is None:
            self.class_names = DIFFUSIONROUTER_CLASS_NAMES
        if self.chain is None:
            self.chain = DIFFUSIONROUTER_DEFAULT_CHAIN


def parse_class(value: int | str, class_names: Mapping[int, str]) -> int:
    """Parse a class ID or class name into class ID."""
    if isinstance(value, int):
        if value in class_names:
            return value
        raise ValueError(f"Unknown class index: {value}. Valid: {sorted(class_names.keys())}")

    v = value.strip().lower()
    if v.isdigit():
        idx = int(v)
        if idx in class_names:
            return idx
        raise ValueError(f"Unknown class index: {idx}. Valid: {sorted(class_names.keys())}")

    for idx, name in class_names.items():
        if name.lower() == v:
            return idx
    raise ValueError(f"Unknown class name: {value}. Valid: {list(class_names.values())}")


def parse_via_seq(via_seq: None | str | Sequence[int | str], class_names: Mapping[int, str]) -> None | list[int] | str:
    """Parse route sequence, returning None, 'auto', or a list of class indices."""
    if via_seq is None:
        return None

    if isinstance(via_seq, str):
        s = via_seq.strip().lower()
        if s in ("", "none", "-2"):
            return None
        if s in ("auto", "-1"):
            return "auto"
        items = [x.strip() for x in s.split(",") if x.strip()]
        return [parse_class(x, class_names) for x in items]

    parsed = [parse_class(x, class_names) for x in via_seq]
    return parsed if parsed else None


def auto_route(src: int, dst: int, chain: Sequence[int]) -> list[int]:
    """Compute shortest route on a fixed chain topology."""
    if src == dst:
        return [src]
    pos = {cid: i for i, cid in enumerate(chain)}
    if src not in pos or dst not in pos:
        return [src, dst]
    i0, i1 = pos[src], pos[dst]
    step = 1 if i1 > i0 else -1
    return list(chain[i0 : i1 + step : step])


def compose_route(
    src: int,
    dst: int,
    via_seq: None | str | Sequence[int | str],
    *,
    class_names: Mapping[int, str],
    chain: Sequence[int],
) -> list[int]:
    """Compose final route [src, ..., dst] with optional intermediate hops."""
    parsed = parse_via_seq(via_seq, class_names)
    if parsed is None:
        route = [src, dst]
    elif parsed == "auto":
        route = auto_route(src, dst, chain)
    else:
        route = [src, *parsed, dst]

    dedup = [route[0]]
    for cid in route[1:]:
        if cid != dedup[-1]:
            dedup.append(cid)
    return dedup
