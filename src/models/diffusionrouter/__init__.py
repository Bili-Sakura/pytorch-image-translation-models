# Copyright (c) 2026 EarthBridge Team.
# Credits: DiffusionRouter (kvmduc) - https://github.com/kvmduc/DiffusionRouter

"""DiffusionRouter routing utilities and configuration."""

from src.models.diffusionrouter.routing import (
    DIFFUSIONROUTER_CLASS_NAMES,
    DIFFUSIONROUTER_DEFAULT_CHAIN,
    DiffusionRouterConfig,
    auto_route,
    compose_route,
    parse_class,
    parse_via_seq,
)

__all__ = [
    "DIFFUSIONROUTER_CLASS_NAMES",
    "DIFFUSIONROUTER_DEFAULT_CHAIN",
    "DiffusionRouterConfig",
    "auto_route",
    "compose_route",
    "parse_class",
    "parse_via_seq",
]
