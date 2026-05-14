# Copyright (c) 2026 EarthBridge Team.
# Credits: EGSDE (Zhao et al., NeurIPS 2022) — https://github.com/Bili-Sakura/EGSDE-diffusers

"""Constants for the EGSDE community pipeline."""

from __future__ import annotations

from typing import Literal

# Two-domain translation presets shipped with the upstream repo (profiles/*/args.py).
EGSDE_TASKS: tuple[str, ...] = ("cat2dog", "wild2dog", "male2female")

EGSDETask = Literal["cat2dog", "wild2dog", "male2female"]
