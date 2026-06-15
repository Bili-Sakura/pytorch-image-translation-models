# Copyright (c) 2026 EarthBridge Team.
# Credits: NEGCUT from Wang et al. ICCV 2021 — https://github.com/WeilunWang/NEGCUT

"""NEGCUT training entry point."""

from examples.negcut.config import NEGCUTConfig
from examples.negcut.train_negcut import NEGCUTTrainer

__all__ = ["NEGCUTConfig", "NEGCUTTrainer"]
