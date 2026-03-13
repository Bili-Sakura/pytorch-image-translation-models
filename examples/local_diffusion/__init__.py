# Credits: Local Diffusion from Kim et al. "Tackling Structural Hallucination in Image Translation with Local Diffusion" ECCV 2024 Oral.
"""Local Diffusion (hallucination-aware diffusion) training and inference."""

from examples.local_diffusion.config import LocalDiffusionConfig
from examples.local_diffusion.train_local_diffusion import LocalDiffusionTrainer

__all__ = ["LocalDiffusionConfig", "LocalDiffusionTrainer"]
