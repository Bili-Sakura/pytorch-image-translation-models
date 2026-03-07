"""Local Diffusion (hallucination-aware diffusion) training and inference."""

from examples.local_diffusion.config import LocalDiffusionConfig
from examples.local_diffusion.train_local_diffusion import LocalDiffusionTrainer

__all__ = ["LocalDiffusionConfig", "LocalDiffusionTrainer"]
