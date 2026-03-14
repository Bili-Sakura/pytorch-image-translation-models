# Credits: CUT from Park et al. "Contrastive Learning for Unpaired Image-to-Image Translation" ECCV 2020.
"""CUT (Contrastive Unpaired Translation) training and inference."""

from examples.cut.config import CUTConfig
from examples.cut.train_cut import CUTTrainer

__all__ = ["CUTConfig", "CUTTrainer"]
