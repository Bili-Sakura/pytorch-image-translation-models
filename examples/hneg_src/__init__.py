# Credits: Hneg-SRC from Jung et al. CVPR 2022 — https://github.com/jcy132/Hneg_SRC

"""Hneg-SRC (Semantic Relation Contrastive) training and inference."""

from examples.hneg_src.config import HnegSRCConfig
from examples.hneg_src.train_hneg_src import HnegSRCTrainer

__all__ = ["HnegSRCConfig", "HnegSRCTrainer"]
