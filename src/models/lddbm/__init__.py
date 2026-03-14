# Copyright (c) 2026 EarthBridge Team.
# Credits: Berman et al., NeurIPS 2025 - https://github.com/boschresearch/Multimodal-Distribution-Translation-MDT
# "Towards General Modality Translation with Contrastive and Predictive Latent Diffusion Bridge"

"""MDT (Multimodal Distribution Translation) / LDDBM models.

Latent diffusion bridge for modality translation: super-resolution, multi-view to 3D, etc.
"""

from src.models.lddbm.mm_dist_trans import ModalityTranslationBridge
from src.models.lddbm.models_loader import create_bridge, create_decoder, create_encoder
from src.models.lddbm.names import (
    BridgeModelsTyps,
    Decoders,
    Encoders,
    ReconstructionLoss,
    TrainingStrategy,
)

__all__ = [
    "ModalityTranslationBridge",
    "create_bridge",
    "create_decoder",
    "create_encoder",
    "BridgeModelsTyps",
    "Decoders",
    "Encoders",
    "ReconstructionLoss",
    "TrainingStrategy",
]
