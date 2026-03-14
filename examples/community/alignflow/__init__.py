# Copyright (c) 2026 EarthBridge Team.
# Credits: AlignFlow (Grover et al., AAAI 2020) - https://github.com/ermongroup/alignflow

"""AlignFlow: Cycle-consistent learning from multiple domains via normalizing flows.

AlignFlow uses normalizing flow models for unpaired image-to-image translation,
with cycle-consistency guaranteed by invertible generators. Supports:

* CycleFlow – Single RealNVP flow for src↔tgt (no cycle loss needed)
* Flow2Flow – Two RealNVP flows via shared latent space (GAN + MLE)

Paper: AlignFlow: Cycle Consistent Learning from Multiple Domains via Normalizing Flows
(Grover et al., AAAI 2020) https://arxiv.org/abs/1905.12892
"""

from .config import AlignFlowConfig
from .models import CycleFlow, Flow2Flow
from .pipeline import AlignFlowPipeline, load_alignflow_pipeline
from .train import AlignFlowTrainer

__all__ = [
    "AlignFlowConfig",
    "AlignFlowPipeline",
    "AlignFlowTrainer",
    "CycleFlow",
    "Flow2Flow",
    "load_alignflow_pipeline",
]
