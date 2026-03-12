# Copyright (c) 2026 EarthBridge Team.
# Credits: DDIB (Su et al., ICLR 2023) https://github.com/suxuann/ddib

"""DDIB community pipeline for OpenAI/guided_diffusion-style checkpoints."""

from examples.community.ddib.model import OpenAIDDIBUNet
from examples.community.ddib.pipeline import load_ddib_community_pipeline

__all__ = ["OpenAIDDIBUNet", "load_ddib_community_pipeline"]
