# Credits: UNSB from Kim et al. "Unpaired Image-to-Image Translation via Neural Schrödinger Bridge" ICLR 2024.
"""UNSB (Unpaired Neural Schrödinger Bridge) training and inference."""

from examples.unsb.config import UNSBConfig
from examples.unsb.train_unsb import UNSBTrainer

__all__ = ["UNSBConfig", "UNSBTrainer"]
