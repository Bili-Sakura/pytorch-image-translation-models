"""I2SB example: configuration and training."""

from examples.i2sb.config import (
    TaskConfig,
    rgb2ir_config,
    sar2eo_config,
    sar2ir_config,
    sar2rgb_config,
)
from examples.i2sb.trainer import I2SBTrainer

__all__ = [
    "TaskConfig",
    "sar2eo_config",
    "rgb2ir_config",
    "sar2ir_config",
    "sar2rgb_config",
    "I2SBTrainer",
]
