# Copyright (c) 2026 EarthBridge Team.
# Credits: img2img-turbo from Parmar et al. 2024 —
# https://github.com/GaParmar/img2img-turbo

"""Image preprocessing helpers for pix2pix-turbo."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def canny_from_pil(image: Image.Image, low_threshold: int = 100, high_threshold: int = 200) -> Image.Image:
    """Extract Canny edges from a PIL image and return as RGB."""
    edges = cv2.Canny(np.array(image), low_threshold, high_threshold)
    edges = np.concatenate([edges[:, :, None]] * 3, axis=2)
    return Image.fromarray(edges)


__all__ = ["canny_from_pil"]
