# Credits: Built on open-source libraries and papers acknowledged in README.md citations.
#
"""sFID — Sparse / Spatial Fréchet Inception Distance (Nash et al., ICML 2021)."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

_SFID_REQUIRED_MSG = """
╔══════════════════════════════════════════════════════════════════════════════╗
║  sFID REQUIRES pyiqa — NO FALLBACK                                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Install with: pip install pyiqa                                           ║
║  https://github.com/chaofengc/IQA-PyTorch                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


def _save_images_to_dir(images: torch.Tensor, directory: Path) -> None:
    """Save tensor images [0, 1] to directory as PNG."""
    from torchvision.utils import save_image

    images = (images * 255).clamp(0, 255)
    for i, img in enumerate(images.cpu()):
        save_image(img.float() / 255, directory / f"{i:06d}.png")


def compute_sfid(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    feature_dim: int = 2048,
    device: torch.device | None = None,
    **kwargs,
) -> float:
    """Compute Sparse Fréchet Inception Distance (spatial-structure FID).

    sFID uses spatial/dense Inception features (before global pooling) to
    better capture layout and structure. Requires ``pyiqa``. No fallback.

    Parameters
    ----------
    real_images, fake_images :
        Tensors of shape ``(N, C, H, W)`` in [0, 1].
    feature_dim :
        Ignored (sFID uses spatial features). Kept for API compatibility.
    device :
        Device for computation.

    Returns
    -------
    float :
        sFID score. Lower is better.

    Raises
    ------
    ImportError :
        If pyiqa is not installed.

    References
    ----------
    .. [1] Nash et al., "Generating Images with Sparse Representations",
           ICML 2021.
    """
    try:
        import pyiqa
    except ImportError:
        print(_SFID_REQUIRED_MSG, file=sys.stderr)
        raise ImportError(
            "sFID requires pyiqa. Install with: pip install pyiqa. "
            "There is no fallback to standard FID."
        ) from None

    metric = pyiqa.create_metric("sfid")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        real_dir = root / "real"
        fake_dir = root / "fake"
        real_dir.mkdir()
        fake_dir.mkdir()

        _save_images_to_dir(real_images, real_dir)
        _save_images_to_dir(fake_images, fake_dir)

        score = metric(str(fake_dir), str(real_dir))

    if isinstance(score, torch.Tensor):
        score = score.item()
    return float(score)
