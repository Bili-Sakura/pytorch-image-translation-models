# Credits: Built on open-source libraries and papers acknowledged in README.md citations.
#
"""FWD — Fréchet Wavelet Distance (Veeramacheneni et al., ICLR 2025)."""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

_FWD_REQUIRED_MSG = """
╔══════════════════════════════════════════════════════════════════════════════╗
║  FWD REQUIRES pytorchfwd — NO FALLBACK                                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Install with: pip install pytorchfwd                                       ║
║  https://github.com/BonnBytes/PyTorch-FWD                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


def _save_images_to_dir(images: torch.Tensor, directory: Path) -> None:
    """Save tensor images [0, 1] to directory as PNG."""
    from torchvision.utils import save_image

    images = (images * 255).clamp(0, 255)
    for i, img in enumerate(images.cpu()):
        save_image(img.float() / 255, directory / f"{i:06d}.png")


def compute_fwd(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    device: torch.device | None = None,
    wavelet: str = "Haar",
    max_level: int | None = None,
    batch_size: int = 128,
    **kwargs,
) -> float:
    """Compute Fréchet Wavelet Distance.

    Uses wavelet packet transform instead of Inception for domain-agnostic
    evaluation. Requires ``pytorchfwd``. No fallback.

    Parameters
    ----------
    real_images, fake_images :
        Tensors of shape ``(N, C, H, W)`` in [0, 1].
    device :
        Ignored (FWD runs on CPU via subprocess). Kept for API compatibility.
    wavelet :
        Wavelet type (default: Haar).
    max_level :
        Wavelet decomposition level. Auto-derived from spatial size if None:
        level 4 for 256px, 3 for 128px, 2 for 64px.
    batch_size :
        Batch size for wavelet transform.

    Returns
    -------
    float :
        FWD score. Lower is better.

    References
    ----------
    .. [1] Veeramacheneni et al., "Fréchet Wavelet Distance: A Domain-Agnostic
           Metric for Image Generation", ICLR 2025.
           https://github.com/BonnBytes/PyTorch-FWD
           https://pypi.org/project/pytorchfwd/

    Raises
    ------
    ImportError :
        If pytorchfwd is not installed.
    ValueError :
        If pytorchfwd subprocess fails or output cannot be parsed.
    """
    try:
        import pytorchfwd  # noqa: F401
    except ImportError:
        print(_FWD_REQUIRED_MSG, file=sys.stderr)
        raise ImportError(
            "FWD requires pytorchfwd. Install with: pip install pytorchfwd. "
            "There is no fallback to standard FID."
        ) from None

    h, w = real_images.shape[-2], real_images.shape[-1]
    if max_level is None:
        size = min(h, w)
        max_level = 4 if size >= 256 else (3 if size >= 128 else 2)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        real_dir = root / "real"
        fake_dir = root / "fake"
        real_dir.mkdir()
        fake_dir.mkdir()

        _save_images_to_dir(real_images, real_dir)
        _save_images_to_dir(fake_images, fake_dir)

        cmd = [
            "python",
            "-m",
            "pytorchfwd",
            str(fake_dir),
            str(real_dir),
            "--batch-size",
            str(batch_size),
            "--wavelet",
            wavelet,
            "--max_level",
            str(max_level),
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            cwd=root,
        )

        if result.returncode != 0:
            print(_FWD_REQUIRED_MSG, file=sys.stderr)
            raise ValueError(
                f"pytorchfwd failed (exit code {result.returncode}). "
                f"stderr: {result.stderr[:500] if result.stderr else 'N/A'}. "
                "There is no fallback to standard FID."
            )

        out = result.stdout or result.stderr or ""
        floats = re.findall(r"\d+\.\d+", out)
        if not floats:
            print(_FWD_REQUIRED_MSG, file=sys.stderr)
            raise ValueError(
                "Could not parse FWD score from pytorchfwd output. "
                "There is no fallback to standard FID."
            )

        return float(floats[-1])
