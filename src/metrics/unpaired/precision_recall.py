"""P&R — Improved Precision and Recall (Kynkäänniemi et al., NeurIPS 2019)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch


def compute_precision_recall(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    device: torch.device | None = None,
    **kwargs,
) -> tuple[float, float]:
    """Compute Improved Precision and Recall via torch-fidelity.

    Saves tensors to a temp dir and calls torch_fidelity. Returns (precision,
    recall). Higher is better for both.

    Parameters
    ----------
    real_images, fake_images :
        Tensors of shape ``(N, C, H, W)`` in [0, 1].

    Returns
    -------
    tuple[float, float] :
        (precision, recall).

    References
    ----------
    .. [1] Kynkäänniemi et al., "Improved Precision and Recall Metric for
           Assessing Generative Models", NeurIPS 2019.
    """
    try:
        import torch_fidelity
    except ImportError as e:
        raise ImportError(
            "P&R requires torch-fidelity. Install with: pip install torch-fidelity"
        ) from e

    from torchvision.utils import save_image

    dev = device if device is not None else real_images.device
    real_images = (real_images * 255).clamp(0, 255).byte()
    fake_images = (fake_images * 255).clamp(0, 255).byte()

    with tempfile.TemporaryDirectory() as tmp:
        real_dir = Path(tmp) / "real"
        fake_dir = Path(tmp) / "fake"
        real_dir.mkdir()
        fake_dir.mkdir()
        for i, img in enumerate(real_images.cpu()):
            save_image(img.float() / 255, real_dir / f"{i:06d}.png")
        for i, img in enumerate(fake_images.cpu()):
            save_image(img.float() / 255, fake_dir / f"{i:06d}.png")

        metrics = torch_fidelity.calculate_metrics(
            str(fake_dir),  # input1 = generated
            str(real_dir),  # input2 = real
            prc=True,
            prc_neighborhood=5,
            verbose=False,
        )
    return (
        metrics["precision"],
        metrics["recall"],
    )


def compute_precision(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    device: torch.device | None = None,
    **kwargs,
) -> float:
    """Improved Precision only."""
    p, _ = compute_precision_recall(real_images, fake_images, device, **kwargs)
    return p


def compute_recall(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    device: torch.device | None = None,
    **kwargs,
) -> float:
    """Improved Recall only."""
    _, r = compute_precision_recall(real_images, fake_images, device, **kwargs)
    return r


def compute_pr_f1(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    device: torch.device | None = None,
    **kwargs,
) -> float:
    """F1 (harmonic mean) of Improved Precision and Recall."""
    p, r = compute_precision_recall(real_images, fake_images, device, **kwargs)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0
