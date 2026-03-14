# Credits: Built on open-source libraries and papers acknowledged in README.md citations.
#
"""CMMD — Metric Maximum Mean Discrepancy (Jayasumana et al., CVPR 2024)."""

from __future__ import annotations

from PIL import Image

import torch


def _mmd_rbf(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> float:
    """Unbiased MMD with RBF kernel."""
    n, m = x.shape[0], y.shape[0]
    if n < 2 or m < 2:
        return float("inf")
    x = x.double()
    y = y.double()
    gamma = 1.0 / (2 * sigma**2)
    k_xx = torch.exp(-gamma * torch.cdist(x, x).pow(2))
    k_yy = torch.exp(-gamma * torch.cdist(y, y).pow(2))
    k_xy = torch.exp(-gamma * torch.cdist(x, y).pow(2))
    k_xx = (k_xx.sum() - n) / (n * (n - 1) + 1e-8)
    k_yy = (k_yy.sum() - m) / (m * (m - 1) + 1e-8)
    k_xy = k_xy.sum() / (n * m)
    return float((k_xx + k_yy - 2 * k_xy).clamp(min=0).sqrt())


def compute_cmmd(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    model_id: str = "openai/clip-vit-base-patch32",
    device: torch.device | None = None,
    **kwargs,
) -> float:
    """Compute CMMD (CLIP Maximum Mean Discrepancy).

    Uses CLIP image embeddings instead of Inception. Better aligns with
    human judgment and text-image models.

    Parameters
    ----------
    real_images, fake_images :
        Tensors of shape ``(N, C, H, W)`` in [0, 1].
    model_id :
        HuggingFace model id for CLIP.

    Returns
    -------
    float :
        CMMD score. Lower is better.

    References
    ----------
    .. [1] Jayasumana et al., "Rethinking FID: Towards a Better Evaluation
           Metric for Image Generation", CVPR 2024.
    """
    from transformers import CLIPModel, CLIPProcessor

    dev = device if device is not None else real_images.device
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(dev).eval()

    def embed_batch(imgs: torch.Tensor) -> torch.Tensor:
        arr = (imgs.permute(0, 2, 3, 1).cpu().numpy() * 255).clip(0, 255).astype("uint8")
        pil = [Image.fromarray(x) for x in arr]
        inp = processor(images=pil, return_tensors="pt", padding=True)
        inp = {k: v.to(dev) for k, v in inp.items()}
        with torch.no_grad():
            return model.get_image_features(**inp).float()

    feats_real = embed_batch(real_images)
    feats_fake = embed_batch(fake_images)
    return _mmd_rbf(feats_real, feats_fake)
