# Credits: Built on open-source libraries and papers acknowledged in README.md citations.
#
"""iFID — Making Reconstruction FID Predictive of Diffusion Generation FID (2026)."""

from __future__ import annotations

import sys
from typing import Any

import torch

_iFID_VAE_REQUIRED_MSG = """
╔══════════════════════════════════════════════════════════════════════════════╗
║  iFID REQUIRES A VAE — NO FALLBACK                                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  iFID must run with a VAE. Use one of:                                       ║
║                                                                              ║
║    1. vae_path="stabilityai/sd-vae-ft-ema"                                  ║
║       (or any diffusers VAE: AutoencoderKL, AutoencoderTiny,                 ║
║        AsymmetricAutoencoderKL, ConsistencyDecoderVAE, etc.)                 ║
║                                                                              ║
║    2. vae=<loaded diffusers VAE model>                                       ║
║                                                                              ║
║  Example: compute_ifid(real, fake, vae_path="stabilityai/sd-vae-ft-ema")     ║
║  Or:      compute_ifid(real, fake, vae=AutoencoderTiny.from_pretrained(...)) ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


def _nearest_neighbor_indices(latents: torch.Tensor) -> torch.Tensor:
    """For each latent z_i, return index j (j != i) of nearest neighbor."""
    N = latents.shape[0]
    dists = torch.cdist(latents, latents, p=2)
    dists.fill_diagonal_(float("inf"))
    return dists.argmin(dim=1)


def _get_latents_from_encode(encode_out: Any) -> torch.Tensor:
    """Extract latents from VAE encode output (supports various diffusers VAEs)."""
    if hasattr(encode_out, "latent_dist"):
        return encode_out.latent_dist.sample()
    if hasattr(encode_out, "latents"):
        return encode_out.latents
    if isinstance(encode_out, torch.Tensor):
        return encode_out
    raise TypeError(
        "VAE encode() output has no latent_dist, latents, or tensor. "
        f"Got type {type(encode_out)}. Use a diffusers VAE (AutoencoderKL, "
        "AutoencoderTiny, AsymmetricAutoencoderKL, etc.)."
    )


def _get_image_from_decode(decode_out: Any) -> torch.Tensor:
    """Extract image tensor from VAE decode output."""
    if hasattr(decode_out, "sample"):
        return decode_out.sample
    if isinstance(decode_out, torch.Tensor):
        return decode_out
    raise TypeError(
        f"VAE decode() output has no .sample or tensor. Got type {type(decode_out)}."
    )


def compute_ifid(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    device: torch.device | None = None,
    *,
    vae: torch.nn.Module | None = None,
    vae_path: str | None = None,
    vae_cls: type | str | None = None,
    interpolation_factor: float = 0.5,
    feature_dim: int = 2048,
    **kwargs,
) -> float:
    """Compute iFID (interpolated FID) for reconstruction/diffusion evaluation.

    Per the paper: for each image in the dataset, retrieve its nearest neighbor
    in latent space, interpolate their latents, decode, and compute FID between
    decoded samples and the original dataset. iFID correlates with diffusion
    generation FID (gFID), unlike standard reconstruction FID (rFID).

    Uses ``real_images`` as the reference dataset; ``fake_images`` is ignored
    (iFID is a single-distribution metric that predicts gFID).

    Parameters
    ----------
    real_images :
        Reference images, shape ``(N, C, H, W)`` in [0, 1]. C must be 3 for
        common VAEs.
    fake_images :
        Ignored (kept for API compatibility with unpaired evaluator).
    device :
        Device for computation.
    vae :
        Pre-loaded diffusers VAE (AutoencoderKL, AutoencoderTiny,
        AsymmetricAutoencoderKL, etc.). Overrides vae_path if both given.
    vae_path :
        HuggingFace model id or local path to load a diffusers VAE.
        Requires diffusers. Ignored if ``vae`` is provided.
    vae_cls :
        VAE class for loading from ``vae_path``. Default: ``AutoencoderKL``.
        Can be any diffusers VAE class, e.g. ``AutoencoderTiny``,
        ``AsymmetricAutoencoderKL``, ``ConsistencyDecoderVAE``.
    interpolation_factor :
        Alpha in z_interp = alpha*z_i + (1-alpha)*z_nn. Default 0.5.
    feature_dim :
        Inception feature dim for the final FID computation.

    Returns
    -------
    float :
        iFID score. Lower is better.

    Raises
    ------
    ValueError :
        If neither ``vae`` nor ``vae_path`` is provided.
    ImportError :
        If diffusers is not installed when loading from ``vae_path``.

    References
    ----------
    .. [1] Xu et al., "Making Reconstruction FID Predictive of Diffusion
           Generation FID", arXiv 2026.
           https://github.com/tongdaxu/Making-rFID-Predictive-of-Diffusion-gFID
    """
    if vae is None and vae_path is None:
        print(_iFID_VAE_REQUIRED_MSG, file=sys.stderr)
        raise ValueError(
            "iFID requires a VAE. Pass vae=<model> or vae_path=<path>. "
            "There is no fallback to standard FID."
        )

    dev = device if device is not None else real_images.device
    real_images = real_images.to(dev)

    if vae is None:
        try:
            from diffusers import AutoencoderKL
        except ImportError:
            raise ImportError(
                "iFID with vae_path requires diffusers. Install with: pip install diffusers"
            ) from None

        _vae_cls_map: dict[str, type] = {"AutoencoderKL": AutoencoderKL}
        try:
            from diffusers import AutoencoderTiny

            _vae_cls_map["AutoencoderTiny"] = AutoencoderTiny
        except ImportError:
            pass
        try:
            from diffusers import AsymmetricAutoencoderKL

            _vae_cls_map["AsymmetricAutoencoderKL"] = AsymmetricAutoencoderKL
        except ImportError:
            pass
        try:
            from diffusers import ConsistencyDecoderVAE

            _vae_cls_map["ConsistencyDecoderVAE"] = ConsistencyDecoderVAE
        except ImportError:
            pass
        if vae_cls is None:
            vae_cls = AutoencoderKL
        elif isinstance(vae_cls, str):
            vae_cls = _vae_cls_map.get(vae_cls, AutoencoderKL)
        vae = vae_cls.from_pretrained(vae_path).to(dev).eval()
    else:
        vae = vae.to(dev).eval()

    # Common diffusers VAEs expect [-1, 1]; our API uses [0, 1]
    x = real_images * 2.0 - 1.0

    with torch.no_grad():
        encode_out = vae.encode(x)
        latents = _get_latents_from_encode(encode_out)

        N, C, H, W = latents.shape
        latents_flat = latents.reshape(N, -1)

        nn_indices = _nearest_neighbor_indices(latents_flat)

        alpha = interpolation_factor
        z_nn = latents[nn_indices]
        z_interp = alpha * latents + (1.0 - alpha) * z_nn

        decode_out = vae.decode(z_interp)
        decoded = _get_image_from_decode(decode_out)

        decoded = (decoded + 1.0) * 0.5
        decoded = decoded.clamp(0.0, 1.0)

    from src.metrics.unpaired.fid import compute_fid

    return compute_fid(
        real_images,
        decoded,
        device=dev,
        feature_dim=feature_dim,
        **kwargs,
    )
