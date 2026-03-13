# Credits: Built on open-source libraries and papers acknowledged in README.md citations.
"""DISTS — Deep Image Structure and Texture Similarity (Ding et al., IEEE TPAMI 2022)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class L2Pooling(nn.Module):
    """L2 pooling layer used in DISTS backbone."""

    def __init__(
        self,
        filter_size: int = 5,
        stride: int = 2,
        channels: int | None = None,
    ) -> None:
        super().__init__()
        self.padding = (filter_size - 2) // 2
        self.stride = stride
        self.channels = channels
        a = np.hanning(filter_size)[1:-1]
        g = torch.tensor(a[:, None] * a[None, :], dtype=torch.float32)
        g = g / g.sum()
        self.register_buffer(
            "filter",
            g[None, None, :, :].repeat((self.channels or 1, 1, 1, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x**2
        out = F.conv2d(
            x, self.filter, stride=self.stride, padding=self.padding, groups=x.shape[1]
        )
        return (out + 1e-12).sqrt()


class DISTSBackbone(nn.Module):
    """VGG16 backbone with L2 pooling for DISTS."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        from torchvision.models import vgg16
        from torchvision.models import VGG16_Weights

        vgg = vgg16(
            weights=VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        ).features
        self.stage1 = nn.Sequential(*[vgg[i] for i in range(0, 4)])
        self.stage2 = nn.Sequential(
            L2Pooling(channels=64),
            *[vgg[i] for i in range(5, 9)],
        )
        self.stage3 = nn.Sequential(
            L2Pooling(channels=128),
            *[vgg[i] for i in range(10, 16)],
        )
        self.stage4 = nn.Sequential(
            L2Pooling(channels=256),
            *[vgg[i] for i in range(17, 23)],
        )
        self.stage5 = nn.Sequential(
            L2Pooling(channels=512),
            *[vgg[i] for i in range(24, 30)],
        )
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        h = self.stage1(x)
        h1 = h
        h = self.stage2(h)
        h2 = h
        h = self.stage3(h)
        h3 = h
        h = self.stage4(h)
        h4 = h
        h = self.stage5(h)
        h5 = h
        return [x, h1, h2, h3, h4, h5]


class DISTS(nn.Module):
    """DISTS metric: Deep Image Structure and Texture Similarity."""

    chns = [3, 64, 128, 256, 512, 512]

    def __init__(
        self,
        load_weights: bool = True,
        weights_path: str | None = None,
        pretrained_vgg: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = DISTSBackbone(pretrained=pretrained_vgg)
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1),
        )
        self.register_parameter(
            "alpha",
            nn.Parameter(torch.randn(1, sum(self.chns), 1, 1) * 0.01 + 0.1),
        )
        self.register_parameter(
            "beta",
            nn.Parameter(torch.randn(1, sum(self.chns), 1, 1) * 0.01 + 0.1),
        )
        if load_weights and weights_path:
            self._load_weights(weights_path)
        elif load_weights:
            self._try_load_hf_weights()

    def _load_weights(self, path: str) -> None:
        state = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(state, dict):
            if "alpha" in state and "beta" in state:
                self.alpha.data = state["alpha"].to(self.alpha.dtype)
                self.beta.data = state["beta"].to(self.beta.dtype)
            elif "state_dict" in state:
                self._load_weights_from_state_dict(state["state_dict"])

    def _load_weights_from_state_dict(self, sd: dict) -> None:
        for key in ("alpha", "beta"):
            if key in sd:
                getattr(self, key).data = sd[key].to(getattr(self, key).dtype)

    def _try_load_hf_weights(self) -> None:
        try:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(
                repo_id="chaofengc/IQA-PyTorch-Weights",
                filename="DISTS_weights-f5e65c96.pth",
            )
            self._load_weights(path)
        except Exception:
            pass

    def forward_once(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = (x - self.mean) / self.std
        return self.backbone(x)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        batch_average: bool = True,
    ) -> torch.Tensor:
        with torch.no_grad():
            feats0 = self.forward_once(x)
            feats1 = self.forward_once(y)

        dist1 = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
        dist2 = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
        c1, c2 = 1e-6, 1e-6
        w_sum = self.alpha.sum() + self.beta.sum()
        alphas = torch.split(self.alpha / w_sum, self.chns, dim=1)
        betas = torch.split(self.beta / w_sum, self.chns, dim=1)

        for k in range(len(self.chns)):
            x_mean = feats0[k].mean([2, 3], keepdim=True)
            y_mean = feats1[k].mean([2, 3], keepdim=True)
            s1 = (2 * x_mean * y_mean + c1) / (x_mean**2 + y_mean**2 + c1)
            dist1 = dist1 + (alphas[k] * s1).sum(1, keepdim=True)

            x_var = ((feats0[k] - x_mean) ** 2).mean([2, 3], keepdim=True)
            y_var = ((feats1[k] - y_mean) ** 2).mean([2, 3], keepdim=True)
            xy_cov = (
                (feats0[k] * feats1[k]).mean([2, 3], keepdim=True)
                - x_mean * y_mean
            )
            s2 = (2 * xy_cov + c2) / (x_var + y_var + c2)
            dist2 = dist2 + (betas[k] * s2).sum(1, keepdim=True)

        score = 1 - (dist1 + dist2).squeeze()
        return score.mean() if batch_average else score


_dists_model: DISTS | None = None


def compute_dists(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    device: torch.device | None = None,
    weights_path: str | None = None,
    **kwargs,
) -> float:
    """Compute DISTS similarity score.

    Parameters
    ----------
    pred, target :
        Tensors of shape ``(N, C, H, W)`` in [0, 1].
    data_range :
        Value range (1.0 for [0,1] images).
    weights_path :
        Optional path to local ``.pt`` or ``.pth`` file with ``alpha`` and ``beta``.
        If omitted, loads from HuggingFace (chaofengc/IQA-PyTorch-Weights).

    Returns
    -------
    float :
        Average DISTS similarity. Higher is better (in [0, 1]).

    References
    ----------
    .. [1] Ding et al., "Image Quality Assessment: Unifying Structure and
           Texture Similarity", IEEE TPAMI 2022.
    """
    global _dists_model
    dev = device if device is not None else pred.device
    if weights_path is not None:
        model = DISTS(load_weights=True, weights_path=weights_path).to(dev).eval()
    else:
        if _dists_model is None:
            _dists_model = DISTS(load_weights=True).to(dev).eval()
        model = _dists_model.to(dev)
    if data_range != 1.0:
        pred = pred / data_range
        target = target / data_range
    with torch.no_grad():
        score = model(pred, target, batch_average=True)
    return score.item()
