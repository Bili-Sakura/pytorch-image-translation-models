# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""CUT model components built on native PyTorch modules.

This module provides the three networks required by the CUT (Contrastive
Unpaired Translation) method:

* :class:`CUTGenerator` – ResNet-based image-to-image generator that also
  supports intermediate feature extraction (``encode_only`` mode) needed by
  the PatchNCE contrastive loss.
* :class:`PatchGANDiscriminator` – N-layer PatchGAN discriminator (70×70
  receptive field by default with ``n_layers=3``).
* :class:`PatchSampleMLP` – lightweight MLP projection head that samples
  spatial patches from multi-layer encoder features and projects them into
  a shared embedding space for the contrastive loss.

Loss modules:

* :class:`GANLoss` – Configurable GAN objective (LSGAN / Vanilla / WGAN-GP).
* :class:`PatchNCELoss` – Per-layer contrastive loss following
  *Park et al. "Contrastive Learning for Unpaired Image-to-Image Translation"
  (ECCV 2020)*.
"""

from __future__ import annotations

import functools
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from diffusers import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _get_norm_layer(norm_type: str = "instance"):
    """Return a normalisation layer factory."""
    if norm_type == "batch":
        return functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    if norm_type == "instance":
        return functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    if norm_type == "none":
        return lambda x: nn.Identity()
    raise NotImplementedError(f"Norm layer [{norm_type}] is not supported")


def _init_weights(net: nn.Module, init_type: str = "xavier", init_gain: float = 0.02) -> None:
    """Initialise network weights (Xavier / Normal / Kaiming / Orthogonal)."""

    def _init_func(m: nn.Module) -> None:
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(f"Init method [{init_type}] not implemented")
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(_init_func)


# ---------------------------------------------------------------------------
# ResNet Generator (with encode_only support for NCE feature extraction)
# ---------------------------------------------------------------------------

class _ResnetBlock(nn.Module):
    """Single ResNet block with skip connection."""

    def __init__(self, dim: int, norm_layer, use_dropout: bool, use_bias: bool) -> None:
        super().__init__()
        layers: list = []
        layers += [nn.ReflectionPad2d(1)]
        layers += [nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            layers += [nn.Dropout(0.5)]
        layers += [nn.ReflectionPad2d(1)]
        layers += [nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias), norm_layer(dim)]
        self.conv_block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv_block(x)


class CUTGenerator(ModelMixin, ConfigMixin):
    """ResNet-based generator following the CUT paper architecture.

    Inherits from :class:`~diffusers.ModelMixin` and
    :class:`~diffusers.ConfigMixin` so that instances can be persisted and
    restored with ``save_pretrained`` / ``from_pretrained``.

    When called with ``layers`` and ``encode_only=True`` the forward pass
    returns a list of intermediate feature maps (used by PatchNCE).

    Parameters
    ----------
    input_nc : int
        Number of input channels.
    output_nc : int
        Number of output channels.
    ngf : int
        Base number of generator filters.
    n_blocks : int
        Number of ResNet blocks (9 for ``resnet_9blocks``, 6 for ``resnet_6blocks``).
    norm_type : str
        Normalisation type (``"instance"`` or ``"batch"``).
    use_dropout : bool
        Whether to use dropout in ResNet blocks.
    no_antialias : bool
        If ``True``, use strided convs instead of anti-aliased downsampling.
    no_antialias_up : bool
        If ``True``, use transposed convs instead of anti-aliased upsampling.
    init_type : str
        Weight initialisation method.
    init_gain : float
        Gain for weight initialisation.
    """

    @register_to_config
    def __init__(
        self,
        input_nc: int = 3,
        output_nc: int = 3,
        ngf: int = 64,
        n_blocks: int = 9,
        norm_type: str = "instance",
        use_dropout: bool = False,
        no_antialias: bool = True,
        no_antialias_up: bool = True,
        init_type: str = "xavier",
        init_gain: float = 0.02,
    ) -> None:
        super().__init__()
        norm_layer = _get_norm_layer(norm_type)
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model: list = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True),
        ]

        # Downsampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            if no_antialias:
                model += [
                    nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                    norm_layer(ngf * mult * 2),
                    nn.ReLU(True),
                ]
            else:
                model += [
                    nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                    norm_layer(ngf * mult * 2),
                    nn.ReLU(True),
                    nn.AvgPool2d(kernel_size=2, stride=2),
                ]

        # ResNet blocks
        mult = 2 ** n_downsampling
        for _ in range(n_blocks):
            model += [_ResnetBlock(ngf * mult, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        # Upsampling
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            if no_antialias_up:
                model += [
                    nn.ConvTranspose2d(
                        ngf * mult, int(ngf * mult / 2),
                        kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias,
                    ),
                    norm_layer(int(ngf * mult / 2)),
                    nn.ReLU(True),
                ]
            else:
                model += [
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                    nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=1, bias=use_bias),
                    norm_layer(int(ngf * mult / 2)),
                    nn.ReLU(True),
                ]

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

        _init_weights(self, init_type, init_gain)

    def forward(
        self,
        x: torch.Tensor,
        layers: Optional[List[int]] = None,
        encode_only: bool = False,
    ) -> torch.Tensor | List[torch.Tensor] | Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass.

        Parameters
        ----------
        x : Tensor (B, C, H, W)
            Input image in [-1, 1].
        layers : list of int, optional
            Layer indices from which to extract intermediate features.
        encode_only : bool
            If ``True`` and *layers* is given, return only the intermediate
            features (stop early). Used by PatchNCE.

        Returns
        -------
        Tensor or list of Tensors
            Generated image, or intermediate features.
        """
        if layers is not None and len(layers) > 0:
            feat = x
            feats: list = []
            for layer_id, layer in enumerate(self.model):
                feat = layer(feat)
                if layer_id in layers:
                    feats.append(feat)
                if layer_id == layers[-1] and encode_only:
                    return feats
            return feat, feats
        return self.model(x)


# ---------------------------------------------------------------------------
# PatchGAN Discriminator
# ---------------------------------------------------------------------------

class PatchGANDiscriminator(nn.Module):
    """N-layer PatchGAN discriminator (defaults to 70×70 receptive field).

    Parameters
    ----------
    input_nc : int
        Number of input channels.
    ndf : int
        Base number of discriminator filters.
    n_layers : int
        Number of convolutional layers.
    norm_type : str
        Normalisation type.
    no_antialias : bool
        If ``True``, use strided convs instead of anti-aliased downsampling.
    init_type : str
        Weight initialisation method.
    init_gain : float
        Gain for weight initialisation.
    """

    def __init__(
        self,
        input_nc: int = 3,
        ndf: int = 64,
        n_layers: int = 3,
        norm_type: str = "instance",
        no_antialias: bool = True,
        init_type: str = "xavier",
        init_gain: float = 0.02,
    ) -> None:
        super().__init__()
        norm_layer = _get_norm_layer(norm_type)
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        if no_antialias:
            sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        else:
            sequence = [
                nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=1, padding=padw),
                nn.LeakyReLU(0.2, True),
                nn.AvgPool2d(kernel_size=2, stride=2),
            ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            if no_antialias:
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True),
                ]
            else:
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True),
                    nn.AvgPool2d(kernel_size=2, stride=2),
                ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        self.model = nn.Sequential(*sequence)

        _init_weights(self, init_type, init_gain)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ---------------------------------------------------------------------------
# PatchSampleMLP (feature projection for contrastive loss)
# ---------------------------------------------------------------------------

class _L2Normalize(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).sum(1, keepdim=True).pow(0.5)
        return x.div(norm + 1e-7)


class PatchSampleMLP(nn.Module):
    """MLP projection head that samples spatial patches from encoder features.

    At first call, the MLP layers are lazily created based on the input
    feature dimensions (data-dependent initialisation following the original
    CUT implementation).

    Parameters
    ----------
    use_mlp : bool
        If ``True``, project features through a 2-layer MLP.
    nc : int
        Output dimension of the MLP projection.
    init_type : str
        Weight initialisation method.
    init_gain : float
        Gain for weight initialisation.
    """

    def __init__(
        self,
        use_mlp: bool = True,
        nc: int = 256,
        init_type: str = "xavier",
        init_gain: float = 0.02,
    ) -> None:
        super().__init__()
        self.l2norm = _L2Normalize()
        self.use_mlp = use_mlp
        self.nc = nc
        self.mlp_init = False
        self.init_type = init_type
        self.init_gain = init_gain

    def create_mlp(self, feats: List[torch.Tensor]) -> None:
        """Lazily create MLP layers matching the feature channel dims."""
        for mlp_id, feat in enumerate(feats):
            input_nc = feat.shape[1]
            mlp = nn.Sequential(nn.Linear(input_nc, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc))
            mlp = mlp.to(feat.device)
            setattr(self, f"mlp_{mlp_id}", mlp)
        _init_weights(self, self.init_type, self.init_gain)
        self.mlp_init = True

    def forward(
        self,
        feats: List[torch.Tensor],
        num_patches: int = 256,
        patch_ids: Optional[List] = None,
    ) -> Tuple[List[torch.Tensor], List]:
        return_ids: list = []
        return_feats: list = []
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)
        for feat_id, feat in enumerate(feats):
            B, H, W = feat.shape[0], feat.shape[2], feat.shape[3]
            feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)
            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[feat_id]
                else:
                    patch_id = np.random.permutation(feat_reshape.shape[1])
                    patch_id = patch_id[: int(min(num_patches, patch_id.shape[0]))]
                patch_id = torch.as_tensor(patch_id, dtype=torch.long, device=feat.device)
                x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)
            else:
                x_sample = feat_reshape
                patch_id = []
            if self.use_mlp:
                mlp = getattr(self, f"mlp_{feat_id}")
                x_sample = mlp(x_sample)
            return_ids.append(patch_id)
            x_sample = self.l2norm(x_sample)
            if num_patches == 0:
                x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])
            return_feats.append(x_sample)
        return return_feats, return_ids


# ---------------------------------------------------------------------------
# GAN Loss
# ---------------------------------------------------------------------------

class GANLoss(nn.Module):
    """GAN objective supporting LSGAN, Vanilla BCE, and WGAN-GP modes.

    Parameters
    ----------
    gan_mode : str
        ``"lsgan"`` | ``"vanilla"`` | ``"wgangp"``
    target_real_label : float
        Label for real images.
    target_fake_label : float
        Label for fake images.
    """

    def __init__(self, gan_mode: str = "lsgan", target_real_label: float = 1.0, target_fake_label: float = 0.0) -> None:
        super().__init__()
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        elif gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ("wgangp", "nonsaturating"):
            self.loss = None
        else:
            raise NotImplementedError(f"GAN mode [{gan_mode}] not implemented")

    def _get_target_tensor(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        target = self.real_label if target_is_real else self.fake_label
        return target.expand_as(prediction)

    def forward(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        if self.gan_mode in ("lsgan", "vanilla"):
            target_tensor = self._get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == "wgangp":
            loss = -prediction.mean() if target_is_real else prediction.mean()
        elif self.gan_mode == "nonsaturating":
            bs = prediction.size(0)
            if target_is_real:
                loss = F.softplus(-prediction).view(bs, -1).mean(dim=1)
            else:
                loss = F.softplus(prediction).view(bs, -1).mean(dim=1)
        return loss


# ---------------------------------------------------------------------------
# PatchNCE Loss
# ---------------------------------------------------------------------------

class PatchNCELoss(nn.Module):
    """Contrastive loss on sampled feature patches (InfoNCE formulation).

    Parameters
    ----------
    nce_T : float
        Temperature for the contrastive loss softmax.
    batch_size : int
        Batch size used for reshaping negatives.
    nce_includes_all_negatives_from_minibatch : bool
        If ``True``, include negatives from the entire mini-batch.
    """

    def __init__(
        self,
        nce_T: float = 0.07,
        batch_size: int = 1,
        nce_includes_all_negatives_from_minibatch: bool = False,
    ) -> None:
        super().__init__()
        self.nce_T = nce_T
        self.batch_size = batch_size
        self.nce_includes_all_negatives_from_minibatch = nce_includes_all_negatives_from_minibatch
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")

    def forward(self, feat_q: torch.Tensor, feat_k: torch.Tensor) -> torch.Tensor:
        num_patches = feat_q.shape[0]
        feat_k = feat_k.detach()

        # Positive logit
        l_pos = torch.bmm(feat_q.view(num_patches, 1, -1), feat_k.view(num_patches, -1, 1))
        l_pos = l_pos.view(num_patches, 1)

        # Negative logits
        if self.nce_includes_all_negatives_from_minibatch:
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.batch_size

        dim = feat_q.shape[1]
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        # Mask out self-similarity on diagonal
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=torch.bool)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.nce_T
        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long, device=feat_q.device))
        return loss


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def create_generator(
    input_nc: int = 3,
    output_nc: int = 3,
    ngf: int = 64,
    netG: str = "resnet_9blocks",
    norm_type: str = "instance",
    use_dropout: bool = False,
    no_antialias: bool = True,
    no_antialias_up: bool = True,
    init_type: str = "xavier",
    init_gain: float = 0.02,
    **kwargs,
) -> CUTGenerator:
    """Factory for the CUT generator."""
    n_blocks = {"resnet_9blocks": 9, "resnet_6blocks": 6, "resnet_4blocks": 4}.get(netG, 9)
    return CUTGenerator(
        input_nc=input_nc,
        output_nc=output_nc,
        ngf=ngf,
        n_blocks=n_blocks,
        norm_type=norm_type,
        use_dropout=use_dropout,
        no_antialias=no_antialias,
        no_antialias_up=no_antialias_up,
        init_type=init_type,
        init_gain=init_gain,
    )


def create_discriminator(
    input_nc: int = 3,
    ndf: int = 64,
    netD: str = "basic",
    n_layers_D: int = 3,
    norm_type: str = "instance",
    no_antialias: bool = True,
    init_type: str = "xavier",
    init_gain: float = 0.02,
    **kwargs,
) -> PatchGANDiscriminator:
    """Factory for the CUT discriminator."""
    n_layers = 3 if netD == "basic" else n_layers_D
    return PatchGANDiscriminator(
        input_nc=input_nc,
        ndf=ndf,
        n_layers=n_layers,
        norm_type=norm_type,
        no_antialias=no_antialias,
        init_type=init_type,
        init_gain=init_gain,
    )


def create_patch_sample_mlp(
    use_mlp: bool = True,
    nc: int = 256,
    init_type: str = "xavier",
    init_gain: float = 0.02,
    **kwargs,
) -> PatchSampleMLP:
    """Factory for the PatchSampleMLP feature projection network."""
    return PatchSampleMLP(
        use_mlp=use_mlp,
        nc=nc,
        init_type=init_type,
        init_gain=init_gain,
    )
