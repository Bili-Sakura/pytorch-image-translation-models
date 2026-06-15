# Credits: CycleGAN (Zhu et al., ICCV 2017) and pix2pix (Isola et al., CVPR 2017).
# Adapted from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

"""Network factories for CycleGAN and pix2pix (junyanz/pytorch-CycleGAN-and-pix2pix)."""

from __future__ import annotations

import functools
from typing import Literal

import torch
import torch.nn as nn
from diffusers import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from torch.nn import init

from src.models.discriminators.patchgan import PatchGANDiscriminator
from src.models.generators.resnet import ResNetGenerator, ResidualBlock
from src.models.generators.unet import UNetGenerator, UNetSkipConnection
from src.losses.adversarial import GANLoss

NetGName = Literal["resnet_9blocks", "resnet_6blocks", "unet_128", "unet_256"]
NetDName = Literal["basic", "n_layers", "pixel"]
NormName = Literal["batch", "instance", "none"]


def get_norm_layer(norm_type: NormName = "instance"):
    """Return a normalization layer factory matching the upstream repo."""
    if norm_type == "batch":
        return functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    if norm_type == "instance":
        return functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    if norm_type == "none":
        return lambda _: nn.Identity()
    raise NotImplementedError(f"Normalization layer [{norm_type}] is not supported")


def init_weights(net: nn.Module, init_type: str = "normal", init_gain: float = 0.02) -> None:
    """Initialize network weights (normal | xavier | kaiming | orthogonal)."""

    def _init_func(module: nn.Module) -> None:
        classname = module.__class__.__name__
        if hasattr(module, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                init.normal_(module.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(module.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(module.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(module.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(f"Init method [{init_type}] not implemented")
            if hasattr(module, "bias") and module.bias is not None:
                init.constant_(module.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(module.weight.data, 1.0, init_gain)
            init.constant_(module.bias.data, 0.0)

    net.apply(_init_func)


class CycleGANGenerator(ModelMixin, ConfigMixin):
    """ResNet generator for CycleGAN with diffusers checkpoint support."""

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_filters: int = 64,
        n_residual_blocks: int = 9,
        norm_type: NormName = "instance",
        use_dropout: bool = False,
        init_type: str = "normal",
        init_gain: float = 0.02,
    ) -> None:
        super().__init__()
        norm_layer = get_norm_layer(norm_type)
        backbone = ResNetGenerator(
            in_channels=in_channels,
            out_channels=out_channels,
            base_filters=base_filters,
            n_residual_blocks=n_residual_blocks,
            norm_layer=norm_layer,
        )
        self.model = backbone.model
        init_weights(self, init_type, init_gain)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Pix2PixGenerator(ModelMixin, ConfigMixin):
    """U-Net generator for pix2pix with diffusers checkpoint support."""

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_downs: int = 8,
        base_filters: int = 64,
        use_dropout: bool = True,
        norm_type: NormName = "batch",
        init_type: str = "normal",
        init_gain: float = 0.02,
    ) -> None:
        super().__init__()
        norm_layer = get_norm_layer(norm_type)
        backbone = UNetGenerator(
            in_channels=in_channels,
            out_channels=out_channels,
            num_downs=num_downs,
            base_filters=base_filters,
            use_dropout=use_dropout,
            norm_layer=norm_layer,
        )
        self.model = backbone.model
        init_weights(self, init_type, init_gain)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class PixelDiscriminator(nn.Module):
    """1×1 PixelGAN discriminator from the upstream repo."""

    def __init__(
        self,
        in_channels: int = 6,
        base_filters: int = 64,
        norm_type: NormName = "batch",
    ) -> None:
        super().__init__()
        norm_layer = get_norm_layer(norm_type)
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(base_filters, base_filters * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(base_filters * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(base_filters * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias),
        )

    def forward(self, x):
        return self.net(x)


def _n_blocks_from_netg(netG: NetGName) -> int:
    if netG == "resnet_9blocks":
        return 9
    if netG == "resnet_6blocks":
        return 6
    raise ValueError(f"netG [{netG}] is not a ResNet generator")


def _num_downs_from_netg(netG: NetGName) -> int:
    if netG == "unet_128":
        return 7
    if netG == "unet_256":
        return 8
    raise ValueError(f"netG [{netG}] is not a U-Net generator")


def create_generator(
    input_nc: int = 3,
    output_nc: int = 3,
    ngf: int = 64,
    netG: NetGName = "resnet_9blocks",
    norm: NormName = "instance",
    use_dropout: bool = False,
    init_type: str = "normal",
    init_gain: float = 0.02,
) -> nn.Module:
    """Create a generator matching junyanz/pytorch-CycleGAN-and-pix2pix."""
    if netG in ("resnet_9blocks", "resnet_6blocks"):
        return CycleGANGenerator(
            in_channels=input_nc,
            out_channels=output_nc,
            base_filters=ngf,
            n_residual_blocks=_n_blocks_from_netg(netG),
            norm_type=norm,
            use_dropout=use_dropout,
            init_type=init_type,
            init_gain=init_gain,
        )
    if netG in ("unet_128", "unet_256"):
        return Pix2PixGenerator(
            in_channels=input_nc,
            out_channels=output_nc,
            num_downs=_num_downs_from_netg(netG),
            base_filters=ngf,
            use_dropout=use_dropout,
            norm_type=norm,
            init_type=init_type,
            init_gain=init_gain,
        )
    raise NotImplementedError(f"Generator model name [{netG}] is not recognized")


def create_discriminator(
    input_nc: int = 3,
    ndf: int = 64,
    netD: NetDName = "basic",
    n_layers_D: int = 3,
    norm: NormName = "batch",
    init_type: str = "normal",
    init_gain: float = 0.02,
) -> nn.Module:
    """Create a discriminator matching junyanz/pytorch-CycleGAN-and-pix2pix."""
    norm_layer = get_norm_layer(norm)
    if netD == "basic":
        n_layers = 3
    elif netD == "n_layers":
        n_layers = n_layers_D
    elif netD == "pixel":
        disc = PixelDiscriminator(input_nc=input_nc, base_filters=ndf, norm_type=norm)
        init_weights(disc, init_type, init_gain)
        return disc
    else:
        raise NotImplementedError(f"Discriminator model name [{netD}] is not recognized")

    disc = PatchGANDiscriminator(
        in_channels=input_nc,
        base_filters=ndf,
        n_layers=n_layers,
        norm_layer=norm_layer,
    )
    init_weights(disc, init_type, init_gain)
    return disc


def patch_instance_norm_state_dict(state_dict: dict) -> dict:
    """Remove legacy InstanceNorm keys that break strict loading."""
    cleaned = {}
    for key, value in state_dict.items():
        if "running_mean" in key or "running_var" in key or "num_batches_tracked" in key:
            continue
        cleaned[key] = value
    return cleaned


def load_upstream_generator_state(
    net: nn.Module,
    checkpoint_path: str,
    *,
    strict: bool = True,
) -> None:
    """Load a raw ``.pth`` checkpoint from junyanz/pytorch-CycleGAN-and-pix2pix."""
    import torch
    from pathlib import Path

    state = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = patch_instance_norm_state_dict(state)
    net.load_state_dict(state, strict=strict)


__all__ = [
    "CycleGANGenerator",
    "Pix2PixGenerator",
    "PixelDiscriminator",
    "GANLoss",
    "get_norm_layer",
    "init_weights",
    "create_generator",
    "create_discriminator",
    "patch_instance_norm_state_dict",
    "load_upstream_generator_state",
    "ResidualBlock",
    "UNetSkipConnection",
]
