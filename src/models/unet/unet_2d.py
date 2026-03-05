# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Factory function for creating I2SB UNet models."""

from __future__ import annotations

from src.models.unet.i2sb_unet import I2SBUNet


def create_model(
    image_size: int,
    in_channels: int,
    num_channels: int,
    num_res_blocks: int,
    attention_resolutions: str = "",
    condition_mode: str | None = None,
    channel_mult: tuple[int, ...] | None = None,
    **kwargs,
) -> I2SBUNet:
    """Create an I2SB UNet model.

    Parameters
    ----------
    image_size : int
        Spatial resolution of the input.
    in_channels : int
        Number of input image channels.
    num_channels : int
        Base channel width.
    num_res_blocks : int
        Number of residual blocks per resolution level.
    attention_resolutions : str
        Comma-separated spatial resolutions for self-attention (e.g.
        ``"16,8"``).  Pass ``""`` to disable attention entirely.
    condition_mode : str | None
        ``"concat"`` for conditional generation, ``None`` for unconditional.
    channel_mult : tuple[int, ...] | None
        Channel multipliers per level.  When *None*, a default is chosen
        based on *image_size*.
    """
    if channel_mult is None:
        if image_size >= 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size >= 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size >= 64:
            channel_mult = (1, 2, 3, 4)
        else:
            channel_mult = (1, 2, 4)

    # Parse attention resolutions string
    attn_res: set[int] = set()
    if isinstance(attention_resolutions, str) and attention_resolutions.strip():
        attn_res = {int(x) for x in attention_resolutions.split(",")}

    return I2SBUNet(
        image_size=image_size,
        in_channels=in_channels,
        model_channels=num_channels,
        out_channels=in_channels,
        num_res_blocks=num_res_blocks,
        attention_resolutions=attn_res,
        channel_mult=channel_mult,
        condition_mode=condition_mode,
    )
