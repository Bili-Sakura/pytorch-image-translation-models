# Copyright (c) 2026 EarthBridge Team.
# Credits: ECSI (https://github.com/szhan311/ECSI) - UNet and diffusion model factory.

"""ECSI model factory and diffusion creation."""

from __future__ import annotations

from src.models.ecsi._utils.edm_unet import EDM
from src.models.ecsi._utils.unet import UNetModel
from src.models.ecsi.diffusion import KarrasDenoiser

NUM_CLASSES = 1000


def create_model(
    image_size: int,
    in_channels: int,
    num_channels: int,
    num_res_blocks: int,
    unet_type: str = "adm",
    channel_mult: str = "",
    learn_sigma: bool = False,
    class_cond: bool = False,
    use_checkpoint: bool = False,
    attention_resolutions: str = "16",
    num_heads: int = 1,
    num_head_channels: int = -1,
    num_heads_upsample: int = -1,
    use_scale_shift_norm: bool = False,
    dropout: float = 0.0,
    resblock_updown: bool = False,
    use_fp16: bool = False,
    use_new_attention_order: bool = False,
    attention_type: str = "default",
    condition_mode: str | None = None,
) -> UNetModel | EDM:
    """Create ECSI UNet (ADM or EDM)."""
    if channel_mult == "":
        channel_mult_map = {
            512: (0.5, 1, 1, 2, 2, 4, 4),
            256: (1, 1, 2, 2, 4, 4),
            128: (1, 1, 2, 3, 4),
            64: (1, 2, 3, 4),
            32: (1, 2, 3, 4),
        }
        channel_mult = channel_mult_map.get(
            image_size, (1, 2, 3, 4)
        )
    else:
        channel_mult = tuple(int(m) for m in channel_mult.split(","))

    attention_ds = [image_size // int(r) for r in attention_resolutions.split(",")]

    if unet_type == "adm":
        return UNetModel(
            image_size=image_size,
            in_channels=in_channels,
            model_channels=num_channels,
            out_channels=in_channels * 2 if learn_sigma else in_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=dropout,
            channel_mult=channel_mult,
            num_classes=NUM_CLASSES if class_cond else None,
            use_checkpoint=use_checkpoint,
            use_fp16=use_fp16,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_new_attention_order=use_new_attention_order,
            attention_type=attention_type,
            condition_mode=condition_mode,
        )
    if unet_type == "edm":
        return EDM(
            img_resolution=image_size,
            in_channels=in_channels,
            out_channels=in_channels * 2 if learn_sigma else in_channels,
            model_channels=num_channels,
            channel_mult=channel_mult,
            num_blocks=4,
            attn_resolutions=[16],
            dropout=dropout,
            channel_mult_noise=2,
            embedding_type="fourier",
            encoder_type="residual",
            decoder_type="standard",
            resample_filter=[1, 3, 3, 1],
            condition_mode=condition_mode,
        )
    raise ValueError(f"Unsupported unet type: {unet_type}")


def create_model_and_diffusion(
    image_size: int,
    in_channels: int,
    class_cond: bool,
    learn_sigma: bool,
    num_channels: int,
    num_res_blocks: int,
    channel_mult: str | tuple,
    num_heads: int,
    num_head_channels: int,
    num_heads_upsample: int,
    attention_resolutions: str,
    dropout: float,
    use_checkpoint: bool,
    use_scale_shift_norm: bool,
    resblock_updown: bool,
    use_fp16: bool,
    use_new_attention_order: bool,
    attention_type: str,
    condition_mode: str | None,
    pred_mode: str,
    weight_schedule: str,
    sigma_data: float = 0.5,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    cov_xy: float = 0.0,
    unet_type: str = "adm",
) -> tuple:
    """Create ECSI model and KarrasDenoiser diffusion."""
    cm = channel_mult if isinstance(channel_mult, str) else ",".join(map(str, channel_mult))
    model = create_model(
        image_size=image_size,
        in_channels=in_channels,
        num_channels=num_channels,
        num_res_blocks=num_res_blocks,
        unet_type=unet_type,
        channel_mult=cm,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
        use_new_attention_order=use_new_attention_order,
        attention_type=attention_type,
        condition_mode=condition_mode,
    )
    diffusion = KarrasDenoiser(
        sigma_data=sigma_data,
        sigma_max=sigma_max,
        sigma_min=sigma_min,
        cov_xy=cov_xy,
        image_size=image_size,
        weight_schedule=weight_schedule,
        pred_mode=pred_mode,
    )
    return model, diffusion
