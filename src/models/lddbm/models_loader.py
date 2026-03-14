# Copyright (c) 2025 Copyright holder of the paper "Towards General Modality Translation with Contrastive and Predictive Latent Diffusion Bridge" submitted to NeurIPS 2025 for review.
# All rights reserved.

"""Model loaders for MDT - Super Resolution only."""

from src.models.lddbm.bridge_model import BridgeModel
from src.models.lddbm.encoder_decoder import (
    AutoencoderKLInnerExtdDecoder,
    AutoencoderKLInnerExtdEncoder,
)
from src.models.lddbm.karras_diffusion import KarrasDenoiser
from src.models.lddbm.names import BridgeModelsTyps, Decoders, Encoders
from src.models.lddbm.resample import (
    LogNormalSampler,
    LossSecondMomentResampler,
    RealUniformSampler,
    UniformSampler,
    create_named_schedule_sampler,
)
from src.models.lddbm.transformer import AutoRegressiveTransformer

# Inlined VAE config (from config.yaml)
DDCONFIG = {
    "double_z": True,
    "z_channels": 64,
    "resolution": 256,
    "in_channels": 3,
    "out_ch": 3,
    "ch": 128,
    "ch_mult": (1, 1, 2, 2, 4, 4),
    "num_res_blocks": 2,
    "attn_resolutions": [16, 8],
    "dropout": 0.0,
    "resamp_with_conv": True,
    "use_linear_attn": False,
    "attn_type": "vanilla",
}
EMBED_DIM = 64


def create_encoder(encoder_type: str, model_args):
    if encoder_type == Encoders.KlVaePreTrainedEncoder16.value:
        return AutoencoderKLInnerExtdEncoder(
            DDCONFIG,
            EMBED_DIM,
            model_type=Encoders.KlVaePreTrainedEncoder16.value,
        )

    elif encoder_type == Encoders.KlVaePreTrainedEncoder128.value:
        return AutoencoderKLInnerExtdEncoder(
            DDCONFIG,
            EMBED_DIM,
            model_type=Encoders.KlVaePreTrainedEncoder128.value,
        )

    else:
        raise NotImplementedError(f"Encoder type {encoder_type} not implemented")


def create_bridge(model_args):
    if model_args.denoiser_type == BridgeModelsTyps.BridgeTransformer.value:
        denoiser = AutoRegressiveTransformer(in_channels=model_args.in_channels)
    else:
        raise NotImplementedError(
            f"Bridge Model {model_args.denoiser_type} not implemented"
        )

    diffusion = KarrasDenoiser(
        sigma_data=model_args.sigma_data,
        sigma_max=model_args.sigma_max,
        sigma_min=model_args.sigma_min,
        beta_d=model_args.beta_d,
        beta_min=model_args.beta_min,
        cov_xy=model_args.cov_xy,
        weight_schedule=model_args.weight_schedule,
        pred_mode=model_args.pred_mode,
    )

    schedule_sampler = create_named_schedule_sampler(
        model_args.schedule_sampler, diffusion
    )

    return BridgeModel(denoiser, diffusion, schedule_sampler)


def create_decoder(decoder_type: str, model_args):
    if decoder_type == Decoders.KlVaePreTrainedDecoder128.value:
        return AutoencoderKLInnerExtdDecoder(DDCONFIG, EMBED_DIM)

    elif decoder_type == Decoders.NoDecoder.value:
        return None

    else:
        raise NotImplementedError(f"Decoder type {decoder_type} not implemented")
