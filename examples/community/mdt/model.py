# Copyright (c) 2026 EarthBridge Team.
# Credits: Bosch Research LDDBM (Multimodal-Distribution-Translation-MDT).

"""MDT / LDDBM – Models live in the upstream lddbm package.

This community pipeline provides a thin wrapper around
https://github.com/boschresearch/Multimodal-Distribution-Translation-MDT (LDDBM).

The architecture consists of:
* Encoder_x / Encoder_y: KL-VAE encoders (or task-specific encoders).
* Decoder_x / Decoder_y: KL-VAE decoders.
* BridgeModel: Latent diffusion bridge (transformer denoiser + Karras schedule).

No local model definitions – see pipeline.py for the inference wrapper.
"""
