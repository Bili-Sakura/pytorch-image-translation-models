# Features

Examples and pipeline snippets in docs default to `device="cuda"` unless explicitly noted. Pipelines support `pipeline.to("cuda")` / `pipeline.to("cpu")`.

## Models

- **GAN generators** — `UNetGenerator` (encoder-decoder with skip connections), `ResNetGenerator` (residual blocks)
- **GAN discriminators** — `PatchGANDiscriminator` (Markovian patch-level classifier)
- **StegoGAN** — `ResnetMaskV1Generator`, `ResnetMaskV3Generator`, `NetMatchability` (steganographic masking for non-bijective translation, CVPR 2024)
- **pix2pixHD** — `Pix2PixHDGenerator`, `Pix2PixHDGlobalGenerator` (high-resolution conditional GAN baseline, CVPR 2018)
- **StarGAN** — `StarGANGenerator`, `StarGANDiscriminator` (single model for multi-domain translation, CVPR 2018)
- **Diffusion bridge** — `I2SBUNet` (ADM-style U-Net in `adm.py` for Image-to-Image Schrödinger Bridge)
- **UNSB** — `UNSBGenerator`, `UNSBDiscriminator`, `UNSBEnergyNet` (time-conditional networks for Unpaired Neural Schrödinger Bridge, ICLR 2024)
- **Local Diffusion** — `LocalDiffusionUNet`, `ConditionEncoder` (conditional denoising U-Net with branch-and-fuse for hallucination suppression, ECCV 2024 Oral)
- **DiT backbone** — `SiTBackbone` (Scalable Interpolant Transformer for diffusion bridges)
- **FCDM** — `FCDM`, `FCDMImageCond` (ConvNeXt-based diffusion backbone for class-conditional and image-conditioned generation, CVPR 2026)

## Schedulers

| Scheduler | Description |
|---|---|
| **I2SBScheduler** | Symmetric beta schedule with forward/reverse bridge kernels for I2SB |
| **DDBMScheduler** | Karras sigma schedule with Heun/Euler sampling for DDBM (VP/VE modes) |
| **BBDMScheduler** | One-way Brownian Bridge reverse sampler (source -> target) for BBDM |
| **BiBBDMScheduler** | Brownian Bridge noise schedule with bidirectional sampling for BiBBDM |
| **DDIBScheduler** | Gaussian diffusion with DDIM forward/reverse steps for DDIB |
| **BDBMScheduler** | Bidirectional Brownian Bridge schedule for BDBM |
| **DBIMScheduler** | Faster bridge sampler with eta-controlled stochasticity for DBIM |
| **CDTSDEScheduler** | Dynamic domain-shift eta schedule for CDTSDE |
| **LBMScheduler** | Flow-matching bridge for single/few-step LBM translation |
| **SiD2Scheduler** | Exact SiD2 continuous scheduler (t ∈ [0,1]) with cosine-interpolated log-SNR; drop-in for DDPMScheduler (CVPR 2025) |
| **UNSBScheduler** | Non-uniform harmonic time schedule with stochastic bridge dynamics for UNSB |
| **LocalDiffusionScheduler** | Gaussian diffusion (DDPM/DDIM) with sigmoid/cosine/linear beta schedules for Local Diffusion |
| **FCDMScheduler** | DDPM/DDIM with linear schedule and pred_noise for FCDM latent diffusion |

## Pipelines

| Pipeline | Description |
|---|---|
| **I2SBPipeline** | End-to-end inference for I2SB models |
| **DDBMPipeline** | DDBM bridge diffusion with Heun's method |
| **BBDMPipeline** | One-way BBDM reverse Brownian Bridge translation |
| **BiBBDMPipeline** | Bidirectional Brownian Bridge translation (b2a / a2b) |
| **DDIBPipeline** | Dual-model DDIM encode/decode translation |
| **BDBMPipeline** | Bidirectional diffusion bridge with context conditioning |
| **DBIMPipeline** | Fast DBIM bridge sampling with bridge preconditioning |
| **CDTSDEPipeline** | CDTSDE with dynamic domain-shift scheduling |
| **LBMPipeline** | LBM flow-matching for single/few-step image translation |
| **UNSBPipeline** | Multi-step Schrödinger Bridge with adversarial + contrastive losses |
| **LocalDiffusionPipeline** | Branch-and-fuse diffusion for hallucination-aware image translation |
| **Pix2PixHDPipeline** | Native pix2pixHD single-pass generator inference with checkpoint loader |
| **StarGANPipeline** | Native StarGAN single-pass multi-domain translation with label conditioning |
| **FCDMPipeline** | Class-conditional FCDM latent diffusion (DDPM/DDIM) with optional VAE decode and CFG |
| **FCDMImageCondPipeline** | Image-conditioned FCDM for translation (source latent → target) |

All pipelines support `"pt"`, `"pil"`, and `"np"` output types.

## Data

- `PairedImageDataset` / `UnpairedImageDataset` with configurable transform pipelines

## Losses

- `GANLoss` (vanilla / LSGAN / hinge), VGG-based `PerceptualLoss`

## Training

- `Pix2PixTrainer` — Paired GAN training with checkpoint save/load
- **HF Storage Buckets** — Sync checkpoints and TensorBoard logs to [Hugging Face Storage Buckets](storage-buckets.md) (CUT, pix2pix tutorials; requires `huggingface_hub` ≥ 1.5.0)
- `StegoGANTrainer` — StegoGAN unpaired training with steganographic masking and consistency losses
- `I2SBTrainer` — I2SB bridge model training (in `examples/i2sb/`)
- `StarGANTrainer` — WGAN-GP + domain classification + reconstruction training for multi-domain translation (in ``examples/stargan/``)

## Metrics

- **Paired** (reference-based): `PairedImageMetricEvaluator` — PSNR, SSIM, LPIPS, DISTS, SAMScore
- **Unpaired** (distribution-based): `UnpairedImageMetricEvaluator` — FID, KID, IS, SFD, CMMD, Precision/Recall
- One-stop usage and custom HuggingFace/local checkpoints: [src/metrics/README.md](../src/metrics/README.md)

## Community Pipelines

Self-contained, single-file modules contributed by the community (inspired by [diffusers community pipelines](https://github.com/huggingface/diffusers/tree/main/examples/community)):

| Pipeline | Paper | Description |
|----------|-------|-------------|
| [`diffuseit/`](../examples/community/diffuseit/) | [Kwon & Ye, ICLR 2023](https://arxiv.org/abs/2209.15264) | Diffusion-based image translation with disentangled style/content (text- and image-guided) |
| [`parallel_gan/`](../examples/community/parallel_gan/) | [Wang et al., TGRS 2022](https://ieeexplore.ieee.org/document/9864654) | SAR-to-Optical with hierarchical latent features |
| [`e3diff/`](../examples/community/e3diff/) | [Qin et al., IEEE GRSL 2024](https://ieeexplore.ieee.org/document/10767752) | Efficient End-to-End Diffusion for one-step SAR-to-Optical |
| [`sar2optical/`](../examples/community/sar2optical/) | [Isola et al., CVPR 2017](https://arxiv.org/abs/1611.07004) | Pix2Pix cGAN SAR-to-Optical translation adapted from yuuIind/SAR2Optical |
