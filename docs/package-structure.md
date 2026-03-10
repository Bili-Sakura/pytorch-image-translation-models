# Package Structure

```text
src/                                 # ← Core library (single source of truth)
├── __init__.py                      # Public API
├── models/
│   ├── generators/
│   │   ├── unet.py                  # UNetSkipConnection, UNetGenerator
│   │   └── resnet.py                # ResidualBlock, ResNetGenerator
│   ├── pix2pixhd/
│   │   ├── blocks.py                # _ResnetBlock
│   │   └── generator.py             # Pix2PixHDGenerator, Pix2PixHDGlobalGenerator
│   ├── discriminators/
│   │   └── patchgan.py              # PatchGANDiscriminator
│   ├── unet/
│   │   ├── i2sb_unet.py            # I2SBUNet (native ADM-style backbone)
│   │   ├── unet_2d.py              # create_model factory
│   │   └── diffusers_wrappers.py   # DDBMUNet, DDIBUNet, … (diffusers UNet2DModel wrappers)
│   ├── dit/
│   │   └── sit.py                  # SiTBackbone (Diffusion Transformer)
│   └── stegogan/
│       ├── generators.py           # ResnetMaskV1Generator, ResnetMaskV3Generator
│       └── networks.py             # NetMatchability, mask_generate, ResnetBlock
│   └── unsb/
│       └── unsb_model.py           # UNSBGenerator, UNSBDiscriminator, UNSBEnergyNet
│   └── local_diffusion/
│       └── local_diffusion_model.py # LocalDiffusionUNet, ConditionEncoder
│   └── stargan/
│       ├── blocks.py               # StarGANResidualBlock
│       ├── generator.py            # StarGANGenerator
│       └── discriminator.py        # StarGANDiscriminator
├── schedulers/                      # One scheduler per method
│   ├── i2sb.py                     # I2SBScheduler
│   ├── ddbm.py                     # DDBMScheduler
│   ├── bbdm.py                     # BBDMScheduler (one-way)
│   ├── bibbdm.py                   # BiBBDMScheduler
│   ├── ddib.py                     # DDIBScheduler
│   ├── bdbm.py                     # BDBMScheduler
│   ├── dbim.py                     # DBIMScheduler
│   ├── cdtsde.py                   # CDTSDEScheduler
│   └── lbm.py                      # LBMScheduler
│   └── unsb.py                     # UNSBScheduler
│   └── local_diffusion.py          # LocalDiffusionScheduler (DDPM/DDIM)
├── pipelines/                       # One pipeline per method
│   ├── i2sb.py                     # I2SBPipeline
│   ├── ddbm.py                     # DDBMPipeline
│   ├── bbdm.py                     # BBDMPipeline (one-way)
│   ├── bibbdm.py                   # BiBBDMPipeline
│   ├── ddib.py                     # DDIBPipeline
│   ├── bdbm.py                     # BDBMPipeline
│   ├── dbim.py                     # DBIMPipeline
│   ├── cdtsde.py                   # CDTSDEPipeline
│   └── lbm.py                      # LBMPipeline
│   └── unsb.py                     # UNSBPipeline
│   └── local_diffusion.py          # LocalDiffusionPipeline
│   └── pix2pixhd.py                # Pix2PixHDPipeline, load_pix2pixhd_pipeline
│   └── stargan.py                  # StarGANPipeline, load_stargan_pipeline
├── data/
│   ├── datasets.py                 # PairedImageDataset, UnpairedImageDataset
│   └── transforms.py               # get_transforms, default_transforms
├── losses/
│   ├── adversarial.py              # GANLoss
│   └── perceptual.py               # PerceptualLoss
├── inference/
│   └── predictor.py                # ImageTranslator
└── metrics/
    └── image_quality.py            # PSNR, SSIM, LPIPS, FID
examples/                            # ← Training/inference scripts (import from src/)
├── community/                       # Community-contributed pipelines (single-file)
│   ├── parallel_gan/               # Parallel-GAN (Wang et al., TGRS 2022)
│   ├── e3diff/                     # E3Diff (Qin et al., IEEE GRSL 2024)
│   └── sar2optical/                # SAR2Optical Pix2Pix cGAN (Isola et al., CVPR 2017)
├── i2sb/                            # I2SB paper-oriented training code
│   ├── config.py                   # TaskConfig, sar2eo_config, etc.
│   └── trainer.py                  # I2SBTrainer
├── pix2pix/                         # Pix2Pix paired translation (Pix2PixTrainer)
├── stegogan/                        # StegoGAN unpaired translation (StegoGANTrainer)
├── stargan/                         # StarGAN multi-domain translation (StarGANTrainer)
├── cut/                             # CUT contrastive unpaired translation (CUTTrainer)
└── inference/
    └── run_inference.py            # Unified inference script for all methods
```
