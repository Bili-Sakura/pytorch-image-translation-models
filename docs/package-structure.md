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
│   │   ├── adm.py                  # ADM-style UNet (I2SBUNet, create_model, DDBM/DDIB/… wrappers)
│   │   ├── edm.py                  # EDM placeholder
│   │   ├── vdmpp.py                # VDM++ placeholder
│   │   ├── rin.py                  # RIN placeholder
│   │   ├── sid.py                  # SID placeholder
│   │   └── sid2.py                 # SiD2 placeholder
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
│   └── sid2.py                     # SiD2Scheduler
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
│   └── pix2pix.py                 # ImageTranslator (Pix2Pix single-pass)
├── data/
│   ├── datasets.py                 # PairedImageDataset, UnpairedImageDataset
│   └── transforms.py               # get_transforms, default_transforms
├── losses/
│   ├── adversarial.py              # GANLoss
│   └── perceptual.py               # PerceptualLoss
└── metrics/
    ├── paired/                     # Reference-based metrics
    │   ├── evaluator.py            # PairedImageMetricEvaluator
    │   ├── psnr.py                 # PSNR (Wang et al., IEEE TIP 2004)
    │   ├── ssim.py                 # SSIM (Wang et al., IEEE TIP 2004)
    │   ├── lpips_impl.py           # LPIPS (Zhang et al., CVPR 2018)
    │   ├── dists_impl.py           # DISTS (Ding et al., IEEE TPAMI 2022)
    │   ├── samscore_impl.py        # SAMScore (Li et al., IEEE TAI 2025)
    │   └── registry.py             # Metric registry
    └── unpaired/                   # Distribution-based metrics
        ├── evaluator.py            # UnpairedImageMetricEvaluator
        ├── registry.py             # Metric registry
        ├── fid.py                  # FID (Heusel et al., NeurIPS 2017)
        ├── kid.py                  # KID (Binkowski et al., ICLR 2018)
        ├── is_.py                  # IS (Salimans et al., NeurIPS 2016)
        ├── sfd.py                  # SFD (Kim et al., Sensors 2020)
        ├── sfid.py                 # sFID (Nash et al., ICML 2021)
        ├── precision_recall.py     # P&R (Kynkäänniemi et al., NeurIPS 2019)
        ├── cmmd.py                 # CMMD (Jayasumana et al., CVPR 2024)
        ├── fwd.py                  # FWD (ICLR 2025)
        └── ifid.py                 # iFID (arXiv 2026)
examples/                            # ← Training/inference scripts (import from src/)
├── community/                       # Community-contributed pipelines (single-file)
│   ├── diffuseit/                  # DiffuseIT (Kwon & Ye, ICLR 2023)
│   ├── parallel_gan/               # Parallel-GAN (Wang et al., TGRS 2022)
│   ├── e3diff/                     # E3Diff (Qin et al., IEEE GRSL 2024)
│   └── sar2optical/                # SAR2Optical Pix2Pix cGAN (Isola et al., CVPR 2017)
├── i2sb/                            # I2SB paper-oriented training code
│   ├── config.py                   # TaskConfig, sar2eo_config, etc.
│   └── trainer.py                  # I2SBTrainer
├── pix2pix/                         # Pix2Pix paired translation (Pix2PixTrainer)
├── stegogan/                        # StegoGAN unpaired translation (StegoGANTrainer)
├── stargan/                         # StarGAN multi-domain translation (StarGANTrainer)
└── cut/                            # CUT contrastive unpaired translation (CUTTrainer)
```
