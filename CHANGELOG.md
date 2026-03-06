# Changelog

All notable changes to **pytorch-image-translation-models** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-03-06

### Added

- **Diffusers UNet wrappers** (`src/models/unet/diffusers_wrappers.py`): consolidated all 8 UNet wrapper classes (`DDBMUNet`, `DDIBUNet`, `I2SBDiffusersUNet`, `BiBBDMUNet`, `BDBMUNet`, `DBIMUNet`, `CDTSDEUNet`, `LBMUNet`) into a single shared module. These wrappers pair `diffusers.ModelMixin`/`ConfigMixin` with `UNet2DModel` for each method's calling convention.
- Added `guidance` parameter to `DDBMScheduler.step()` and `DDBMScheduler.step_heun()` for scaling the ODE derivative.
- Added `ot_ode` parameter to `I2SBScheduler.step()` and `I2SBPipeline.__call__()` for deterministic OT-ODE sampling.
- New `examples/inference/run_inference.py`: unified inference script for all 8 methods, importing components from `src/` instead of duplicating code.

### Changed

- **Major architecture refactor** following the [diffusers](https://github.com/huggingface/diffusers) project structure:
  - `src/` is the single source of truth for models, schedulers, and pipelines.
  - `examples/` contains paper-oriented training code and documentation (no duplicated pipeline code).
  - `examples/community/` provides community-contributed self-contained pipelines.
- **Removed `examples/pipelines/`** (~3300 lines of duplicated code): the 8 self-contained pipeline directories have been replaced by importing from `src/models/`, `src/schedulers/`, and `src/pipelines/`.
- Rewrote `test_pipeline_examples.py` and `test_dit_lbm.py` to import from `src/` exclusively.
- Renamed the diffusers-compatible I2SB UNet wrapper to `I2SBDiffusersUNet` to avoid conflict with the native `I2SBUNet` backbone.

## [0.1.3] - 2026-03-06

### Added

- **Community pipelines** (`examples/community/`): a new contribution model inspired by [Hugging Face diffusers community pipelines](https://github.com/huggingface/diffusers/tree/main/examples/community). Each community pipeline is a self-contained, single-file module that bundles all model, loss, and utility code.
  - `examples/community/README.md` with contribution guidelines and pipeline catalog.
  - `examples/community/__init__.py` module docstring documenting the pattern.
- **Parallel-GAN community pipeline** (Wang et al., IEEE TGRS 2022): SAR-to-optical translation with hierarchical latent features.
  - `examples/community/parallel_gan.py`: self-contained module with `ParaGAN` (translation generator), `Resrecon` (reconstruction network), `VGGLoss`, `ParallelGANTrainer`, and `ParallelGANConfig`.
  - Two-stage training: Stage 1 (reconstruction) and Stage 2 (translation with frozen recon-net feature supervision).
  - Tests for all Parallel-GAN components (15 test cases).

## [0.1.2] - 2026-03-06

### Added

- **StegoGAN integration** (Wu et al., CVPR 2024): non-bijective image-to-image translation with steganographic masking.
  - `ResnetMaskV1Generator` (`G_A`): ResNet generator with optional steganographic feature injection.
  - `ResnetMaskV3Generator` (`G_B`): ResNet generator with per-pixel matchability masking via `NetMatchability`.
  - Helper modules: `NetMatchability`, `mask_generate`, `ResnetBlock`, `SoftClamp`.
  - `StegoGANTrainer` and `StegoGANConfig` for end-to-end training with cycle, identity, consistency, and mask regularisation losses.
  - Tests for all new StegoGAN components (19 test cases).

### Changed

- **Refactored I2SB code duplication**: the standalone `examples/pipelines/i2sb/pipeline.py` now imports the core `I2SBScheduler` from `src.schedulers.i2sb` instead of re-implementing ~120 lines of scheduler math.
- Added `clip_denoise` parameter to `I2SBScheduler.compute_pred_x0()`.
- Updated package exports to include StegoGAN models and trainer.

## [0.1.1] - 2026-03-06

### Changed

- Bumped package version to `0.1.1`.
- Prepared release metadata for PyPI publication.

## [0.1.0] - 2026-03-05

### Added

- Initial release of `pytorch-image-translation-models`.
- **GAN Models**: UNet and ResNet generators, PatchGAN discriminator.
- **Diffusion Bridge Models**: I2SB UNet backbone (ADM-style), I2SB Scheduler, I2SB Pipeline.
- **Schedulers**: I2SBScheduler with symmetric beta schedule.
- **Pipelines**: I2SBPipeline for end-to-end inference (pt, pil, np outputs).
- **Data**: PairedImageDataset, UnpairedImageDataset with configurable transforms.
- **Losses**: GANLoss (vanilla/LSGAN/hinge), VGG PerceptualLoss.
- **Training**: Pix2PixTrainer (GAN), I2SBTrainer (diffusion bridge).
- **Inference**: ImageTranslator for single-image, batch, and file-based prediction.
- **Metrics**: PSNR, SSIM, LPIPS, FID evaluation helpers.
- **Examples**: I2SB task configs (sar2eo, rgb2ir, sar2ir, sar2rgb).
- Packaging via `pyproject.toml` with optional dependency groups (`training`, `metrics`, `dev`, `all`).
- GitHub Actions workflow for automated PyPI publishing on tagged releases.

[Unreleased]: https://github.com/Bili-Sakura/pytorch-image-translation-models/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/Bili-Sakura/pytorch-image-translation-models/releases/tag/v0.2.0
[0.1.3]: https://github.com/Bili-Sakura/pytorch-image-translation-models/releases/tag/v0.1.3
[0.1.2]: https://github.com/Bili-Sakura/pytorch-image-translation-models/releases/tag/v0.1.2
[0.1.1]: https://github.com/Bili-Sakura/pytorch-image-translation-models/releases/tag/v0.1.1
[0.1.0]: https://github.com/Bili-Sakura/pytorch-image-translation-models/releases/tag/v0.1.0
