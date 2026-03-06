# Changelog

All notable changes to **pytorch-image-translation-models** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/Bili-Sakura/pytorch-image-translation-models/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/Bili-Sakura/pytorch-image-translation-models/releases/tag/v0.1.2
[0.1.1]: https://github.com/Bili-Sakura/pytorch-image-translation-models/releases/tag/v0.1.1
[0.1.0]: https://github.com/Bili-Sakura/pytorch-image-translation-models/releases/tag/v0.1.0
