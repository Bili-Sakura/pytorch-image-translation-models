# Changelog

All notable changes to **pytorch-image-translation-models** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-03-12

### Added

- **Unified diffusion loss** (`src/losses/diffusion.py`): single `DiffusionLoss` module supporting ε-prediction, x0-prediction, v-prediction, EDM preconditioning, and SiD2 sigmoid weighting. Loss types: `mse` (plain), `min_snr` (SNR weighting), `sid2` (SiD2 continuous/discrete), `edm`.
- **`get_diffusion_loss(loss_type, prediction_type, **kwargs)`**: one-line factory for diffusion loss used across local_diffusion, I2SB, and other methods.
- **SiD2 cosine log-SNR** (`cosine_interpolated_logsnr`): exact schedule from SiD2 Appendix B for continuous t ∈ [0,1].
- **Target override**: `DiffusionLoss` accepts precomputed `target=` for bridge methods (e.g. I2SB with custom labels).
- **`loss_norm`**: optional `"l1"` or `"mse"` pixel loss for E3Diff-style training.

### Changed

- **Local Diffusion** (`examples/local_diffusion`): now uses `DiffusionLoss` via `loss_type` config (`mse` | `min_snr` | `sid2` | `edm`). Replaced `scheduler.compute_loss` with unified loss.
- **I2SB** (`examples/i2sb`): now uses `DiffusionLoss` with `target=` override for bridge labels. Added `loss_type` to `TaskConfig`.
- Exported `get_diffusion_loss` from `src.losses`.

## [0.2.10] - 2026-03-12

### Added

- **SiD2 UNet** *(new)* (`src/models/unet/sid2.py`): Residual U-ViT backbone for Simpler Diffusion v2 (Hoogeboom et al., CVPR 2025). Uses diffusers `ResnetBlock2D` and `BasicTransformerBlock` with a single level-wise residual skip. `SiD2UNet`, `SiD2ResBlock2D`, `SiD2TransformerBlock` exported from `src.models.unet`.

## [0.2.9] - 2026-03-12

### Added

- **MDT/LDDBM community pipeline** (`examples/community/mdt/`): Latent diffusion bridge for general modality translation (Bosch Research, NeurIPS 2025 submission). Compatible with [Multimodal-Distribution-Translation-MDT](https://github.com/boschresearch/Multimodal-Distribution-Translation-MDT).
  - `MDTPipeline`, `MDTPipelineOutput`, `load_mdt_community_pipeline`.
  - `convert_pt_to_mdt.py` for converting raw `.pt` checkpoints to the project layout (config.json + safetensors per component).
  - Supports super-resolution (16×16 → 128×128) and multi-view→3D (ShapeNet) tasks.

## [0.2.8] - 2026-03-11

### Added

- **L1 and L2 metrics** for paired image evaluation: `compute_l1` (MAE) and `compute_l2` (MSE) in `src/metrics/paired/l1_l2.py`. Integrated into `PairedImageMetricEvaluator` and metric registry.
- **DiffuseIT community pipeline** (`examples/community/diffuseit/`): diffusion-based image translation (Kwon & Ye, ICLR 2023) with text-guided and image-guided modes. Compatible with [BiliSakura/DiffuseIT-ckpt](https://huggingface.co/BiliSakura/DiffuseIT-ckpt).
  - `DiffuseITPipeline`, `DiffuseITPipelineOutput`, `load_diffuseit_community_pipeline`.
  - `convert_ckpt_to_diffuseit.py` for converting raw DiffuseIT checkpoints to the BiliSakura layout.
- **CDTSDE community pipeline** (`examples/community/cdtsde/`): ControlLDM for solar defect identification. `convert_ckpt_to_cdtsde.py` for raw `.ckpt` conversion; `load_cdtsde_community_pipeline` for inference.
- **DDIB community pipeline** (`examples/community/ddib/`): dual source/target UNets for OpenAI/guided_diffusion-style checkpoints. `OpenAIDDIBUNet`, `load_ddib_community_pipeline`, `convert_pt_to_ddib.py`.

### Changed

- **DiffuseIT moved to community**: DiffuseIT is now a community pipeline under `examples/community/diffuseit/` instead of `examples/baselines/diffuseit/`. Removed `examples/baselines/` structure.
- Updated README, docs (`examples.md`, `features.md`, `package-structure.md`, `checkpoint-layouts.md`), and community catalog to reflect DiffuseIT, CDTSDE, and DDIB as integrated pipelines.

## [0.2.7] - 2026-03-11

### Added

- **iFID** (interpolated FID, Xu et al. 2026): full implementation with VAE encode → nearest-neighbor latent interpolation → decode → FID. Supports multiple diffusers VAEs (`AutoencoderKL`, `AutoencoderTiny`, `AsymmetricAutoencoderKL`, `ConsistencyDecoderVAE`) via `vae_path` + `vae_cls` or pre-loaded `vae`.
- **FWD** (Fréchet Wavelet Distance, ICLR 2025): implementation via `pytorchfwd` subprocess. Domain-agnostic wavelet-based evaluation.
- **sFID** (Sparse FID, Nash et al. ICML 2021): implementation via `pyiqa` spatial Inception features.

### Changed

- **Metrics no-fallback policy**: iFID, FWD, and sFID no longer fall back to standard FID when required backends are missing. They now raise `ImportError` or `ValueError` with prominent stderr warnings.
- iFID requires `vae_path` or `vae`; FWD requires `pytorchfwd`; sFID requires `pyiqa`.
- Documented a CUDA-first usage pattern across docs (`README.md`, `docs/features.md`, `docs/examples.md`), including explicit guidance to switch to CPU with `"cpu"` when needed.
- Standardized examples to show `pipeline.to("cuda")` / `pipeline.to("cpu")` as the preferred device placement flow.
- Improved custom pipeline ergonomics for device migration:
  - `I2SBPipeline` now exposes `device`/`dtype` properties and auto-moves tensor inputs to the pipeline device during inference.
  - `StegoGANPipeline` now exposes `device`/`dtype` properties and auto-moves tensor inputs to the pipeline device during inference.
- Added regression tests for `.to(...)` behavior in `tests/test_pipeline_examples.py` (I2SB) and `tests/test_stegogan.py` (StegoGAN).

## [0.2.6] - 2026-03-10

### Added

- Native **StarGAN integration** in `src/`:
  - `src/models/stargan/` (`StarGANGenerator`, `StarGANDiscriminator`, `StarGANResidualBlock`)
  - `src/pipelines/stargan.py` (`StarGANPipeline`, `load_stargan_pipeline`)
  - `src/training/stargan_trainer.py` (`StarGANTrainer`, `StarGANTrainingConfig`)
- StarGAN tests (`tests/test_stargan.py` and pipeline coverage in `tests/test_pipeline_examples.py`).

### Changed

- Decoupled **BBDM** from **BiBBDM** in `src/` by introducing explicit `BBDMUNet`, `BBDMScheduler`, and `BBDMPipeline` classes and exports.
- Updated unified inference and tests to treat `bbdm` as a standalone method (separate from `bibbdm`).
- Refactored model layout under `src/models/` from flat files to package folders:
  - `discriminators/`, `generators/`, `pix2pixhd/`, and `stargan/`.

## [0.2.5] - 2026-03-09

### Added

- **DDBM community pipeline** (`examples/community/ddbm/`): compatible with [BiliSakura/DDBM-ckpt](https://huggingface.co/BiliSakura/DDBM-ckpt) OpenAI-style checkpoints.
  - Custom UNet architecture and conversion utilities (`convert_pt_to_unet.py`) to convert raw `.pt` checkpoints to the required unet format.
  - `load_ddbm_pipeline` and `from_pretrained` methods for loading DDBM pipelines from local directories.
  - Documentation with usage examples and checkpoint layout conventions.
- **`from_pretrained` pipeline loading**: implemented for BDBM, BiBBDM, CDTSDE, CUT, DBIM, DDBM, DDIB, I2SB, LBM, LocalDiffusion, Pix2PixHD, StegoGAN, UNSB, plus community pipelines (E3Diff, ParallelGAN, SAR2Optical).
- **OpenAI-style DDBM UNet** (`src/models/unet/openai_ddbm_unet.py`) for compatibility with existing DDBM checkpoints.

### Changed

- Enhanced error handling for missing configuration files and weights in pipeline loading.
- Updated community catalog to include DDBM as an integrated pipeline option.

## [0.2.4] - 2026-03-08

### Added

- **OpenEarthMap-SAR community pipeline** (`examples/community/openearthmap_sar/`): CUT models for SAR ↔ optical image translation, compatible with [OpenEarthMap-SAR](https://github.com/cliffbb/OpenEarthMap-SAR) checkpoints.
  - `OpenEarthMapSARGenerator`: CUT ResNet generator with anti-aliased down/upsampling.
  - `load_openearthmap_sar_pipeline`: loads safetensors or `.pth` checkpoints and returns `CUTPipeline`.
  - Supports `opt2sar`, `sar2opt`, `seman2opt`, `seman2sar` and pseudo variants.
  - CLI: `python -m examples.community.openearthmap_sar --checkpoint-dir PATH --model sar2opt`.
- Added `examples/community/sar2optical/` as an in-repo community pipeline adapted from [yuuIind/SAR2Optical](https://github.com/yuuIind/SAR2Optical), including:
  - `SAR2OpticalGenerator` and `SAR2OpticalDiscriminator` (pix2pix-style U-Net + PatchGAN/PixelGAN).
  - `SAR2OpticalPipeline` + `load_sar2optical_pipeline` for single-pass inference.
  - `SAR2OpticalConfig` + `SAR2OpticalTrainer` for cGAN training steps.

### Changed

- Updated community docs/catalog entries to list `openearthmap_sar/` and `sar2optical/` as integrated pipelines instead of external-only references.

## [0.2.2] - 2026-03-07

### Changed

- Refactored community pipelines into per-model subfolders under `examples/community/<model_name>/` with `model.py`, `pipeline.py`, `train.py`, and `README.md`.
- Updated `parallel_gan` and `e3diff` inference to use `DiffusionPipeline`-based pipeline classes.
- Normalized community documentation filenames from `readme.md` to `README.md`.

## [0.2.1] - 2026-03-06

### Added

- **StegoGAN pipeline** (`src/pipelines/stegogan.py`): `StegoGANPipeline` and `StegoGANPipelineOutput` for bidirectional inference with trained StegoGAN generators (`G_A`: A→B, `G_B`: B→A with matchability masking).

### Changed

- **Moved training code to `examples/`** following the [diffusers](https://github.com/huggingface/diffusers) project structure where training scripts live in self-contained example subfolders:
  - `src/training/trainer.py` → `examples/pix2pix/train_pix2pix.py`
  - `src/training/stegogan_trainer.py` → `examples/stegogan/train_stegogan.py`
  - `src/training/stargan_trainer.py` → `examples/stargan/train_stargan.py`
  - `src/training/` removed; all training code lives in `examples/`.
- Training classes (`Pix2PixTrainer`, `StegoGANTrainer`, `StegoGANConfig`, `StarGANTrainer`, `StarGANTrainingConfig`, `CUTTrainer`, `CUTConfig`) are no longer exported from the top-level `src` package; import from `examples.pix2pix`, `examples.stegogan`, `examples.stargan`, or `examples.cut` instead.

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

[Unreleased]: https://github.com/Bili-Sakura/pytorch-image-translation-models/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/Bili-Sakura/pytorch-image-translation-models/releases/tag/v0.3.0
[0.2.10]: https://github.com/Bili-Sakura/pytorch-image-translation-models/releases/tag/v0.2.10
[0.2.9]: https://github.com/Bili-Sakura/pytorch-image-translation-models/releases/tag/v0.2.9
[0.2.8]: https://github.com/Bili-Sakura/pytorch-image-translation-models/releases/tag/v0.2.8
[0.2.7]: https://github.com/Bili-Sakura/pytorch-image-translation-models/releases/tag/v0.2.7
[0.2.6]: https://github.com/Bili-Sakura/pytorch-image-translation-models/releases/tag/v0.2.6
[0.2.5]: https://github.com/Bili-Sakura/pytorch-image-translation-models/releases/tag/v0.2.5
[0.2.4]: https://github.com/Bili-Sakura/pytorch-image-translation-models/releases/tag/v0.2.4
[0.2.2]: https://github.com/Bili-Sakura/pytorch-image-translation-models/releases/tag/v0.2.2
[0.2.1]: https://github.com/Bili-Sakura/pytorch-image-translation-models/releases/tag/v0.2.1
[0.2.0]: https://github.com/Bili-Sakura/pytorch-image-translation-models/releases/tag/v0.2.0
[0.1.3]: https://github.com/Bili-Sakura/pytorch-image-translation-models/releases/tag/v0.1.3
[0.1.2]: https://github.com/Bili-Sakura/pytorch-image-translation-models/releases/tag/v0.1.2
[0.1.1]: https://github.com/Bili-Sakura/pytorch-image-translation-models/releases/tag/v0.1.1
[0.1.0]: https://github.com/Bili-Sakura/pytorch-image-translation-models/releases/tag/v0.1.0
