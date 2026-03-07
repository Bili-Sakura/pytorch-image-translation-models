# pytorch-image-translation-models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![PyPI version](https://img.shields.io/pypi/v/pytorch-image-translation-models.svg)](https://pypi.org/project/pytorch-image-translation-models/)

A PyTorch library for multi-modal image translation with diffusion bridges, GANs, and transformer backbones.


## Installation

### Install from PyPI

```bash
pip install pytorch-image-translation-models
```

### Install from source

```bash
pip install -e .
```

With optional dependencies:

```bash
# With training extras (accelerate, peft, datasets, tensorboard)
pip install -e ".[training]"

# With metrics extras (torchmetrics, lpips, torch-fidelity, scipy)
pip install -e ".[metrics]"

# Everything
pip install -e ".[all]"
```

> **Note:** PyTorch is listed as a dependency but you may want to install a specific CUDA build first.
> See [PyTorch — Get Started](https://pytorch.org/get-started/previous-versions/) for details.

## Features

### Models

- **GAN generators** — `UNetGenerator` (encoder-decoder with skip connections), `ResNetGenerator` (residual blocks)
- **GAN discriminators** — `PatchGANDiscriminator` (Markovian patch-level classifier)
- **StegoGAN** — `ResnetMaskV1Generator`, `ResnetMaskV3Generator`, `NetMatchability` (steganographic masking for non-bijective translation, CVPR 2024)
- **Diffusion bridge** — `I2SBUNet` (ADM-style U-Net for Image-to-Image Schrödinger Bridge)
- **UNSB** — `UNSBGenerator`, `UNSBDiscriminator`, `UNSBEnergyNet` (time-conditional networks for Unpaired Neural Schrödinger Bridge, ICLR 2024)
- **Local Diffusion** — `LocalDiffusionUNet`, `ConditionEncoder` (conditional denoising U-Net with branch-and-fuse for hallucination suppression, ECCV 2024 Oral)
- **DiT backbone** — `SiTBackbone` (Scalable Interpolant Transformer for diffusion bridges)

### Schedulers

| Scheduler | Description |
|---|---|
| **I2SBScheduler** | Symmetric beta schedule with forward/reverse bridge kernels for I2SB |
| **DDBMScheduler** | Karras sigma schedule with Heun/Euler sampling for DDBM (VP/VE modes) |
| **BiBBDMScheduler** | Brownian Bridge noise schedule with bidirectional sampling for BiBBDM |
| **DDIBScheduler** | Gaussian diffusion with DDIM forward/reverse steps for DDIB |
| **BDBMScheduler** | Bidirectional Brownian Bridge schedule for BDBM |
| **DBIMScheduler** | Faster bridge sampler with eta-controlled stochasticity for DBIM |
| **CDTSDEScheduler** | Dynamic domain-shift eta schedule for CDTSDE |
| **LBMScheduler** | Flow-matching bridge for single/few-step LBM translation |
| **UNSBScheduler** | Non-uniform harmonic time schedule with stochastic bridge dynamics for UNSB |
| **LocalDiffusionScheduler** | Gaussian diffusion (DDPM/DDIM) with sigmoid/cosine/linear beta schedules for Local Diffusion |

### Pipelines

| Pipeline | Description |
|---|---|
| **I2SBPipeline** | End-to-end inference for I2SB models |
| **DDBMPipeline** | DDBM bridge diffusion with Heun's method |
| **BiBBDMPipeline** | Bidirectional Brownian Bridge translation (b2a / a2b) |
| **DDIBPipeline** | Dual-model DDIM encode/decode translation |
| **BDBMPipeline** | Bidirectional diffusion bridge with context conditioning |
| **DBIMPipeline** | Fast DBIM bridge sampling with bridge preconditioning |
| **CDTSDEPipeline** | CDTSDE with dynamic domain-shift scheduling |
| **LBMPipeline** | LBM flow-matching for single/few-step image translation |
| **UNSBPipeline** | Multi-step Schrödinger Bridge with adversarial + contrastive losses |
| **LocalDiffusionPipeline** | Branch-and-fuse diffusion for hallucination-aware image translation |

All pipelines support `"pt"`, `"pil"`, and `"np"` output types.

### Data

- `PairedImageDataset` / `UnpairedImageDataset` with configurable transform pipelines

### Losses

- `GANLoss` (vanilla / LSGAN / hinge), VGG-based `PerceptualLoss`

### Training

- `Pix2PixTrainer` — Paired GAN training with checkpoint save/load
- `StegoGANTrainer` — StegoGAN unpaired training with steganographic masking and consistency losses
- `I2SBTrainer` — I2SB bridge model training (in `examples/i2sb/`)

### Metrics

- `compute_psnr`, `compute_ssim`, `compute_lpips`, `compute_fid`

### Community Pipelines

Self-contained, single-file modules contributed by the community (inspired by [diffusers community pipelines](https://github.com/huggingface/diffusers/tree/main/examples/community)):

| Pipeline | Paper | Description |
|----------|-------|-------------|
| [`parallel_gan.py`](examples/community/parallel_gan.py) | [Wang et al., TGRS 2022](https://ieeexplore.ieee.org/document/9864654) | SAR-to-Optical with hierarchical latent features |

## Quick Start

### GAN-based translation (Pix2Pix)

```python
import src

gen = src.UNetGenerator(in_channels=3, out_channels=3)
disc = src.PatchGANDiscriminator(in_channels=6)

from src.training import Pix2PixTrainer, TrainingConfig
config = TrainingConfig(epochs=100, device="cuda")
trainer = Pix2PixTrainer(gen, disc, config)
trainer.fit(dataloader)  # expects {"source": tensor, "target": tensor}

translator = src.ImageTranslator(gen, device="cuda")
result = translator.predict(pil_image)
```

### Diffusion bridge translation (I2SB)

```python
from src.models.unet import I2SBUNet, create_model
from src.schedulers import I2SBScheduler
from src.pipelines.i2sb import I2SBPipeline

# Create model and scheduler
model = create_model(
    image_size=256, in_channels=3, num_channels=128,
    num_res_blocks=2, attention_resolutions="32,16,8",
    condition_mode="concat",
)
scheduler = I2SBScheduler(interval=1000, beta_max=0.3)

# Inference pipeline
pipeline = I2SBPipeline(unet=model, scheduler=scheduler)
result = pipeline(source_tensor, nfe=20, output_type="pt")
```

### DDBM bridge diffusion

```python
from src.schedulers import DDBMScheduler
from src.pipelines import DDBMPipeline

scheduler = DDBMScheduler(pred_mode="vp", num_train_timesteps=40)
pipeline = DDBMPipeline(unet=my_unet, scheduler=scheduler)
result = pipeline(source_image, num_inference_steps=40, output_type="pil")
```

### BiBBDM bidirectional translation

```python
from src.schedulers import BiBBDMScheduler
from src.pipelines import BiBBDMPipeline

scheduler = BiBBDMScheduler(num_timesteps=1000, sample_step=100)
pipeline = BiBBDMPipeline(unet=my_unet, scheduler=scheduler)
# Source → Target
result = pipeline(source_tensor, direction="b2a", output_type="pt")
# Target → Source
result = pipeline(target_tensor, direction="a2b", output_type="pt")
```

### DDIB dual-model translation

```python
from src.schedulers import DDIBScheduler
from src.pipelines import DDIBPipeline

scheduler = DDIBScheduler(num_train_timesteps=1000)
pipeline = DDIBPipeline(source_unet=src_model, target_unet=tgt_model, scheduler=scheduler)
result = pipeline(source_image, num_inference_steps=250, output_type="pil")
```

### LBM flow-matching translation

```python
from src.schedulers import LBMScheduler
from src.pipelines import LBMPipeline

scheduler = LBMScheduler(num_train_timesteps=1000)
pipeline = LBMPipeline(unet=my_unet, scheduler=scheduler)
result = pipeline(source_image, num_inference_steps=1, output_type="pil")
```

### DiT backbone (SiT) for diffusion bridges

```python
from src.models.dit import SiTBackbone, SIT_CONFIGS

# Create a SiT-S/2 backbone (small, patch size 2)
depth, hidden_size, num_heads = SIT_CONFIGS["S"]
model = SiTBackbone(
    image_size=256, patch_size=2, in_channels=3,
    hidden_size=hidden_size, depth=depth, num_heads=num_heads,
    condition_mode="concat",
)
# Use as drop-in replacement for UNet in any bridge pipeline
output = model(noisy_sample, timestep, xT=source_image)
```

### UNSB unpaired translation (multi-step Schrödinger Bridge)

```python
from src.models.unsb import create_generator
from src.schedulers.unsb import UNSBScheduler
from src.pipelines.unsb import UNSBPipeline

# Create time-conditional generator and scheduler
generator = create_generator(input_nc=3, output_nc=3, ngf=64, n_blocks=9)
scheduler = UNSBScheduler(num_timesteps=5, tau=0.01)

# Inference pipeline (multi-step stochastic refinement)
pipeline = UNSBPipeline(generator=generator, scheduler=scheduler)
result = pipeline(source_image, output_type="pt")
print(result.nfe)  # 5 function evaluations
```

### UNSB training

```python
from examples.unsb.config import UNSBConfig
from examples.unsb.train_unsb import UNSBTrainer

cfg = UNSBConfig(
    input_nc=3, output_nc=3, ngf=64,
    num_timesteps=5, tau=0.01,
    lambda_GAN=1.0, lambda_SB=1.0, lambda_NCE=1.0,
    device="cuda",
)
trainer = UNSBTrainer(cfg)
# Single training step with unpaired data
losses = trainer.train_step(real_A_batch, real_B_batch)
```

### Local Diffusion hallucination-aware translation

```python
from src.models.local_diffusion import create_unet
from src.schedulers.local_diffusion import LocalDiffusionScheduler
from src.pipelines.local_diffusion import LocalDiffusionPipeline

# Create conditional U-Net and Gaussian diffusion scheduler
unet = create_unet(dim=32, channels=1, dim_mults=(1, 2, 4, 8))
scheduler = LocalDiffusionScheduler(num_train_timesteps=250, beta_schedule="sigmoid")

# Standard inference
pipeline = LocalDiffusionPipeline(unet=unet, scheduler=scheduler)
result = pipeline(cond_image, output_type="pt")

# Branch-and-fuse inference (hallucination suppression)
result = pipeline(
    cond_image, anomaly_mask=mask,
    branch_out=True, fusion_timestep=2, output_type="pt",
)
```

### Local Diffusion training

```python
from examples.local_diffusion.config import LocalDiffusionConfig
from examples.local_diffusion.train_local_diffusion import LocalDiffusionTrainer

cfg = LocalDiffusionConfig(
    dim=32, channels=1,
    num_train_timesteps=250, beta_schedule="sigmoid",
    objective="pred_x0", device="cuda",
)
trainer = LocalDiffusionTrainer(cfg)
losses = trainer.train_step(source_batch, target_batch)
```

### I2SB training with task configs

```python
from examples.i2sb.config import sar2eo_config
from examples.i2sb.trainer import I2SBTrainer

cfg = sar2eo_config(resolution=256, train_batch_size=8)
trainer = I2SBTrainer(cfg)
model = trainer.build_model()
scheduler = trainer.build_scheduler()

# Single-step loss computation
loss = I2SBTrainer.compute_training_loss(model, scheduler, source_batch, target_batch)
loss.backward()
```

### StegoGAN non-bijective translation

```python
from src.training import StegoGANTrainer, StegoGANConfig

cfg = StegoGANConfig(
    input_nc=3, output_nc=3, ngf=64,
    lambda_reg=0.3, lambda_consistency=1.0,
    resnet_layer=8, fusionblock=True,
    device="cuda",
)
trainer = StegoGANTrainer(cfg)
# Run a single training step with unpaired data
losses = trainer.train_step(real_A_batch, real_B_batch)
```

## Package Structure

```
src/                                 # ← Core library (single source of truth)
├── __init__.py                      # Public API
├── models/
│   ├── generators.py                # UNetGenerator, ResNetGenerator
│   ├── discriminators.py            # PatchGANDiscriminator
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
├── schedulers/                      # One scheduler per method
│   ├── i2sb.py                     # I2SBScheduler
│   ├── ddbm.py                     # DDBMScheduler
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
│   ├── bibbdm.py                   # BiBBDMPipeline
│   ├── ddib.py                     # DDIBPipeline
│   ├── bdbm.py                     # BDBMPipeline
│   ├── dbim.py                     # DBIMPipeline
│   ├── cdtsde.py                   # CDTSDEPipeline
│   └── lbm.py                      # LBMPipeline
│   └── unsb.py                     # UNSBPipeline
│   └── local_diffusion.py          # LocalDiffusionPipeline
├── data/
│   ├── datasets.py                 # PairedImageDataset, UnpairedImageDataset
│   └── transforms.py               # get_transforms, default_transforms
├── losses/
│   ├── adversarial.py              # GANLoss
│   └── perceptual.py               # PerceptualLoss
├── training/
│   ├── trainer.py                  # Pix2PixTrainer, TrainingConfig
│   └── stegogan_trainer.py         # StegoGANTrainer, StegoGANConfig
├── inference/
│   └── predictor.py                # ImageTranslator
└── metrics/
    └── image_quality.py            # PSNR, SSIM, LPIPS, FID
examples/                            # ← Training/inference scripts (import from src/)
├── community/                       # Community-contributed pipelines (single-file)
│   └── parallel_gan.py             # Parallel-GAN (Wang et al., TGRS 2022)
├── i2sb/                            # I2SB paper-oriented training code
│   ├── config.py                   # TaskConfig, sar2eo_config, etc.
│   └── trainer.py                  # I2SBTrainer
└── inference/
    └── run_inference.py            # Unified inference script for all methods
```

## Credits

### Reference papers

- [I2SB: Image-to-Image Schrödinger Bridge (ICML 2023)](https://openreview.net/forum?id=WH2Cy3eQd0)
- [DDBM: Denoising Diffusion Bridge Models (ICLR 2024)](https://openreview.net/forum?id=FKksTayvGo)
- [DDIB: Dual Diffusion Implicit Bridges (ICLR 2023)](https://openreview.net/forum?id=5HLoTvVGDe)
- [BBDM: Image-to-Image Translation with Brownian Bridge Diffusion Models (CVPR 2023)](http://openaccess.thecvf.com/content/CVPR2023/papers/Li_BBDM_Image-to-Image_Translation_With_Brownian_Bridge_Diffusion_Models_CVPR_2023_paper.pdf)
- [DBIM: Diffusion Bridge Implicit Models (2024)](https://arxiv.org/abs/2405.15885)
- [LBM: Latent Bridge Matching for Fast Image-to-Image Translation (2025)](https://arxiv.org/abs/2503.07535)
- [SiT: Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers (2024)](https://arxiv.org/abs/2401.08740)
- [StegoGAN: Leveraging Steganography for Non-Bijective Image-to-Image Translation (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/papers/Wu_StegoGAN_Leveraging_Steganography_for_Non-Bijective_Image-to-Image_Translation_CVPR_2024_paper.pdf)
- [Parallel-GAN: SAR-to-Optical Image Translation with Hierarchical Latent Features (TGRS 2022)](https://ieeexplore.ieee.org/document/9864654)
- [CUT: Contrastive Unpaired Translation (ECCV 2020)](https://link.springer.com/chapter/10.1007/978-3-030-58545-7_19)
- [UNSB: Unpaired Image-to-Image Translation via Neural Schrödinger Bridge (ICLR 2024)](https://openreview.net/forum?id=uQBW7ELXfO)
- [Local Diffusion: Tackling Structural Hallucination in Image Translation with Local Diffusion (ECCV 2024 Oral)](https://arxiv.org/abs/2407.17578)
- [CycleGAN (ICCV 2017)](https://openaccess.thecvf.com/content_iccv_2017/html/Zhu_Unpaired_Image-To-Image_Translation_ICCV_2017_paper.html)
- [img2img-turbo (2024)](https://doi.org/10.48550/arXiv.2403.12036)

## License

MIT
