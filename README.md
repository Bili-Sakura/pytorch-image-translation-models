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
- **Diffusion bridge** — `I2SBUNet` (ADM-style U-Net for Image-to-Image Schrödinger Bridge)

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

All pipelines support `"pt"`, `"pil"`, and `"np"` output types.

### Data

- `PairedImageDataset` / `UnpairedImageDataset` with configurable transform pipelines

### Losses

- `GANLoss` (vanilla / LSGAN / hinge), VGG-based `PerceptualLoss`

### Training

- `Pix2PixTrainer` — Paired GAN training with checkpoint save/load
- `I2SBTrainer` — I2SB bridge model training (in `examples/i2sb/`)

### Metrics

- `compute_psnr`, `compute_ssim`, `compute_lpips`, `compute_fid`

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

## Package Structure

```
src/
├── __init__.py              # Public API
├── models/
│   ├── generators.py        # UNetGenerator, ResNetGenerator
│   ├── discriminators.py    # PatchGANDiscriminator
│   └── unet/                # ADM-style U-Net for I2SB
│       ├── i2sb_unet.py     # I2SBUNet
│       └── unet_2d.py       # create_model factory
├── schedulers/
│   ├── i2sb.py              # I2SBScheduler
│   ├── ddbm.py              # DDBMScheduler
│   ├── bibbdm.py            # BiBBDMScheduler
│   ├── ddib.py              # DDIBScheduler
│   ├── bdbm.py              # BDBMScheduler
│   ├── dbim.py              # DBIMScheduler
│   └── cdtsde.py            # CDTSDEScheduler
├── pipelines/
│   ├── i2sb.py              # I2SBPipeline
│   ├── ddbm.py              # DDBMPipeline
│   ├── bibbdm.py            # BiBBDMPipeline
│   ├── ddib.py              # DDIBPipeline
│   ├── bdbm.py              # BDBMPipeline
│   ├── dbim.py              # DBIMPipeline
│   └── cdtsde.py            # CDTSDEPipeline
├── data/
│   ├── datasets.py          # PairedImageDataset, UnpairedImageDataset
│   └── transforms.py        # get_transforms, default_transforms
├── losses/
│   ├── adversarial.py       # GANLoss
│   └── perceptual.py        # PerceptualLoss
├── training/
│   └── trainer.py           # Pix2PixTrainer, TrainingConfig
├── inference/
│   └── predictor.py         # ImageTranslator
└── metrics/
    └── image_quality.py     # PSNR, SSIM, LPIPS, FID
examples/
└── i2sb/
    ├── config.py            # TaskConfig, sar2eo_config, etc.
    └── trainer.py           # I2SBTrainer
```

## Credits

### Reference papers

- [I2SB: Image-to-Image Schrödinger Bridge (ICML 2023)](https://openreview.net/forum?id=WH2Cy3eQd0)
- [DDBM: Denoising Diffusion Bridge Models (ICLR 2024)](https://openreview.net/forum?id=FKksTayvGo)
- [DDIB: Dual Diffusion Implicit Bridges (ICLR 2023)](https://openreview.net/forum?id=5HLoTvVGDe)
- [BBDM: Image-to-Image Translation with Brownian Bridge Diffusion Models (CVPR 2023)](http://openaccess.thecvf.com/content/CVPR2023/papers/Li_BBDM_Image-to-Image_Translation_With_Brownian_Bridge_Diffusion_Models_CVPR_2023_paper.pdf)
- [DBIM: Diffusion Bridge Implicit Models (2024)](https://arxiv.org/abs/2405.15885)
- [CUT: Contrastive Unpaired Translation (ECCV 2020)](https://link.springer.com/chapter/10.1007/978-3-030-58545-7_19)
- [CycleGAN (ICCV 2017)](https://openaccess.thecvf.com/content_iccv_2017/html/Zhu_Unpaired_Image-To-Image_Translation_ICCV_2017_paper.html)
- [img2img-turbo (2024)](https://doi.org/10.48550/arXiv.2403.12036)

## License

MIT
