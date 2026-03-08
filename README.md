# pytorch-image-translation-models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![PyPI version](https://img.shields.io/pypi/v/pytorch-image-translation-models.svg)](https://pypi.org/project/pytorch-image-translation-models/) [![Checkpoint Collections](https://img.shields.io/badge/🤗%20Checkpoints-Collection-FFD21E)](https://huggingface.co/collections/BiliSakura/image-translation-checkpoint-collections)

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

## Quick Start

```python
import src
from PIL import Image
from examples.community.e3diff import E3DiffPipeline

# Baseline method (DDBM) - one-stop load
ddbm = src.DDBMPipeline.from_pretrained(
    "/path/to/DDBM-ckpt/diode-vp",
    subfolder="unet",
    device="cuda",
)

source = Image.open("/path/to/source.png").convert("RGB")
baseline_out = ddbm(source_image=source, num_inference_steps=40, output_type="pil")
baseline_out.images[0].save("ddbm_output.png")

# Community method (E3Diff) - one-stop load
e3diff = E3DiffPipeline.from_pretrained(
    "/path/to/E3Diff-ckpt/SEN12 ",
    device="cuda",
)
community_out = e3diff(source_image=source, num_inference_steps=50, output_type="pil")
community_out.images[0].save("e3diff_output.png")
```

## Checkpoint Layout Conventions

`from_pretrained(...)` expects method-specific subfolders under your checkpoint root.

| Method | Expected layout (relative to method root) |
| --- | --- |
| `DDBM`, `BDBM`, `BiBBDM`, `DBIM`, `CDTSDE`, `LBM` | `unet/config.json`, `unet/diffusion_pytorch_model.safetensors`, optional `scheduler/scheduler_config.json` |
| `DDIB` | `source_unet/config.json`, `source_unet/diffusion_pytorch_model.safetensors`, `target_unet/config.json`, `target_unet/diffusion_pytorch_model.safetensors`, optional `scheduler/scheduler_config.json` |
| `I2SB` | `unet/config.json`, `unet/diffusion_pytorch_model.safetensors`, optional `scheduler_config.json` (or `scheduler/scheduler_config.json`) |
| `CUT` | `generator/config.json`, `generator/diffusion_pytorch_model.safetensors` |
| `Pix2PixHD` | `generator/config.json`, `generator/diffusion_pytorch_model.safetensors` |
| `UNSB` | `generator/config.json`, `generator/diffusion_pytorch_model.safetensors` |
| `StegoGAN` | `generator_A/config.json`, `generator_A/diffusion_pytorch_model.safetensors`, `generator_B/config.json`, `generator_B/diffusion_pytorch_model.safetensors` |
| `LocalDiffusion` | `unet/config.json`, `unet/diffusion_pytorch_model.safetensors`, optional `scheduler/scheduler_config.json` |
| Community `sar2optical` | `generator/config.json`, `generator/diffusion_pytorch_model.safetensors` |
| Community `parallel_gan` | `generator/config.json`, `generator/diffusion_pytorch_model.safetensors` |
| Community `e3diff` | `config.json`, `diffusion_pytorch_model.safetensors` |

## Documentation

| Doc | Description |
| --- | --- |
| [Features](docs/features.md) | Models, schedulers, pipelines, data, losses, training, metrics |
| [Examples](docs/examples.md) | Extended usage for I2SB, DDBM, UNSB, Local Diffusion, etc. |
| [Package structure](docs/package-structure.md) | Source layout and module overview |
| [Credits](docs/credits.md) | Reference papers and citations |

## License

MIT
