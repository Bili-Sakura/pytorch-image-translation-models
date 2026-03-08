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

## Documentation

| Doc | Description |
| --- | --- |
| [Features](docs/features.md) | Models, schedulers, pipelines, data, losses, training, metrics |
| [Examples](docs/examples.md) | Extended usage for I2SB, DDBM, UNSB, Local Diffusion, etc. |
| [Package structure](docs/package-structure.md) | Source layout and module overview |
| [Credits](docs/credits.md) | Reference papers and citations |

## License

MIT
