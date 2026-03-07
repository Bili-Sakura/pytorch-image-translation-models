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

gen = src.UNetGenerator(in_channels=3, out_channels=3)
disc = src.PatchGANDiscriminator(in_channels=6)

from src.training import Pix2PixTrainer, TrainingConfig
config = TrainingConfig(epochs=100, device="cuda")
trainer = Pix2PixTrainer(gen, disc, config)
trainer.fit(dataloader)  # expects {"source": tensor, "target": tensor}

translator = src.ImageTranslator(gen, device="cuda")
result = translator.predict(pil_image)
```

## Documentation

| Doc | Description |
|-----|-------------|
| [Features](docs/features.md) | Models, schedulers, pipelines, data, losses, training, metrics |
| [Examples](docs/examples.md) | Extended usage for I2SB, DDBM, UNSB, Local Diffusion, etc. |
| [Package structure](docs/package-structure.md) | Source layout and module overview |
| [Credits](docs/credits.md) | Reference papers and citations |

## License

MIT
