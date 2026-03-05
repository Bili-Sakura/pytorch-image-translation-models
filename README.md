# pytorch-image-translation-models

Production-ready PyTorch models and utilities for general image-to-image translation tasks.

## Features

- **Generator architectures** – U-Net and ResNet-based generators for paired and unpaired translation.
- **Discriminator architectures** – PatchGAN discriminator.
- **Data loading** – `PairedImageDataset` and `UnpairedImageDataset` with configurable transforms.
- **Loss functions** – GAN losses (vanilla, LSGAN, hinge) and VGG perceptual loss.
- **Training** – `Pix2PixTrainer` with checkpoint saving and logging.
- **Inference** – `ImageTranslator` for single-image, batch, and file-based prediction.
- **Metrics** – PSNR, SSIM, LPIPS and FID evaluation helpers.

## Installation

```bash
pip install pytorch-image-translation-models
```

Or install from source:

```bash
pip install -e .
```

## Quick Start

```python
import src

# Create models
generator = src.UNetGenerator(in_channels=3, out_channels=3)
discriminator = src.PatchGANDiscriminator(in_channels=6)

# Set up training
from src.training import Pix2PixTrainer, TrainingConfig

config = TrainingConfig(epochs=100, device="cuda")
trainer = Pix2PixTrainer(generator, discriminator, config)

# Train (dataloader yields {"source": tensor, "target": tensor})
trainer.fit(dataloader)

# Inference
translator = src.ImageTranslator(generator, device="cuda")
result = translator.predict(pil_image)
```

## Package Structure

```
src/
├── __init__.py          # Public API
├── models/
│   ├── generators.py    # UNetGenerator, ResNetGenerator
│   └── discriminators.py# PatchGANDiscriminator
├── data/
│   ├── datasets.py      # PairedImageDataset, UnpairedImageDataset
│   └── transforms.py    # get_transforms, default_transforms
├── losses/
│   ├── adversarial.py   # GANLoss
│   └── perceptual.py    # PerceptualLoss
├── training/
│   └── trainer.py       # Pix2PixTrainer, TrainingConfig
├── inference/
│   └── predictor.py     # ImageTranslator
└── metrics/
    └── image_quality.py # PSNR, SSIM, LPIPS, FID
```

## License

MIT
