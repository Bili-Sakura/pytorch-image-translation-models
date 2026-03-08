# SAR2Optical (Community Pipeline)

This community integration ports the core pix2pix cGAN workflow from the external
project [yuuIind/SAR2Optical](https://github.com/yuuIind/SAR2Optical) into
`examples/community/sar2optical/`.

## Included components

- `model.py`
  - `SAR2OpticalGenerator` (U-Net generator with skip-connections)
  - `SAR2OpticalDiscriminator` (PatchGAN/PixelGAN)
- `pipeline.py`
  - `SAR2OpticalPipeline` (DiffusionPipeline-style single-pass inference)
  - `load_sar2optical_pipeline` (checkpoint loader)
- `train.py`
  - `SAR2OpticalConfig`
  - `SAR2OpticalTrainer` (step-wise training/validation utilities)

## Quick start

```python
import torch
from examples.community.sar2optical import (
    SAR2OpticalGenerator,
    SAR2OpticalPipeline,
    SAR2OpticalConfig,
    SAR2OpticalTrainer,
)

# Inference
gen = SAR2OpticalGenerator(c_in=3, c_out=3)
pipe = SAR2OpticalPipeline(generator=gen)
sar = torch.randn(1, 3, 256, 256)
out = pipe(source_image=sar, output_type="pt")

# Training step
cfg = SAR2OpticalConfig(c_in=3, c_out=3, device="cpu")
trainer = SAR2OpticalTrainer(cfg)
losses = trainer.train_step(real_images=sar, target_images=torch.randn(1, 3, 256, 256))
```

## Citation

If you use this integration, please cite:

- SAR2Optical repository: <https://github.com/yuuIind/SAR2Optical>
- pix2pix: Isola et al., CVPR 2017.
