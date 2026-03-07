# E3Diff

**Paper:** *Efficient End-to-End Diffusion Model for One-step SAR-to-Optical Translation* (Qin et al., IEEE GRSL 2024)

## Architecture

Two-stage training for SAR-to-Optical image translation:
- **Stage 1** (`E3DiffTrainer` with `stage=1`): A conditional U-Net (E3DiffUNet) learns to denoise optical images conditioned on SAR inputs, trained with L1 noise-prediction loss and optional Focal Frequency Loss. The conditioning signal is processed by CPEN (Conditional Prior Enhancement Network), a multi-scale encoder that produces five feature maps injected at each U-Net resolution.
- **Stage 2** (`E3DiffTrainer` with `stage=2`): The pre-trained diffusion model is fine-tuned for one-step DDIM inference with an adversarial PatchGAN objective, enabling fast single-forward-pass translation.

> **Note on SoftPool:** The original E3Diff uses `SoftPool` (an external CUDA extension) inside CPEN. This community pipeline uses `AvgPool2d` instead so that it remains fully self-contained with no external build dependencies.

## Module layout

| File | Contents |
|------|----------|
| `model.py` | `E3DiffUNet`, `CPEN`, `FocalFrequencyLoss`, `_NLayerDiscriminator`, `_GANLoss`, building blocks |
| `pipeline.py` | `E3DiffPipeline` (inherits `DiffusionPipeline`), `E3DiffPipelineOutput`, `GaussianDiffusion`, beta schedules |
| `train.py` | `E3DiffConfig`, `E3DiffTrainer` |

## Quick start

### Inference (pipeline)

```python
import torch
from examples.community.e3diff import (
    E3DiffUNet, GaussianDiffusion, E3DiffPipeline,
)

unet = E3DiffUNet(out_channel=3, inner_channel=64, condition_ch=3, image_size=256)
diff = GaussianDiffusion(denoise_fn=unet, image_size=256, channels=3)
diff.set_noise_schedule(n_timestep=1000, schedule="linear", device="cuda")
# diff.load_state_dict(torch.load("diffusion_checkpoint.pth"))

pipeline = E3DiffPipeline(diffusion=diff)
output = pipeline(source_image=sar_tensor, num_inference_steps=50, output_type="pil")
images = output.images  # list of PIL images
```

### Training

```python
import torch
from examples.community.e3diff import E3DiffConfig, E3DiffTrainer

# Stage 1: Train diffusion model
cfg = E3DiffConfig(stage=1, condition_ch=3, out_ch=3, device="cuda")
trainer = E3DiffTrainer(cfg)
losses = trainer.train_step(sar_batch, optical_batch)
# losses = {'l_pix': ...}

# Stage 2: Fine-tune for one-step inference with GAN loss
cfg2 = E3DiffConfig(stage=2, condition_ch=3, out_ch=3, lambda_gan=0.1, device="cuda")
trainer2 = E3DiffTrainer(cfg2)
losses2 = trainer2.train_step(sar_batch, optical_batch)
# losses2 = {'l_pix': ..., 'l_G': ..., 'l_D': ...}
```

## Citation

```bibtex
@ARTICLE{10767752,
  author={Qin, Jiang and Zou, Bin and Li, Haolin and Zhang, Lamei},
  journal={IEEE Geoscience and Remote Sensing Letters},
  title={Efficient End-to-End Diffusion Model for One-step SAR-to-Optical Translation},
  year={2024},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/LGRS.2024.3506566}}
```
