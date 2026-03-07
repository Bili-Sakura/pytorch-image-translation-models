# E3Diff

**Paper:** *Efficient End-to-End Diffusion Model for One-step SAR-to-Optical Translation* (Qin et al., IEEE GRSL 2024)

**Original:** [https://github.com/DeepSARRS/E3Diff](https://github.com/DeepSARRS/E3Diff)

## Architecture

E3Diff is a two-stage training approach for SAR-to-Optical image translation:

- **Stage 1** (`E3DiffTrainer` with `stage=1`): A conditional U-Net (E3DiffUNet) learns to denoise optical images conditioned on SAR inputs, trained with L1 noise-prediction loss and optional Focal Frequency Loss. The conditioning signal is processed by CPEN (Conditional Prior Enhancement Network), a multi-scale encoder that produces five feature maps injected at each U-Net resolution.
- **Stage 2** (`E3DiffTrainer` with `stage=2`): The pre-trained diffusion model is fine-tuned for one-step DDIM inference with an adversarial PatchGAN objective, enabling fast single-forward-pass translation.

> **Note on SoftPool:** The original E3Diff uses `SoftPool` (an external CUDA extension) inside CPEN. This community pipeline uses `AvgPool2d` instead so that it remains fully self-contained with no external build dependencies.

## Files

| File | Description |
|------|-------------|
| `model.py` | Network architectures: `E3DiffUNet`, `CPEN`, `GaussianDiffusion`, `FocalFrequencyLoss`, `_NLayerDiscriminator`, `_GANLoss` |
| `pipeline.py` | Configuration (`E3DiffConfig`) and trainer (`E3DiffTrainer`) |

## Quick Start

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

# Inference (DDIM, typically 50 steps for Stage-1 / 1 step for Stage-2 models)
with torch.no_grad():
    optical_pred = trainer2.sample(sar_batch, n_ddim_steps=1)
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

@article{qin2024conditional,
  title={Conditional Diffusion Model with Spatial-Frequency Refinement for
         SAR-to-Optical Image Translation},
  author={Qin, Jiang and Wang, Kai and Zou, Bin and Zhang, Lamei
          and van de Weijer, Joost},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2024},
  publisher={IEEE}
}
```
