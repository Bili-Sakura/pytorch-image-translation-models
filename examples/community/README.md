# Community Pipelines

Community pipelines are **self-contained modules** contributed by the community.  Each pipeline lives in its own subfolder under `examples/community/<model_name>/` and includes:

| File | Purpose |
|------|---------|
| `model.py` | All network architecture definitions and loss helpers |
| `pipeline.py` | Configuration dataclass and training / inference pipeline |
| `readme.md` | Documentation, usage examples, and citation |
| `__init__.py` | Re-exports public symbols for convenient imports |

Pipelines bundle *all* model, loss, and utility code needed for training or inference so that they work without importing any other project code from `src/`.

## How to use

```python
# Import directly from the community pipeline package
from examples.community.parallel_gan import ParaGAN, Resrecon, ParallelGANTrainer, ParallelGANConfig
```

## How to contribute

1. Create a new subfolder under `examples/community/` named after the model (e.g. `my_model/`).
2. Add `model.py` with all network definitions and loss helpers.
3. Add `pipeline.py` with a configuration dataclass and a training / inference class.
4. Add `readme.md` describing the paper, usage, and citation.
5. Add `__init__.py` that re-exports public symbols.
6. Add an entry to this README.
7. Add tests in `tests/test_community_pipelines.py`.

---

## Available Pipelines

| Pipeline | Paper | Description |
|----------|-------|-------------|
| [`parallel_gan/`](parallel_gan/) | [Wang et al., TGRS 2022](https://ieeexplore.ieee.org/document/9864654) | SAR-to-Optical translation with hierarchical latent features via a two-stage approach (reconstruction + translation) |
| [`e3diff/`](e3diff/) | [Qin et al., IEEE GRSL 2024](https://ieeexplore.ieee.org/document/10767752) | Efficient End-to-End Diffusion Model for one-step SAR-to-Optical translation using a conditional U-Net (CPEN) and two-stage diffusion + GAN training |

---

### Parallel-GAN

**Paper:** *SAR-to-Optical Image Translation with Hierarchical Latent Features* (Wang et al., IEEE TGRS 2022)

**Architecture:** Two-stage training:
- **Stage 1** (`Resrecon`): ResNet-50 encoder + transposed-conv decoder learns to reconstruct optical → optical.
- **Stage 2** (`ParaGAN`): ResNet-50 encoder (accepts SAR) + decoder with encoder skip connections + hierarchical feature loss aligning with the frozen Stage 1 network.

**Quick start:**

```python
from examples.community.parallel_gan import (
    ParaGAN,
    Resrecon,
    ParallelGANTrainer,
    ParallelGANConfig,
)

# Stage 1: Train reconstruction network
cfg = ParallelGANConfig(input_nc=3, output_nc=3, device="cuda")
trainer = ParallelGANTrainer(cfg)  # No recon_net → Stage 1
losses = trainer.train_step_recon(optical_batch)

# Stage 2: Train translation network with pre-trained recon net
recon_net = Resrecon()
recon_net.load_state_dict(torch.load("recon_checkpoint.pth"))
trainer = ParallelGANTrainer(cfg, recon_net=recon_net)
losses = trainer.train_step_trans(sar_batch, optical_batch)
```

**Citation:**

```bibtex
@ARTICLE{9864654,
  author={Wang, Haixia and Zhang, Zhigang and Hu, Zhanyi and Dong, Qiulei},
  journal={IEEE Trans. Geoscience and Remote Sensing},
  title={SAR-to-Optical Image Translation with Hierarchical Latent Features},
  year={2022},
  volume={60},
  pages={1-12},
  doi={10.1109/TGRS.2022.3200996}}
```

---

### E3Diff

**Paper:** *Efficient End-to-End Diffusion Model for One-step SAR-to-Optical Translation* (Qin et al., IEEE GRSL 2024)

**Architecture:** Two-stage training for SAR-to-Optical image translation:
- **Stage 1** (`E3DiffTrainer` with `stage=1`): A conditional U-Net (E3DiffUNet) learns to denoise optical images conditioned on SAR inputs, trained with L1 noise-prediction loss and optional Focal Frequency Loss. The conditioning signal is processed by CPEN (Conditional Prior Enhancement Network), a multi-scale encoder that produces five feature maps injected at each U-Net resolution.
- **Stage 2** (`E3DiffTrainer` with `stage=2`): The pre-trained diffusion model is fine-tuned for one-step DDIM inference with an adversarial PatchGAN objective, enabling fast single-forward-pass translation.

> **Note on SoftPool:** The original E3Diff uses `SoftPool` (an external CUDA extension) inside CPEN. This community pipeline uses `AvgPool2d` instead so that it remains fully self-contained with no external build dependencies.

**Quick start:**

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

**Citation:**

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
