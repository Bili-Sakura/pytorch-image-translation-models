# Parallel-GAN

**Paper:** *SAR-to-Optical Image Translation with Hierarchical Latent Features* (Wang et al., IEEE TGRS 2022)

## Architecture

Two-stage training:
- **Stage 1** (`Resrecon`): ResNet-50 encoder + transposed-conv decoder learns to reconstruct optical → optical.
- **Stage 2** (`ParaGAN`): ResNet-50 encoder (accepts SAR) + decoder with encoder skip connections + hierarchical feature loss aligning with the frozen Stage 1 network.

## Module layout

| File | Contents |
|------|----------|
| `model.py` | `ParaGAN`, `Resrecon`, `_NLayerDiscriminator`, `VGGLoss`, `_GANLoss`, `_init_weights` |
| `pipeline.py` | `translate()` – inference helper |
| `train.py` | `ParallelGANConfig`, `ParallelGANTrainer` |

## Quick start

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

## Citation

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
