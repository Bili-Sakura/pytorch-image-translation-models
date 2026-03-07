# Parallel-GAN

**Paper:** *SAR-to-Optical Image Translation with Hierarchical Latent Features* (Wang et al., IEEE TGRS 2022)

**Original:** [https://github.com/ZZG-Z/Parallel-GAN](https://github.com/ZZG-Z/Parallel-GAN)

## Architecture

The Parallel-GAN uses a two-stage training approach:

- **Stage 1** (`Resrecon`): A ResNet-50 encoder + transposed-conv decoder trained to reconstruct optical images from optical images, learning a rich feature hierarchy.
- **Stage 2** (`ParaGAN`): A ResNet-50 encoder (accepts SAR input) with residual bottleneck blocks and a decoder that fuses encoder skip connections (like U-Net) plus a *hierarchical feature loss* that aligns intermediate features with the pre-trained reconstruction network.

## Files

| File | Description |
|------|-------------|
| `model.py` | Network architectures: `ParaGAN`, `Resrecon`, `_NLayerDiscriminator`, `VGGLoss`, `_GANLoss` |
| `pipeline.py` | Configuration (`ParallelGANConfig`) and trainer (`ParallelGANTrainer`) |

## Quick Start

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
