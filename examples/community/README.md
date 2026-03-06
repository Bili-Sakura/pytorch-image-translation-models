# Community Pipelines

Community pipelines are **self-contained, single-file modules** contributed by the community.  They follow the same pattern as [Hugging Face diffusers community pipelines](https://github.com/huggingface/diffusers/tree/main/examples/community): each file bundles *all* model, loss, and utility code needed for training or inference so that it works without importing any other project code from `src/`.

## How to use

```python
# Import directly from the single file
from examples.community.parallel_gan import ParaGAN, Resrecon, ParallelGANTrainer, ParallelGANConfig
```

## How to contribute

1. Create a single Python file under `examples/community/` named after the model (e.g. `my_model.py`).
2. Include all network definitions, loss helpers, and a training/inference class in that one file.
3. Add a module docstring describing the paper, usage, and citation.
4. Add an entry to this README.
5. Add tests in `tests/test_community_pipelines.py`.

---

## Available Pipelines

| Pipeline | Paper | Description |
|----------|-------|-------------|
| [`parallel_gan.py`](parallel_gan.py) | [Wang et al., TGRS 2022](https://ieeexplore.ieee.org/document/9864654) | SAR-to-Optical translation with hierarchical latent features via a two-stage approach (reconstruction + translation) |

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
