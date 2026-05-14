# EGSDE (community)

**Paper:** [EGSDE: Unpaired Image-to-Image Translation via Energy-Guided Stochastic Differential Equations](https://arxiv.org/abs/2207.06635) (Zhao et al., NeurIPS 2022)

**Upstream:** [Bili-Sakura/EGSDE-diffusers](https://github.com/Bili-Sakura/EGSDE-diffusers) (diffusers-oriented fork of [ML-GSAI/EGSDE](https://github.com/ML-GSAI/EGSDE))

This folder wraps the upstream VP-EGSDE implementation so you can run inference from **pytorch-image-translation-models** with the same `DiffusionPipeline`-style API used elsewhere in the repo.

## Setup

1. Clone the upstream repository next to this project (or anywhere on disk):

   ```bash
   git clone https://github.com/Bili-Sakura/EGSDE-diffusers.git
   ```

2. Download pretrained diffusion checkpoints and domain-specific extractors (DSE) as described in the [upstream README](https://github.com/Bili-Sakura/EGSDE-diffusers/blob/master/README.md) into `EGSDE-diffusers/pretrained_model/`.

## Inference

```python
from examples.community.egsde import load_egsde_community_pipeline
from PIL import Image

pipe = load_egsde_community_pipeline(
    "/path/to/EGSDE-diffusers",
    task="cat2dog",
    device="cuda",
)
pipe.to("cuda")

src = Image.open("cat.png").convert("RGB")
out = pipe(source_image=src, output_type="pil")
out.images[0].save("dog_like.png")
```

### Custom checkpoints (no profile)

```python
pipe = load_egsde_community_pipeline(
    "/path/to/EGSDE-diffusers",
    task=None,
    ckpt="/path/to/score_model.pt",
    dsepath="/path/to/dse.pt",
    config_path="/path/to/config.yml",
    diffusionmodel="ADM",  # or "DDPM"
    device="cuda",
)
```

## Citation

```bibtex
@article{zhao2022egsde,
  title={Egsde: Unpaired image-to-image translation via energy-guided stochastic differential equations},
  author={Zhao, Min and Bao, Fan and Li, Chongxuan and Zhu, Jun},
  journal={arXiv preprint arXiv:2207.06635},
  year={2022}
}
```
