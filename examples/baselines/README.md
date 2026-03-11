# Baseline Pipelines

Baseline pipelines integrate published image translation methods as reference implementations. These are distinguished from community examples by being primary baselines for comparison.

## DiffuseIT

**Paper:** *Diffusion-based Image Translation using Disentangled Style and Content Representation* (Kwon & Ye, ICLR 2023)

**Source:** [cyclomon/DiffuseIT](https://github.com/cyclomon/DiffuseIT)

**Architecture:** Guided diffusion (OpenAI-style) with CLIP text guidance and VIT content/style losses. Supports text-guided and image-guided translation.

### Setup

1. Clone DiffuseIT and download checkpoints:

```bash
git clone https://github.com/cyclomon/DiffuseIT.git projects/DiffuseIT
cd projects/DiffuseIT
mkdir -p checkpoints
# Download 256x256 model: https://openaipublic.blob.core.windows.net/diffusion/march-2021/256x256_diffusion_uncond.pt
# Or FFHQ: https://github.com/cyclomon/DiffuseIT (see README for links)
```

2. Install DiffuseIT dependencies (from DiffuseIT root):

```bash
cd projects/DiffuseIT
pip install ftfy regex lpips kornia opencv-python color-matcher
pip install git+https://github.com/openai/CLIP.git
```

### Quick start

**Text-guided translation:**

```python
from examples.baselines.diffuseit import DiffuseITPipeline

pipe = DiffuseITPipeline.from_pretrained(
    "projects/DiffuseIT",  # path to DiffuseIT root (with checkpoints/)
    timestep_respacing="100",
    skip_timesteps=80,
)
pipe.to("cuda")

from PIL import Image
source = Image.open("lion.jpg").convert("RGB")
out = pipe(
    source_image=source,
    prompt="Black Leopard",
    source="Lion",
    use_range_restart=True,
    use_noise_aug_all=True,
    output_type="pil",
)
out.images[0].save("leopard.png")
```

**Image-guided translation:**

```python
out = pipe(
    source_image=source,
    target_image=Image.open("style_ref.jpg").convert("RGB"),
    use_colormatch=True,
    use_noise_aug_all=True,
    use_range_restart=True,
    output_type="pil",
)
```

### Citation

```bibtex
@inproceedings{kwon2023diffuseit,
  title={Diffusion-based Image Translation using Disentangled Style and Content Representation},
  author={Kwon, Gihyun and Ye, Jong Chul},
  booktitle={ICLR},
  year={2023},
  url={https://arxiv.org/abs/2209.15264}
}
```
