# DiffuseIT (Community)

**Paper:** *Diffusion-based Image Translation using Disentangled Style and Content Representation* (Kwon & Ye, ICLR 2023)

**Source:** [cyclomon/DiffuseIT](https://github.com/cyclomon/DiffuseIT)

## Converting from raw checkpoints

```bash
python -m examples.community.diffuseit.convert_ckpt_to_diffuseit \
  --raw-root /root/worksapce/models/raw/DiffuseIT-ckpt-raw \
  --output-root /root/worksapce/models/BiliSakura/DiffuseIT-ckpt
# Outputs: imagenet256-uncond/, ffhq-256/, id_model/
```

## Quick start

```python
from examples.community.diffuseit import load_diffuseit_community_pipeline

pipe = load_diffuseit_community_pipeline(
    "/root/worksapce/models/BiliSakura/DiffuseIT-ckpt/imagenet256-uncond",
    diffuseit_src_path="projects/DiffuseIT",
)
pipe.to("cuda")

# Text-guided
out = pipe(
    source_image=source,
    prompt="Black Leopard",
    source="Lion",
    use_range_restart=True,
    use_noise_aug_all=True,
    output_type="pil",
)

# Image-guided
out = pipe(
    source_image=source,
    target_image=style_ref,
    use_colormatch=True,
    output_type="pil",
)
```

## Citation

```bibtex
@inproceedings{kwon2023diffuseit,
  title={Diffusion-based Image Translation using Disentangled Style and Content Representation},
  author={Kwon, Gihyun and Ye, Jong Chul},
  booktitle={ICLR},
  year={2023},
  url={https://arxiv.org/abs/2209.15264}
}
```
