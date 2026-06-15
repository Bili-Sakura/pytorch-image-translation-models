# Hneg-SRC (Semantic Relation Contrastive)

**Paper:** [Exploring Patch-wise Semantic Relation for Contrastive Learning in Image-to-Image Translation Tasks](https://arxiv.org/abs/2203.01532) (Jung, Kwon & Ye, CVPR 2022)

**Source:** [jcy132/Hneg_SRC](https://github.com/jcy132/Hneg_SRC)

Hneg-SRC extends [CUT](https://github.com/taesungp/contrastive-unpaired-translation) with two complementary losses:

- **SRC** — aligns patch-wise semantic relation matrices between source and translated features (JSD objective).
- **HDCE** — hard-negative patch contrastive loss weighted by SRC-derived semantic relations.

Training uses the CUT ResNet generator and PatchGAN discriminator from `src.models.cut`. Inference is a single generator forward pass via :class:`~src.pipelines.cut.HnegSRCPipeline`.

## Training

```python
from examples.hneg_src import HnegSRCConfig, HnegSRCTrainer

cfg = HnegSRCConfig(
    lambda_HDCE=0.1,
    lambda_SRC=0.05,
    use_curriculum=True,
    hdce_gamma=50.0,
    hdce_gamma_min=10.0,
    dce_idt=True,
    save_dir="./checkpoints/hneg_src/h2z",
)
trainer = HnegSRCTrainer(cfg)
trainer.train("data/domain_a/train", "data/domain_b/train")
```

Recommended settings from the upstream repo (horse→zebra):

```bash
python -c "
from examples.hneg_src import HnegSRCConfig, HnegSRCTrainer
cfg = HnegSRCConfig(
    lambda_HDCE=0.1, lambda_SRC=0.05, dce_idt=True,
    use_curriculum=True, hdce_gamma=50, hdce_gamma_min=10,
    save_dir='./checkpoints/hneg_src/h2z',
)
HnegSRCTrainer(cfg).train('./datasets/horse2zebra/trainA', './datasets/horse2zebra/trainB')
"
```

For Cityscapes (`BtoA`), add `step_gamma=True` and `step_gamma_epoch=200`.

## Inference

```python
from examples.community.hneg_src import load_hneg_src_pipeline

pipe = load_hneg_src_pipeline(
    "./checkpoints/hneg_src/h2z/checkpoint-epoch-200",
    device="cuda",
)
out = pipe(source_image=image, output_type="pil")
out.images[0].save("translated.png")
```

Or load directly from `src`:

```python
from src.pipelines.cut import HnegSRCPipeline

pipe = HnegSRCPipeline.from_pretrained("./checkpoints/hneg_src/h2z/checkpoint-epoch-200")
pipe.to("cuda")
```

## Citation

```bibtex
@inproceedings{jung2022exploring,
  title={Exploring patch-wise semantic relation for contrastive learning in image-to-image translation tasks},
  author={Jung, Chanyong and Kwon, Gihyun and Ye, Jong Chul},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={18260--18269},
  year={2022}
}
```

## Acknowledgements

Built on [CUT](https://github.com/taesungp/contrastive-unpaired-translation) (Park et al., ECCV 2020). The upstream repository also includes multimodal and compression variants; this integration covers the single-modal SRC+HDCE training path used in the main paper experiments.
