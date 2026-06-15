# NEGCUT (Hard Negative Contrastive Unpaired Translation)

**Paper:** [Instance-wise Hard Negative Example Generation for Contrastive Learning in Unpaired Image-to-Image Translation](https://github.com/WeilunWang/NEGCUT) (Wang et al., ICCV 2021)

**Source:** [WeilunWang/NEGCUT](https://github.com/WeilunWang/NEGCUT)

NEGCUT extends [CUT](https://github.com/taesungp/contrastive-unpaired-translation) by training a negative generator adversarially to produce hard contrastive negatives online, improving unpaired translation quality at the same inference cost as CUT.

Training uses the CUT ResNet generator and PatchGAN discriminator from `src.models.cut`, plus:

- **LearnedPatchNCELoss** — PatchNCE with generator-produced hard negatives.
- **NegativeGenerator** — MLP-based negative sampler (`neg_gen_momentum` by default).
- **MS diversity loss** — encourages diverse negative patches (`lambda_MS_neg`).

Inference is a single generator forward pass via :class:`~src.pipelines.cut.NEGCUTPipeline`.

## Training

```python
from examples.negcut import NEGCUTConfig, NEGCUTTrainer

cfg = NEGCUTConfig(
    netN="neg_gen_momentum",
    lambda_NCE=1.0,
    lambda_MS_neg=1.0,
    nce_idt=True,
    save_dir="./checkpoints/negcut/h2z",
)
trainer = NEGCUTTrainer(cfg)
trainer.train("data/domain_a/train", "data/domain_b/train")
```

Recommended settings from the upstream repo (horse→zebra / cityscapes):

```bash
python -c "
from examples.negcut import NEGCUTConfig, NEGCUTTrainer
cfg = NEGCUTConfig(
    netN='neg_gen_momentum',
    lambda_NCE=1.0,
    lambda_MS_neg=1.0,
    nce_idt=True,
    save_dir='./checkpoints/negcut/h2z',
)
NEGCUTTrainer(cfg).train('./datasets/horse2zebra/trainA', './datasets/horse2zebra/trainB')
"
```

## Inference

```python
from examples.community.negcut import load_negcut_pipeline

pipe = load_negcut_pipeline(
    "./checkpoints/negcut/h2z/checkpoint-epoch-200",
    device="cuda",
)
out = pipe(source_image=image, output_type="pil")
out.images[0].save("translated.png")
```

Or load directly from `src`:

```python
from src.pipelines.cut import NEGCUTPipeline

pipe = NEGCUTPipeline.from_pretrained("./checkpoints/negcut/h2z/checkpoint-epoch-200")
pipe.to("cuda")
```

## Citation

```bibtex
@inproceedings{wang2021negcut,
  title={Instance-wise Hard Negative Example Generation for Contrastive Learning in Unpaired Image-to-Image Translation},
  author={Wang, Weilun and others},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2021}
}
```

## Acknowledgements

Built on [CUT](https://github.com/taesungp/contrastive-unpaired-translation) (Park et al., ECCV 2020). This integration covers the main NEGCUT training path with adversarial hard-negative generation.
