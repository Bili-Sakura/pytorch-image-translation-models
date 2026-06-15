# Decent (Density Changing Regularization)

**Paper:** [Unsupervised Image-to-Image Translation with Density Changing Regularization](https://github.com/Mid-Push/Decent) (Xie, Ho & Zhang, NeurIPS 2022)

**Source:** [Mid-Push/Decent](https://github.com/Mid-Push/Decent)

Decent performs single-direction unpaired translation (A→B) using a CUT ResNet generator with:

- **Density estimators** — per-domain normalizing flows (BNAF, MAF, NSF, or MAF-MoG) model patch feature densities at multiple encoder layers.
- **Density-changing loss** — penalizes variance in log-density changes between source and translated patches, encouraging content preservation without paired data.
- **Identity loss** — standard L1 identity mapping on the target domain.

Inference is a single generator forward pass via :class:`~src.pipelines.cut.DecentPipeline`.

## Training

```python
from examples.decent import DecentConfig, DecentTrainer

cfg = DecentConfig(
    lambda_var=0.01,
    lambda_idt=10.0,
    flow_type="bnaf",
    flow_blocks=1,
    flow_lr=1e-3,
    flow_ema=0.998,
    var_all=False,
    save_dir="./checkpoints/decent/selfie2anime",
)
trainer = DecentTrainer(cfg)
trainer.train("data/domain_a/train", "data/domain_b/train")
```

Recommended settings from the upstream repo (selfie→anime):

```bash
python -c "
from examples.decent import DecentConfig, DecentTrainer
cfg = DecentConfig(
    lambda_var=0.01, flow_type='bnaf', flow_blocks=1,
    flow_lr=1e-3, flow_ema=0.998, var_all=False,
    save_dir='./checkpoints/decent/selfie2anime',
)
DecentTrainer(cfg).train('./datasets/selfie2anime/trainA', './datasets/selfie2anime/trainB')
"
```

For NSF flows, install the optional dependency: `pip install nflows`.

## Inference

```python
from examples.community.decent import load_decent_pipeline

pipe = load_decent_pipeline(
    "./checkpoints/decent/selfie2anime/checkpoint-epoch-200",
    device="cuda",
)
out = pipe(source_image=image, output_type="pil")
out.images[0].save("translated.png")
```

Or load directly from `src`:

```python
from src.pipelines.cut import DecentPipeline

pipe = DecentPipeline.from_pretrained("./checkpoints/decent/selfie2anime/checkpoint-epoch-200")
pipe.to("cuda")
```

## Citation

```bibtex
@inproceedings{xieunsupervised,
  title={Unsupervised Image-to-Image Translation with Density Changing Regularization},
  author={Xie, Shaoan and Ho, Qirong and Zhang, Kun},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}
```

## Acknowledgements

Built on [CUT](https://github.com/taesungp/contrastive-unpaired-translation) (Park et al., ECCV 2020). The upstream repository also includes CUT and CycleGAN variants; this integration covers the Decent single-direction density-changing training path used in the main paper experiments.
