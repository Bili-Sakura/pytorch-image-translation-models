# F-LSeSim (Spatially-Correlative Loss)

**Paper:** [The Spatially-Correlative Loss for Various Image Translation Tasks](https://arxiv.org/abs/2104.00209) (Zheng, Cham & Cai, CVPR 2021)

**Source:** [lyndonzheng/F-LSeSim](https://github.com/lyndonzheng/F-LSeSim)

F-LSeSim adds a structure-preserving **spatially-correlative loss** to one-sided unpaired translation. The loss compares patch self-similarity maps between source and translated images using VGG16 features, preserving spatial relationships regardless of absolute pixel values.

Training uses a ResNet generator and PatchGAN discriminator from `src.models.cut`. Inference is a single generator forward pass via :class:`~src.pipelines.cut.FLSeSimPipeline`.

## Training

```python
from examples.flsesim import FLSeSimConfig, FLSeSimTrainer

cfg = FLSeSimConfig(
    lambda_spatial=10.0,
    attn_layers="4,7,9",
    patch_size=64,
    loss_mode="cos",
    gan_mode="lsgan",
    save_dir="./checkpoints/flsesim/h2z",
)
trainer = FLSeSimTrainer(cfg)
trainer.train("data/domain_a/train", "data/domain_b/train")
```

Recommended settings from the upstream repo (horse→zebra):

```bash
python -c "
from examples.flsesim import FLSeSimConfig, FLSeSimTrainer
cfg = FLSeSimConfig(
    lambda_spatial=10.0, attn_layers='4,7,9', patch_size=64,
    loss_mode='cos', gan_mode='lsgan',
    save_dir='./checkpoints/flsesim/h2z',
)
FLSeSimTrainer(cfg).train('./datasets/horse2zebra/trainA', './datasets/horse2zebra/trainB')
"
```

Optional flags from upstream:

- `use_norm=True` — cosine similarity map (default uses dot-based attention)
- `learned_attn=True` — trainable 1×1 conv filters and contrastive spatial loss
- `lambda_spatial_idt>0` / `lambda_identity>0` — identity mapping losses (requires matching channel counts)

> **Note:** The upstream single-image StyleGAN2 variant (`SinSCModel`) is not included here; this integration covers the single-modal ResNet translation path used in the main paper experiments.

## Inference

```python
from examples.community.flsesim import load_flsesim_pipeline

pipe = load_flsesim_pipeline(
    "./checkpoints/flsesim/h2z/checkpoint-epoch-200",
    device="cuda",
)
out = pipe(source_image=image, output_type="pil")
out.images[0].save("translated.png")
```

Or load directly from `src`:

```python
from src.pipelines.cut import FLSeSimPipeline

pipe = FLSeSimPipeline.from_pretrained("./checkpoints/flsesim/h2z/checkpoint-epoch-200")
pipe.to("cuda")
```

## Citation

```bibtex
@inproceedings{zheng2021spatiallycorrelative,
  title={The Spatially-Correlative Loss for Various Image Translation Tasks},
  author={Zheng, Chuanxia and Cham, Tat-Jen and Cai, Jianfei},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2021}
}
```

## Acknowledgements

Built on [CUT](https://github.com/taesungp/contrastive-unpaired-translation) and [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) network architectures from the upstream F-LSeSim repository.
