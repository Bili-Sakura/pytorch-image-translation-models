# CycleGAN (Community)

**Paper:** [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) (Zhu et al., ICCV 2017)

**Source:** [junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

CycleGAN learns unpaired domain translation with two generators (G_A: A→B, G_B: B→A) and cycle-consistency losses. This integration uses ResNet-9blocks generators and PatchGAN discriminators matching the upstream repository.

## Training

```python
from examples.cyclegan import CycleGANConfig, CycleGANTrainer

cfg = CycleGANConfig(
    lambda_A=10.0,
    lambda_B=10.0,
    lambda_identity=0.5,
    gan_mode="lsgan",
    save_dir="./checkpoints/cyclegan/maps",
)
trainer = CycleGANTrainer(cfg)
trainer.train("./datasets/maps/trainA", "./datasets/maps/trainB")
```

Horse→zebra (upstream defaults):

```bash
python -c "
from examples.cyclegan import CycleGANConfig, CycleGANTrainer
cfg = CycleGANConfig(save_dir='./checkpoints/cyclegan/h2z')
CycleGANTrainer(cfg).train('./datasets/horse2zebra/trainA', './datasets/horse2zebra/trainB')
"
```

## Inference

```python
from examples.community.cyclegan import load_cyclegan_community_pipeline

# HF checkpoint from CycleGANTrainer
pipe = load_cyclegan_community_pipeline(
    "./checkpoints/cyclegan/h2z/checkpoint-epoch-200",
    device="cuda",
)
out = pipe(source_image=image, direction="a2b", output_type="pil")

# Upstream pretrained (horse2zebra_pretrained/latest_net_G.pth)
pipe = load_cyclegan_community_pipeline(
    "./checkpoints/horse2zebra_pretrained",
    direction="a2b",
    device="cuda",
)
```

## Citation

```bibtex
@inproceedings{CycleGAN2017,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={ICCV},
  year={2017}
}
```
