# pix2pix (Community)

**Paper:** [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) (Isola et al., CVPR 2017)

**Source:** [junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

pix2pix learns paired image-to-image translation with a U-Net generator and PatchGAN discriminator. Training objective: GAN loss + λ·L1.

## Training

```python
from examples.pix2pix import Pix2PixTrainer, TrainingConfig
from src.models.cyclegan_pix2pix import create_generator, create_discriminator

generator = create_generator(netG="unet_256", norm="batch", use_dropout=True)
discriminator = create_discriminator(input_nc=6, norm="batch")

cfg = TrainingConfig(lambda_l1=100.0, save_dir="./checkpoints/pix2pix/facades")
trainer = Pix2PixTrainer(generator, discriminator, cfg)
trainer.fit(dataloader)
```

Facades (upstream defaults):

```bash
python -c "
from examples.pix2pix import Pix2PixTrainer, TrainingConfig
from src.models.cyclegan_pix2pix import create_generator, create_discriminator
from torch.utils.data import DataLoader
from src.data.datasets import PairedImageDataset

gen = create_generator(netG='unet_256', norm='batch', use_dropout=True)
disc = create_discriminator(input_nc=6, norm='batch')
trainer = Pix2PixTrainer(gen, disc, TrainingConfig(save_dir='./checkpoints/pix2pix/facades'))
ds = PairedImageDataset('./datasets/facades/train')
trainer.fit(DataLoader(ds, batch_size=1, shuffle=True))
"
```

## Inference

```python
from examples.community.pix2pix import load_pix2pix_community_pipeline

# HF checkpoint from Pix2PixTrainer
pipe = load_pix2pix_community_pipeline(
    "./checkpoints/pix2pix/facades/checkpoint-epoch-200",
    device="cuda",
)
out = pipe(source_image=label_map, output_type="pil")

# Upstream pretrained (facades_label2photo_pretrained/latest_net_G.pth)
pipe = load_pix2pix_community_pipeline(
    "./checkpoints/facades_label2photo_pretrained",
    device="cuda",
)
```

## Citation

```bibtex
@inproceedings{isola2017image,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={CVPR},
  year={2017}
}
```
