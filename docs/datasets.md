# Image-to-Image Translation Datasets

Commonly used image-to-image translation datasets from the pix2pix and CycleGAN works (UC Berkeley). These have become de facto benchmarks for paired and unpaired translation.

---

## Reference Papers

- **pix2pix (CVPR 2017):** [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) — [Project](https://phillipi.github.io/pix2pix) | [Code](https://github.com/phillipi/pix2pix)
- **CycleGAN (ICCV 2017):** [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) — [Project](https://junyanz.github.io/CycleGAN) | [Code](https://github.com/junyanz/CycleGAN) | [PyTorch (unified)](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

---

## All Datasets

| Dataset | Type | Description | Samples | Classes | Resolution | Size | Download |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **facades** | paired | CMP Facade: labels ↔ photos | 400 | 12 | 256×256 | 29M | [pix2pix](https://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.tar.gz) |
| **maps** | paired | Google Maps ↔ aerial imagery | 1,096 | — | 256×256 | 239M | [pix2pix](https://efrosgans.eecs.berkeley.edu/pix2pix/datasets/maps.tar.gz) |
| **cityscapes** | paired | Urban street segmentation | 2,975 | 30+ | 256×256 | 99M | [pix2pix](https://efrosgans.eecs.berkeley.edu/pix2pix/datasets/cityscapes.tar.gz) |
| **edges2shoes** | paired | Edge maps ↔ shoe photos | 50,025 | — | 256×256 | ~2G | [pix2pix](https://efrosgans.eecs.berkeley.edu/pix2pix/datasets/edges2shoes.tar.gz) |
| **edges2handbags** | paired | Edge maps ↔ handbag photos | 137,000 | — | 256×256 | ~8G | [pix2pix](https://efrosgans.eecs.berkeley.edu/pix2pix/datasets/edges2handbags.tar.gz) |
| **horse2zebra** | unpaired | Horses ↔ zebras (ImageNet) | 939 + 1,177 | 2 domains | 256×256 | 111M | [CycleGAN](http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/horse2zebra.zip) |
| **apple2orange** | unpaired | Apples ↔ oranges (ImageNet) | 996 + 1,020 | 2 domains | 256×256 | 75M | [CycleGAN](http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/apple2orange.zip) |
| **summer2winter_yosemite** | unpaired | Summer ↔ winter (Yosemite) | 1,273 + 854 | 2 domains | 256×256 | 126M | [CycleGAN](http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/summer2winter_yosemite.zip) |
| **maps** | unpaired | Google Maps ↔ aerial imagery | 1,096 | 2 domains | 256×256 | 1.4G | [CycleGAN](http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/maps.zip) |
| **facades** | unpaired | Labels ↔ photos | 400 | 12 | 256×256 | 34M | [CycleGAN](http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/facades.zip) |
| **cityscapes** | unpaired | Labels ↔ photos | 2,975 | 30+ | 256×256 | 58M | [CycleGAN](http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/cityscapes.zip) |
| **ae_photos** | unpaired | Artistic / architectural | — | 2 domains | 256×256 | 10M | [CycleGAN](http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/ae_photos.zip) |
| **grumpifycat** | unpaired | Cat faces ↔ Grumpy Cat | ~214 | 2 domains | 256×256 | 19M | [CycleGAN](http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/grumpifycat.zip) |
| **iphone2dslr_flower** | unpaired | iPhone ↔ DSLR (flowers) | 1,813 + 3,316 | 2 domains | 256×256 | 324M | [CycleGAN](http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/iphone2dslr_flower.zip) |
| **monet2photo** | unpaired | Monet paintings ↔ photos | 1,074 + 6,853 | 2 domains | 256×256 | 291M | [CycleGAN](http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/monet2photo.zip) |
| **vangogh2photo** | unpaired | Van Gogh ↔ photos | 401 + 6,853 | 2 domains | 256×256 | 292M | [CycleGAN](http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/vangogh2photo.zip) |
| **cezanne2photo** | unpaired | Cézanne ↔ photos | 584 + 6,853 | 2 domains | 256×256 | 267M | [CycleGAN](http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/cezanne2photo.zip) |
| **ukiyoe2photo** | unpaired | Ukiyo-e ↔ photos | 1,433 + 6,853 | 2 domains | 256×256 | 279M | [CycleGAN](http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/ukiyoe2photo.zip) |

**Full catalog:** [pix2pix](https://efrosgans.eecs.berkeley.edu/pix2pix/datasets/) | [CycleGAN](http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/)

**Also on Hugging Face:** [horse2zebra](https://huggingface.co/datasets/huggan/horse2zebra) | [apple2orange](https://huggingface.co/datasets/huggan/apple2orange)

---

## Usage Notes

- **Paired vs. unpaired:** pix2pix uses aligned `(input, target)` pairs; CycleGAN provides separate domain folders.
- **Format:** pix2pix datasets: `*.tar.gz`; CycleGAN datasets: `*.zip`.
- **Citation:** See [Credits](credits.md) when using these datasets.
