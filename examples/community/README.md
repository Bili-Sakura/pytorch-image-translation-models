# Community Pipelines

Community pipelines are **self-contained modules** contributed by the community. Each pipeline lives in its own subfolder under `examples/community/<model_name>/` and contains:

| File | Purpose |
|------|---------|
| `model.py` | Network architectures, losses, and utility code |
| `pipeline.py` | Inference / pipeline logic |
| `train.py` | Training configuration and harness |
| `README.md` | Usage examples and citation information |

## How to use

```python
# Import from the package
from examples.community.parallel_gan import ParaGAN, Resrecon, ParallelGANTrainer, ParallelGANConfig
```

## How to contribute

1. Create a subfolder under `examples/community/` named after the model (e.g. `my_model/`).
2. Add `model.py`, `pipeline.py`, `train.py`, and `README.md`.
3. Create an `__init__.py` that re-exports all public symbols.
4. Add an entry to this README.
5. Add tests in `tests/test_community_pipelines.py`.

---

## Available Pipelines

| Pipeline | Paper | Description |
|----------|-------|-------------|
| [`ddbm/`](ddbm/) | [Zhou et al., ICLR 2024](https://arxiv.org/abs/2309.16948) | DDBM for OpenAI-style checkpoints (BiliSakura/DDBM-ckpt); uses improved_diffusion architecture |
| [`bbdm/`](bbdm/) | [Li et al., CVPR 2023](https://arxiv.org/abs/2205.07680) | BBDM for original xuekt98/BBDM-style checkpoints with OpenAI UNet key layout |
| [`parallel_gan/`](parallel_gan/) | [Wang et al., TGRS 2022](https://ieeexplore.ieee.org/document/9864654) | SAR-to-Optical translation with hierarchical latent features via a two-stage approach (reconstruction + translation) |
| [`e3diff/`](e3diff/) | [Qin et al., IEEE GRSL 2024](https://ieeexplore.ieee.org/document/10767752) | Efficient End-to-End Diffusion Model for one-step SAR-to-Optical translation using a conditional U-Net (CPEN) and two-stage diffusion + GAN training |
| [`openearthmap_sar/`](openearthmap_sar/) | [Park et al., ECCV 2020](https://arxiv.org/abs/2007.15651) | CUT models for SAR ↔ optical image translation with anti-aliased ResNet generator (opt2sar, sar2opt, seman2opt, seman2sar, etc.) |
| [`sar2optical/`](sar2optical/) | [Isola et al., CVPR 2017](https://arxiv.org/abs/1611.07004) | Pix2Pix cGAN SAR-to-Optical translation, adapted from yuuIind/SAR2Optical |
| [`mdt/`](mdt/) | [Bosch Research, NeurIPS 2025](https://github.com/boschresearch/Multimodal-Distribution-Translation-MDT) | LDDBM: Latent diffusion bridge for super-resolution and multi-view→3D |
| [`ddib/`](ddib/) | [Su et al., ICLR 2023](https://github.com/suxuann/ddib) | DDIB for OpenAI/guided_diffusion-style checkpoints; dual source/target UNets |
| [`cdtsde/`](cdtsde/) | CDTSDE/PSCDE | ControlLDM for solar defect identification; convert raw .ckpt and one-stop inference |
| [`diffuseit/`](diffuseit/) | [Kwon & Ye, ICLR 2023](https://arxiv.org/abs/2209.15264) | Diffusion-based image translation with disentangled style/content (text- and image-guided) |

---

### DDBM (Community)

**Paper:** *Denoising Diffusion Bridge Models* (Zhou et al., ICLR 2024)

**Architecture:** OpenAI/improved_diffusion-style U-Net (input_blocks, middle_block, output_blocks). Compatible with [BiliSakura/DDBM-ckpt](https://huggingface.co/BiliSakura/DDBM-ckpt). The standard `src` DDBM uses diffusers UNet2DModel and does not match this architecture.

**Quick start:**

```python
from examples.community.ddbm import load_ddbm_community_pipeline

pipe = load_ddbm_community_pipeline(
    "/path/to/DDBM-ckpt/edges2handbags-vp",
    device="cuda",
)
out = pipe(source_image=image, num_inference_steps=40, output_type="pil")
```

**Converting from raw .pt:** `python -m examples.community.ddbm.convert_pt_to_unet /path/to/model_dir --checkpoint ckpt.pt`

---

### BBDM (Community)

**Paper:** *BBDM: Image-to-Image Translation with Brownian Bridge Diffusion Models* (Li et al., CVPR 2023)

**Architecture:** OpenAI/improved_diffusion-style U-Net key layout used by xuekt98/BBDM checkpoints. The standard `src` BBDM wrapper uses diffusers `UNet2DModel`, so raw BBDM checkpoints typically need this community loader.

**Quick start:**

```python
from examples.community.bbdm import load_bbdm_community_pipeline

pipe = load_bbdm_community_pipeline(
    "/path/to/BBDM-ckpt/edges2shoes",
    device="cuda",
)
out = pipe(source_image=image, num_inference_steps=200, output_type="pil")
```

**Converting from raw checkpoints:**

```bash
python -m examples.community.bbdm.convert_ckpt_to_unet \
  --raw-root "/path/to/raw/BBDM Checkpoints" \
  --output-root "/path/to/BBDM-ckpt"
```

---

### DDIB (Community)

**Paper:** *Dual Diffusion Implicit Bridges for Image-to-Image Translation* (Su et al., ICLR 2023)

**Architecture:** OpenAI/guided_diffusion-style unconditional U-Net. Uses two models (source_unet, target_unet) for encode (source→latent) and decode (latent→target) via DDIM. Compatible with raw DDIB checkpoints after conversion.

**Quick start:**

```python
from examples.community.ddib import load_ddib_community_pipeline

pipe = load_ddib_community_pipeline(
    "/path/to/DDIB-ckpt/Synthetic-log2D0-to-log2D1",
    device="cuda",
)
out = pipe(source_image=image, num_inference_steps=250, output_type="pil")
```

**Converting from raw .pt:** See [ddib/README.md](ddib/README.md). Raw ImageNet256 and Synthetic log2D checkpoints may require architecture-specific config adjustments.

---

### DiffuseIT (Community)

**Paper:** *Diffusion-based Image Translation using Disentangled Style and Content Representation* (Kwon & Ye, ICLR 2023)

**Architecture:** Guided diffusion (OpenAI-style) with CLIP text guidance and VIT content/style losses. Supports text-guided and image-guided translation. Requires DiffuseIT repo cloned (for CLIP, model_vit, etc.).

**Converting from raw .pt:**

```bash
python -m examples.community.diffuseit.convert_ckpt_to_diffuseit \
  --raw-root /path/to/DiffuseIT-ckpt-raw \
  --output-root /path/to/BiliSakura/DiffuseIT-ckpt
```

**Quick start:**

```python
from examples.community.diffuseit import load_diffuseit_community_pipeline

pipe = load_diffuseit_community_pipeline(
    "/path/to/DiffuseIT-ckpt/imagenet256-uncond",
    diffuseit_src_path="projects/DiffuseIT",
)
pipe.to("cuda")
# Text-guided
out = pipe(source_image=img, prompt="Black Leopard", source="Lion", output_type="pil")
# Image-guided
out = pipe(source_image=img, target_image=style_ref, use_colormatch=True, output_type="pil")
```

---

### Parallel-GAN

**Paper:** *SAR-to-Optical Image Translation with Hierarchical Latent Features* (Wang et al., IEEE TGRS 2022)

**Architecture:** Two-stage training:
- **Stage 1** (`Resrecon`): ResNet-50 encoder + transposed-conv decoder learns to reconstruct optical → optical.
- **Stage 2** (`ParaGAN`): ResNet-50 encoder (accepts SAR) + decoder with encoder skip connections + hierarchical feature loss aligning with the frozen Stage 1 network.

**Quick start:**

```python
from examples.community.parallel_gan import ParaGAN, ParallelGANPipeline

# Inference via DiffusionPipeline
gen = ParaGAN(input_nc=3, output_nc=3)
pipeline = ParallelGANPipeline(generator=gen)
output = pipeline(source_image=sar_tensor, output_type="pil")

# Training
from examples.community.parallel_gan import ParallelGANTrainer, ParallelGANConfig, Resrecon

cfg = ParallelGANConfig(input_nc=3, output_nc=3, device="cuda")
trainer = ParallelGANTrainer(cfg)  # No recon_net → Stage 1
losses = trainer.train_step_recon(optical_batch)

recon_net = Resrecon()
recon_net.load_state_dict(torch.load("recon_checkpoint.pth"))
trainer = ParallelGANTrainer(cfg, recon_net=recon_net)
losses = trainer.train_step_trans(sar_batch, optical_batch)
```

**Citation:**

```bibtex
@ARTICLE{9864654,
  author={Wang, Haixia and Zhang, Zhigang and Hu, Zhanyi and Dong, Qiulei},
  journal={IEEE Trans. Geoscience and Remote Sensing},
  title={SAR-to-Optical Image Translation with Hierarchical Latent Features},
  year={2022},
  volume={60},
  pages={1-12},
  doi={10.1109/TGRS.2022.3200996}}
```

---

### OpenEarthMap-SAR

**Paper:** *Contrastive Learning for Unpaired Image-to-Image Translation* (Park et al., ECCV 2020)

**Architecture:** CUT ResNet generator with anti-aliased down/upsampling, compatible with OpenEarthMap-SAR checkpoints. Supports opt2sar, sar2opt, seman2opt, seman2sar and pseudo variants.

**Quick start:**

```python
from examples.community.openearthmap_sar import load_openearthmap_sar_pipeline

# SAR → Optical (or opt2sar, seman2opt, etc.)
pipeline = load_openearthmap_sar_pipeline(
    checkpoint_dir="/path/to/OpenEarthMap-SAR",
    model_name="sar2opt",
    device="cuda",
)
output = pipeline(source_image=pil_image, output_type="pil")
```

CLI: `python -m examples.community.openearthmap_sar --checkpoint-dir /path/to/CUT-OpenEarthMap-SAR --input sar.png --output out.png`

---

### E3Diff

**Paper:** *Efficient End-to-End Diffusion Model for One-step SAR-to-Optical Translation* (Qin et al., IEEE GRSL 2024)

**Architecture:** Two-stage training for SAR-to-Optical image translation:
- **Stage 1** (`E3DiffTrainer` with `stage=1`): A conditional U-Net (E3DiffUNet) learns to denoise optical images conditioned on SAR inputs, trained with L1 noise-prediction loss and optional Focal Frequency Loss. The conditioning signal is processed by CPEN (Conditional Prior Enhancement Network), a multi-scale encoder that produces five feature maps injected at each U-Net resolution.
- **Stage 2** (`E3DiffTrainer` with `stage=2`): The pre-trained diffusion model is fine-tuned for one-step DDIM inference with an adversarial PatchGAN objective, enabling fast single-forward-pass translation.

> **Note on SoftPool:** The original E3Diff uses `SoftPool` (an external CUDA extension) inside CPEN. This community pipeline uses `AvgPool2d` instead so that it remains fully self-contained with no external build dependencies.

**Quick start:**

```python
import torch
from examples.community.e3diff import (
    E3DiffUNet, GaussianDiffusion, E3DiffPipeline, E3DiffConfig, E3DiffTrainer,
)

# Inference via DiffusionPipeline
unet = E3DiffUNet(out_channel=3, inner_channel=64, condition_ch=3, image_size=256)
diff = GaussianDiffusion(denoise_fn=unet, image_size=256, channels=3)
diff.set_noise_schedule(n_timestep=1000, schedule="linear", device="cuda")

pipeline = E3DiffPipeline(diffusion=diff)
output = pipeline(source_image=sar_tensor, num_inference_steps=50, output_type="pil")

# Training
cfg = E3DiffConfig(stage=1, condition_ch=3, out_ch=3, device="cuda")
trainer = E3DiffTrainer(cfg)
losses = trainer.train_step(sar_batch, optical_batch)
```

**Citation:**

```bibtex
@ARTICLE{10767752,
  author={Qin, Jiang and Zou, Bin and Li, Haolin and Zhang, Lamei},
  journal={IEEE Geoscience and Remote Sensing Letters},
  title={Efficient End-to-End Diffusion Model for One-step SAR-to-Optical Translation},
  year={2024},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/LGRS.2024.3506566}}
```

---

### CDTSDE (Community)

**Source:** [projects/CDTSDE](../../CDTSDE) — ControlLDM for solar defect identification (PSCDE).

**Architecture:** Latent diffusion with ControlNet conditioning, CLIP text encoder, and dynamic shift sampling. Converts to standard diffusers layout (`unet/`, `controlnet/`, `vae/`, `text_encoder/`, etc.).

**Converting from raw .ckpt:**

```bash
python -m examples.community.cdtsde.convert_ckpt_to_cdtsde \
  --ckpt /path/to/PSCDE.ckpt \
  --output-dir /path/to/CDTSDE-ckpt/solar-defect-pscde
```

**Quick start** (diffusers-style checkpoint only; convert first):

```python
from examples.community.cdtsde import load_cdtsde_community_pipeline

pipe = load_cdtsde_community_pipeline(
    "/path/to/CDTSDE-ckpt/solar-defect-pscde",
    cdtsde_src_path="/path/to/projects/CDTSDE",
)
pipe.to("cuda")
out = pipe(source_image=image, num_inference_steps=50, output_type="pil")
out.images[0].save("semantic_mask.png")
```

---

### SAR2Optical

**Paper:** *Image-to-Image Translation with Conditional Adversarial Networks* (Isola et al., CVPR 2017)

**Source project:** [yuuIind/SAR2Optical](https://github.com/yuuIind/SAR2Optical)

**Architecture:** Pix2Pix-style conditional GAN:
- **Generator**: U-Net encoder-decoder with skip-connections.
- **Discriminator**: PatchGAN (or PixelGAN mode).
- **Objective**: adversarial BCE + weighted L1 reconstruction.

**Quick start:**

```python
import torch
from examples.community.sar2optical import SAR2OpticalGenerator, SAR2OpticalPipeline

gen = SAR2OpticalGenerator(c_in=3, c_out=3)
pipeline = SAR2OpticalPipeline(generator=gen)
out = pipeline(source_image=torch.randn(1, 3, 256, 256), output_type="pt")
```

---

### MDT / LDDBM

**Paper:** *Towards General Modality Translation with Contrastive and Predictive Latent Diffusion Bridge* (Bosch Research, NeurIPS 2025 submission)

**Source:** [boschresearch/Multimodal-Distribution-Translation-MDT](https://github.com/boschresearch/Multimodal-Distribution-Translation-MDT)

**Architecture:** Latent diffusion bridge with KL-VAE encoders/decoders and a transformer-based bridge. Supports super-resolution (16→128) and multi-view to 3D.

**Prerequisites:** Install the MDT repo first: `git clone ... && pip install -e .`

**Checkpoint format:** Each component uses `config.json` + `diffusion_pytorch_model.safetensors` in subfolders (`encoder_x/`, `encoder_y/`, `decoder_x/`, `bridge/`). Convert raw .pt via `python -m examples.community.mdt.convert_pt_to_mdt /path/to/output --encoder-x ... --encoder-y ... --decoder-x ... --bridge ...`

**Quick start:**

```python
from examples.community.mdt import load_mdt_community_pipeline

pipe = load_mdt_community_pipeline(
    checkpoint_dir="/path/to/mdt-checkpoints",
    task="sr_16_to_128",
    device="cuda",
)
out = pipe(source_image=lr_image, num_inference_steps=40, output_type="pil")
```

