# DDIB Community Pipeline

DDIB (Dual Diffusion Implicit Bridges) for OpenAI/guided_diffusion-style checkpoints. [Su et al., ICLR 2023](https://github.com/suxuann/ddib).

The standard `src` DDIB uses diffusers `UNet2DModel`, which does not match raw DDIB checkpoints. This community package provides `OpenAIDDIBUNet` and a conversion script.

## Supported Checkpoints

| Source | Layout | Notes |
|--------|--------|-------|
| BiliSakura/DDIB-ckpt | `source_unet/` + `target_unet/` | Pre-converted format, use `DDIBPipeline.from_pretrained` |
| Raw .pt (guided_diffusion) | Convert with script below | Architecture must match `OpenAIDDIBUNet` |

**Known limitations:** Raw DDIB checkpoints from the official repo (ImageNet256, Synthetic log2D) use architectures that differ from the default `OpenAIDDIBUNet` (different channel layouts, attention, 1D layers). The conversion script creates the correct folder layout; you may need to adjust the model config or use architecture-specific forks for full compatibility.

## Conversion (pt → source_unet/ + target_unet/)

```bash
# Synthetic log2D0 → log2D1
python -m examples.community.ddib.convert_pt_to_ddib \
    --source-pt models/raw/DDIB-ckpt-raw/Synthetic/log2D0/model230000.pt \
    --target-pt models/raw/DDIB-ckpt-raw/Synthetic/log2D1/ema_0.9999_200000.pt \
    --output-dir models/BiliSakura/DDIB-ckpt/Synthetic-log2D0-to-log2D1

# ImageNet256 (strip class embedding for unconditional use)
python -m examples.community.ddib.convert_pt_to_ddib \
    --source-pt models/raw/DDIB-ckpt-raw/ImageNet256/256x256_diffusion.pt \
    --target-pt models/raw/DDIB-ckpt-raw/ImageNet256/256x256_diffusion.pt \
    --output-dir models/BiliSakura/DDIB-ckpt/ImageNet256 \
    --strip-class-embed
```

Output: `source_unet/`, `target_unet/`, and optional `scheduler/`.

## Inference

**With community pipeline (OpenAI-style checkpoints):**

```python
from examples.community.ddib import load_ddib_community_pipeline

pipe = load_ddib_community_pipeline(
    "/path/to/DDIB-ckpt/Synthetic-log2D0-to-log2D1",
    device="cuda",
)
out = pipe(source_image=image, num_inference_steps=250, output_type="pil")
out.images[0].save("output.png")
```

**With core pipeline (diffusers-style checkpoints, e.g. from HuggingFace):**

```python
from src.pipelines import DDIBPipeline

pipe = DDIBPipeline.from_pretrained(
    "path/to/DDIB-ckpt/dataset-name",
    source_subfolder="source_unet",
    target_subfolder="target_unet",
)
pipe.to("cuda")
out = pipe(source_image=image, num_inference_steps=250, output_type="pil")
```

## Citation

```bibtex
@inproceedings{su2023ddib,
  title={Dual Diffusion Implicit Bridges for Image-to-Image Translation},
  author={Su, Xuan and Song, Jiaming and Meng, Chenlin and Ermon, Stefano},
  booktitle={ICLR},
  year={2023}
}
```
