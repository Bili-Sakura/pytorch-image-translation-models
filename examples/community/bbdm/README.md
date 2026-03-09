# BBDM Community Pipeline

Community support for original BBDM checkpoints (xuekt98/BBDM) that store
OpenAI-style UNet weights (`input_blocks.*`, `middle_block.*`, `output_blocks.*`).

## Conversion

Convert raw checkpoints into this repo's checkpoint layout:

```bash
python -m examples.community.bbdm.convert_ckpt_to_unet \
  --raw-root "/root/worksapce/models/raw/BBDM Checkpoints" \
  --output-root "/root/worksapce/models/BiliSakura/BBDM-ckpt"
```

Convert legacy VQGAN checkpoints (for BBDM latent/tokenizer usage) into Diffusers
`VQModel` format:

```bash
python -m examples.community.bbdm.convert_ckpt_to_vqmodel \
  --raw-dir "/root/worksapce/models/raw/BBDM Checkpoints/vq-f4.zip" \
  --raw-dir "/root/worksapce/models/raw/BBDM Checkpoints/vq-f8.zip" \
  --raw-dir "/root/worksapce/models/raw/BBDM Checkpoints/vq-f16.zip" \
  --output-root "/root/worksapce/models/BiliSakura/BBDM-ckpt" \
  --subfolder "vqvae"
```

or with extracted raw folders:

```bash
python -m examples.community.bbdm.convert_ckpt_to_vqmodel \
  --raw-dir "/root/worksapce/models/raw/BBDM Checkpoints/vqgan_f4" \
  --raw-dir "/root/worksapce/models/raw/BBDM Checkpoints/vqgan_f8" \
  --raw-dir "/root/worksapce/models/raw/BBDM Checkpoints/vqgan_f16" \
  --output-root "/root/worksapce/models/BiliSakura/BBDM-ckpt" \
  --subfolder "vqvae"
```

Each converted model folder contains:

- `unet/config.json`
- `unet/diffusion_pytorch_model.safetensors` (if weights are present)
- `scheduler/scheduler_config.json`
- `conversion_status.json`

If no `.pt/.pth/.ckpt` file exists in a source folder, the converter writes
scaffold configs and marks status as `pending_weights`.

## Inference

```python
from examples.community.bbdm import load_bbdm_community_pipeline

pipe = load_bbdm_community_pipeline(
    "/root/worksapce/models/BiliSakura/BBDM-ckpt/edges2shoes",
    device="cuda",
)
out = pipe(source_image=latent_or_image_tensor, num_inference_steps=200, output_type="pt")
```

## Notes

- Converted OpenAI-style checkpoints are generally **not** compatible with
  `src.BBDMPipeline.from_pretrained(...)` (diffusers UNet key format mismatch).
- Use `load_bbdm_community_pipeline(...)` for these checkpoints.
