# CDTSDE Community Pipeline

CDTSDE (ControlNet-based Domain Transfer SDE) for solar defect identification (PSCDE). Uses the original [projects/CDTSDE](https://github.com/...) ControlLDM with dynamic shift sampling.

## Checkpoint conversion

Convert raw `.ckpt` (PyTorch Lightning) to BiliSakura format for one-stop loading:

```bash
python -m examples.community.cdtsde.convert_ckpt_to_cdtsde \
  --ckpt /path/to/PSCDE.ckpt \
  --output-dir /path/to/CDTSDE-ckpt/solar-defect-pscde

# With model config for reproducibility
python -m examples.community.cdtsde.convert_ckpt_to_cdtsde \
  --ckpt /root/worksapce/models/raw/CDTSDE-ckpt-raw/PSCDE.ckpt \
  --output-dir /root/worksapce/models/BiliSakura/CDTSDE-ckpt/solar-defect-pscde \
  --model-config /root/worksapce/projects/CDTSDE/configs/model/cldm_v21_dynamic.yaml
```

## One-stop inference

Inference accepts **only** diffusers-style checkpoint (convert first):

```python
from examples.community.cdtsde import load_cdtsde_community_pipeline

pipe = load_cdtsde_community_pipeline(
    "/path/to/CDTSDE-ckpt/solar-defect-pscde",
    cdtsde_src_path="/path/to/projects/CDTSDE",
)
pipe.to("cuda")

# Run inference
from PIL import Image
img = Image.open("electroluminescence.png").convert("RGB")
out = pipe(
    source_image=img,
    num_inference_steps=50,
    positive_prompt="clean, high-resolution, 8k",
    output_type="pil",
)
out.images[0].save("semantic_mask.png")
```

## Requirements

- **CDTSDE source**: The [projects/CDTSDE](../../CDTSDE) directory must be available. The pipeline adds it to `sys.path` automatically when it is at `projects/CDTSDE`, `./CDTSDE`, or `./projects/CDTSDE`, or you can pass `cdtsde_src_path`.
- **Dependencies**: Install CDTSDE requirements (`omegaconf`, `pytorch_lightning`, `einops`, etc.) as needed by the source project.

## Checkpoint layout (standard diffusers)

After conversion:

```
solar-defect-pscde/
├── unet/
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors
├── controlnet/
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors
├── vae/
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors
├── text_encoder/
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors
├── cond_encoder/
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors
├── nonlinear_lambda/
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors
├── scheduler/
│   └── scheduler_config.json
└── model_config.json
```

Raw `.ckpt` must be converted first; the pipeline does not load raw checkpoints.
