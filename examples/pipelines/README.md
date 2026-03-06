# Production-Ready Custom Pipelines

Self-contained diffusion pipelines for image translation models. **No external project code required** — use these for inference without the full `src/` tree.

## Models

| Model | Pipeline | UNet | Scheduler |
|-------|----------|------|-----------|
| DDBM | `DDBMPipeline` | `DDBMUNet` | `DDBMScheduler` |
| DDIB | `DDIBPipeline` | `DDIBUNet` | `DDIBScheduler` |
| I2SB | `I2SBPipeline` | `I2SBUNet` | `I2SBScheduler` |
| BiBBDM | `BiBBDMPipeline` | `BiBBDMUNet` | `BiBBDMScheduler` |
| BDBM | `BDBMPipeline` | `BDBMUNet` | `BDBMScheduler` |
| DBIM | `DBIMPipeline` | `DBIMUNet` | `DBIMScheduler` |
| CDTSDE | `CDTSDEPipeline` | `CDTSDEUNet` | `CDTSDEScheduler` |
| LBM | `LBMPipeline` | `LBMUNet` | `LBMScheduler` |

## Usage

### 1. Direct import (recommended for deployment)

```python
from examples.pipelines.ddbm.pipeline import DDBMPipeline, DDBMUNet, DDBMScheduler

ckpt_path = "./checkpoints/ddbm/sar2eo/checkpoint-10000"

# Load components (prefer ema_unet when available)
unet = DDBMUNet.from_pretrained(ckpt_path, subfolder="ema_unet")
scheduler = DDBMScheduler.from_pretrained(ckpt_path, subfolder="scheduler")
pipeline = DDBMPipeline(unet=unet, scheduler=scheduler)
pipeline = pipeline.to("cuda")

# Inference: source_image in [-1, 1], shape (B, C, H, W)
result = pipeline(
    source_image=source_tensor,
    num_inference_steps=1000,
    guidance=1.0,
    churn_step_ratio=0.33,
    output_type="pt",  # or "pil", "np"
)
images = result.images
```

### 2. As `custom_pipeline` for diffusers

```python
import torch
from diffusers import DiffusionPipeline

# Pass the pipeline directory so diffusers loads the custom pipeline class
pipeline = DiffusionPipeline.from_pretrained(
    "path/to/checkpoint",
    custom_pipeline="examples/pipelines/ddbm",
    torch_dtype=torch.float16,
    device_map="cuda",
)
```

For checkpoints with `model_index.json` pointing to `src.models` / `src.schedulers`, use the **direct import** pattern (1) — it does not rely on `model_index.json` and works with any checkpoint layout.

### 3. Standalone inference script

```bash
python -m examples.pipelines.run_inference \
    --model ddbm \
    --checkpoint ./checkpoints/ddbm/sar2eo/checkpoint-10000 \
    --input_dir ./path/to/images \
    --output_dir ./outputs \
    --num_steps 1000
```

## Checkpoint layout

DDBM checkpoint (typical):

```
checkpoint-10000/
├── model_index.json
├── unet/
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors
├── ema_unet/          # optional, prefer for inference
│   └── diffusion_pytorch_model.safetensors
└── scheduler/
    └── scheduler_config.json
```

DDIB (combined pipeline):

```
pipeline/
├── model_index.json
├── source_unet/   (or source_ema_unet)
├── target_unet/   (or target_ema_unet)
└── scheduler/
```

## Dependencies

- `torch`
- `diffusers`
- `Pillow`
- `numpy`
- `tqdm`

No imports from `src/` — pipelines are fully self-contained.
