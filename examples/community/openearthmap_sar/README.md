# OpenEarthMap-SAR

CUT (Contrastive Unpaired Translation) models for SAR ↔ optical image translation. Compatible with [pytorch-image-translation-models](https://github.com/Bili-Sakura/pytorch-image-translation-models).

## Models

| Model | Description |
|-------|-------------|
| `opt2sar` | Optical → SAR |
| `sar2opt` | SAR → Optical |
| `seman2opt` | Semantic → Optical |
| `seman2opt_pesudo` | Semantic (pseudo) → Optical |
| `seman2sar` | Semantic → SAR |
| `seman2sar_pesudo` | Semantic (pseudo) → SAR |

## Usage

```python
from examples.community.openearthmap_sar import load_openearthmap_sar_pipeline

pipeline = load_openearthmap_sar_pipeline(
    checkpoint_dir="/path/to/CUT-OpenEarthMap-SAR",
    model_name="sar2opt",
    device="cuda",
)
output = pipeline(source_image=pil_image, output_type="pil")
```

CLI (from project root, `conda activate rsgen`):

```bash
python -m examples.community.openearthmap_sar --checkpoint-dir /path/to/CUT-OpenEarthMap-SAR --model sar2opt --input sar.png --output out.png
```

**Note:** Pass `source_image` as `PIL.Image`. The generator uses anti-aliased down/upsampling to match the original CUT training.

## Architecture

- `in_channels`: 3
- `out_channels`: 3
- `base_filters`: 64
- `n_blocks`: 9
- `norm`: instance
