# Image Quality Metrics

One-stop evaluation for image translation quality. Metrics are split into **paired** (reference-based) and **unpaired** (distribution-based).

## Installation

Install metrics extras:

```bash
pip install -e ".[metrics]"
# or: pip install torchmetrics lpips torch-fidelity
```

## One-Stop Usage

### Paired Metrics (reference required)

Compare generated images to reference images (e.g. image translation, restoration):

```python
import torch
from src.metrics import PairedImageMetricEvaluator

# One-stop: pick metrics, run once
evaluator = PairedImageMetricEvaluator(
    metrics=["psnr", "ssim", "lpips", "dists"],
    data_range=1.0,
    device="cuda",
)
generated = torch.rand(8, 3, 256, 256).cuda()   # (N, C, H, W) in [0, 1]
reference = torch.rand(8, 3, 256, 256).cuda()
scores = evaluator(generated, reference)
# → {"psnr": 28.5, "ssim": 0.92, "lpips": 0.15, "dists": 0.85}
```

### Unpaired Metrics (distribution-based)

Compare distributions of real vs. generated images (e.g. unconditional generation):

```python
from src.metrics import UnpairedImageMetricEvaluator

evaluator = UnpairedImageMetricEvaluator(
    metrics=["fid", "kid", "is"],
    device="cuda",
)
real_images = torch.rand(100, 3, 64, 64).cuda()
fake_images = torch.rand(100, 3, 64, 64).cuda()
scores = evaluator(real_images, fake_images)
# → {"fid": 12.3, "kid": 0.02, "is": 8.5}
```

### Available Metrics

```python
# Paired
print(PairedImageMetricEvaluator.available_metrics())
# ['psnr', 'ssim', 'lpips', 'dists', 'samscore']

# Unpaired
print(UnpairedImageMetricEvaluator.available_metrics())
# ['fid', 'kid', 'is', 'sfd', 'sfid', 'cmmd', 'fwd', 'ifid', 'precision', 'recall', 'pr_f1']
```

---

## Custom Models via HuggingFace / Local Checkpoints

Metrics that use pretrained backbones accept **local paths** or **HuggingFace model IDs**. Pass them as kwargs to the evaluator; they are forwarded to each metric.

### CMMD (CLIP backbone)

Uses `model_id` for CLIP. Accepts both HuggingFace IDs and local paths:

```python
evaluator = UnpairedImageMetricEvaluator(
    metrics=["cmmd"],
    model_id="/path/to/local/clip",  # or "openai/clip-vit-base-patch32"
)
scores = evaluator(real_images, fake_images)
```

**Local checkpoint layout**: a folder with `config.json` and `pytorch_model.bin` (or `model.safetensors`), as from `transformers`:

```bash
/path/to/local/clip/
├── config.json
├── preprocessor_config.json
└── pytorch_model.bin
```

### DISTS (VGG + learned α/β)

Uses `weights_path` for the learned α/β. Default loads from HuggingFace `chaofengc/IQA-PyTorch-Weights`:

```python
evaluator = PairedImageMetricEvaluator(
    metrics=["dists"],
    weights_path="/path/to/DISTS_weights.pt",
)
scores = evaluator(generated, reference)
```

**Local weights file**: a `.pt` file with keys `alpha` and `beta` (tensors).

### SAMScore (SAM backbone)

Uses `model_weight_path` via the `samscore` package. Pass through evaluator kwargs:

```python
evaluator = PairedImageMetricEvaluator(
    metrics=["samscore"],
    model_weight_path="/path/to/sam_vit_b_01ec64.pth",
    model_type="vit_b",
)
scores = evaluator(generated, reference)
```

**Local checkpoint**: a `.pth` file for the SAM image encoder (e.g. `sam_vit_b_01ec64.pth`).

### LPIPS (Alex/VGG/Squeeze backbones)

Pass `model_path` to use custom weights (via evaluator or direct call):

```python
evaluator = PairedImageMetricEvaluator(
    metrics=["lpips"],
    model_path="/path/to/lpips_alex.pth",
    net="alex",
)
# or directly:
from src.metrics.paired.lpips_impl import compute_lpips
score = compute_lpips(pred, target, model_path="/path/to/lpips_alex.pth", net="alex")
```

### FID / KID / IS (Inception backbone)

These use `torchmetrics` and `torch-fidelity`, which support `feature_extractor_weights_path`:

```python
evaluator = UnpairedImageMetricEvaluator(
    metrics=["fid"],
    feature_extractor_weights_path="/path/to/inception-v3-compat.pth",
)
```

---

## Quick Reference: Custom Checkpoint kwargs

| Metric   | Kwarg                        | Accepts                                              |
|----------|------------------------------|------------------------------------------------------|
| CMMD     | `model_id`                   | HuggingFace ID or local path                         |
| DISTS    | `weights_path`               | Path to `.pt` with α/β                               |
| SAMScore | `model_weight_path`          | Path to SAM `.pth` encoder                           |
| LPIPS    | `model_path`                 | Path to LPIPS `.pth` (use `compute_lpips` directly) |
| FID/KID/IS | `feature_extractor_weights_path` | Path to Inception weights                        |
| iFID      | `vae_path`                       | Path to diffusers AutoencoderKL (e.g. `stabilityai/sd-vae-ft-ema`) |

### iFID (VAE required — no fallback)

iFID **requires** a VAE. Pass ``vae_path`` or ``vae``; there is no fallback to FID:

```python
evaluator = UnpairedImageMetricEvaluator(
    metrics=["ifid"],
    vae_path="stabilityai/sd-vae-ft-ema",
)
# Or use another diffusers VAE: AutoencoderTiny, AsymmetricAutoencoderKL, etc.
evaluator = UnpairedImageMetricEvaluator(
    metrics=["ifid"],
    vae_path="madebyollin/taesd",
    vae_cls="AutoencoderTiny",
)
```

### Required backends (no fallback)

| Metric | Backend | Install |
|--------|---------|---------|
| **sFID** | pyiqa | `pip install pyiqa` |
| **FWD**  | pytorchfwd | `pip install pytorchfwd` |

---

## Image Conventions

- **Shape**: `(N, C, H, W)` with `C=3` (RGB).
- **Value range**: `[0, 1]` for paired metrics. Set `data_range=2.0` if inputs are in `[-1, 1]`.
- **Device**: Tensors are moved to the evaluator `device` automatically.
