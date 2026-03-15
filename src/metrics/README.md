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
    metrics=["psnr", "ssim", "lpips", "dists", "l1", "l2"],
    data_range=1.0,
    device="cuda",
)
generated = torch.rand(8, 3, 256, 256).cuda()   # (N, C, H, W) in [0, 1]
reference = torch.rand(8, 3, 256, 256).cuda()
scores = evaluator(generated, reference)
# → {"psnr": 28.5, "ssim": 0.92, "lpips": 0.15, "dists": 0.85, "l1": 0.04, "l2": 0.002}
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

### Conditional Diversity Metrics (VS, AFD)

Measure diversity among multiple outputs generated from each source image (conditional generation evaluation):

```python
from src.metrics import ConditionalDiversityMetricEvaluator

evaluator = ConditionalDiversityMetricEvaluator(
    metrics=["vs", "afd"],
    device="cuda",
)
# generated_groups: list of (L, C, H, W) — L samples per source, or (M, L, C, H, W) tensor
generated_groups = [torch.rand(5, 3, 256, 256).cuda() for _ in range(10)]  # 10 sources, 5 samples each
scores = evaluator(generated_groups)
# → {"vs": 3.2, "afd": 42.1}
```

- **VS (Vendi Score)** [Friedman & Dieng, TMLR 2023](https://github.com/vertaix/Vendi-Score): effective number of unique feature patterns (eigenvalue entropy). Higher = more diverse.
- **AFD (Average Feature Distance)** [Zhang et al., NeurIPS 2025](https://github.com/szhan311/ECSI): mean pairwise L2 distance in Inception feature space. Higher = more diverse.

### Available Metrics

```python
# Paired
print(PairedImageMetricEvaluator.available_metrics())
# ['psnr', 'ssim', 'lpips', 'dists', 'l1', 'l2', 'samscore']

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

## FID, KID, IS: Full Configuration

This section documents the complete configuration of our FID, KID, and IS implementations for reproducibility and comparison with papers.

### Backend

| Component | Value |
|-----------|-------|
| **Library** | `torchmetrics` (image submodule) |
| **Inception** | `torch-fidelity` (InceptionV3, pretrained on ImageNet) |
| **Install** | `pip install torchmetrics torch-fidelity` or `pip install -e ".[metrics]"` |

### Input Format

| Requirement | Specification |
|-------------|---------------|
| **Shape** | `(N, C, H, W)` with `C=3` (RGB) |
| **Value range** | `[0, 1]` (float) |
| **Normalization** | `normalize=True` — images in [0, 1] are cast internally for Inception |
| **Resize** | Images are resized to 299×299 inside the metric (Inception input size) |

Convert from `[-1, 1]` to `[0, 1]` before calling: `(x * 0.5 + 0.5).clamp(0, 1)`.

### FID (Fréchet Inception Distance)

| Parameter | Default | Description |
|-----------|---------|--------------|
| `feature_dim` | `2048` | InceptionV3 feature layer (64, 192, 768, or 2048) |
| `batch_size` | `64` | Max images per Inception forward pass (reduces GPU memory) |
| `normalize` | `True` | Input in [0, 1]; set by implementation |
| `feature_extractor_weights_path` | (torch-fidelity default) | Path to Inception weights for custom backbone |
| `device` | from input | Device for computation |

**Formula**: \( \text{FID} = \|\mu - \mu_w\|^2 + \text{tr}(\Sigma + \Sigma_w - 2(\Sigma \Sigma_w)^{1/2}) \) where \( \mathcal{N}(\mu, \Sigma) \) and \( \mathcal{N}(\mu_w, \Sigma_w) \) are Gaussians fit to real and fake Inception features.

**Interpretation**: Lower is better. Compares distributions of real vs. generated images.

**Override via evaluator**:
```python
evaluator = UnpairedImageMetricEvaluator(
    metrics=["fid"],
    feature_dim=2048,
    batch_size=32,  # smaller if OOM
)
```

### KID (Kernel Inception Distance)

| Parameter | Default | Description |
|-----------|---------|--------------|
| `feature_dim` | `2048` | InceptionV3 feature layer |
| `subsets` | `100` | Number of bootstrap subsets for mean/std |
| `subset_size` | `1000` | Samples per subset (clamped to min(n-1, subset_size)) |
| `batch_size` | `64` | Max images per Inception forward pass |
| `normalize` | `True` | Input in [0, 1] |
| `feature_extractor_weights_path` | (torch-fidelity default) | Path to Inception weights |

**Interpretation**: Lower is better. MMD with polynomial kernel between real and fake feature distributions.

**Override via evaluator**:
```python
evaluator = UnpairedImageMetricEvaluator(
    metrics=["kid"],
    subsets=100,
    subset_size=1000,
    batch_size=32,
)
```

### IS (Inception Score)

| Parameter | Default | Description |
|-----------|---------|--------------|
| `splits` | `10` | Number of splits for mean/std over the score |
| `batch_size` | `64` | Max images per Inception forward pass |
| `normalize` | `True` | Input in [0, 1] |
| `feature_extractor_weights_path` | (torch-fidelity default) | Path to Inception weights |

**Note**: Uses only `fake_images`; `real_images` are ignored.

**Interpretation**: Higher is better. Measures quality and diversity of generated images.

**Override via evaluator**:
```python
evaluator = UnpairedImageMetricEvaluator(
    metrics=["is"],
    splits=10,
    batch_size=32,
)
```

### Batched Processing

All three metrics process images in batches (default 64) to avoid CUDA OOM. The implementation calls `update()` incrementally; results are mathematically equivalent to processing the full set at once.

### Sample Size Sensitivity

**FID and KID are highly sensitive to the number of samples.** Absolute values are not comparable across different sample sizes:

- Fewer samples (e.g. 500–1000) → higher FID, noisier estimates
- Papers often use 2k–50k samples; always match sample count when comparing to reported numbers
- Different implementations (torchmetrics, clean-fid, TensorFlow) can yield 2–3× differences on the same data

For reproducible comparisons, document: backend, sample count, dataset split, and image resolution.

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
