# Hugging Face Storage Buckets

[Hugging Face Storage Buckets](https://huggingface.co/blog/storage-buckets) provide S3-like object storage for ML artifacts—checkpoints, TensorBoard logs, and other mutable intermediate files—without version control. They use [Xet](https://huggingface.co/docs/hub/storage-backends) for content-addressable deduplication, so related artifacts (e.g., successive checkpoints) share storage efficiently.

This package integrates buckets into the CUT and Pix2Pix tutorial trainers so you can sync checkpoints and TensorBoard logs during training.

---

## Requirements

- **huggingface_hub** == 1.7.1 (or ≥ 1.5.0 for basic bucket support)
- **hf-xet** == 1.4.2 (for HF_BUCKET content-addressable storage)
- **HF_TOKEN** set (or logged in via `huggingface-cli login`)

```bash
pip install huggingface_hub==1.7.1 hf-xet==1.4.2
# or
pip install -e ".[training]"
```

---

## API (`src.utils.training_utils`)

| Function | Description |
|----------|-------------|
| `push_checkpoint_to_bucket(folder_path, bucket_id, path_in_bucket, token=None)` | Sync a checkpoint directory to a bucket. Uses `sync_bucket` for incremental uploads with Xet deduplication. |
| `sync_tensorboard_to_bucket(tensorboard_dir, bucket_id, path_in_bucket, token=None)` | Sync TensorBoard event files to a bucket. |

Both functions read `HF_TOKEN` from the environment when `token` is not provided.

---

## Tutorial Usage

The **CUT** and **Pix2Pix** trainers accept:

| Argument | Default | Description |
|----------|---------|-------------|
| `--push_to_bucket` | `false` | Whether to sync checkpoints and TensorBoard to a bucket |
| `--hf_bucket` | — | Bucket ID (e.g. `username/my-training-bucket`) |

### CUT (unpaired)

```bash
# Create bucket via CLI (optional; created automatically if missing)
hf buckets create my-training-bucket --exist-ok

# Run with bucket sync
PUSH_TO_BUCKET=true HF_BUCKET=username/my-bucket bash tutorial/cut/train_facades.sh

# Or pass directly
python tutorial/cut/trainer.py \
  --dataset facades \
  --output_dir ./ckpt/facades \
  --push_to_bucket true \
  --hf_bucket username/my-bucket
```

### Pix2Pix (paired)

```bash
PUSH_TO_BUCKET=true HF_BUCKET=username/my-bucket bash tutorial/pix2pix/train_facades.sh
```

### Run-all (CUT)

```bash
PUSH_TO_BUCKET=true HF_BUCKET=username/my-bucket bash tutorial/cut/run_all.sh
```

---

## Bucket Layout

Artifacts are stored under method- and dataset-specific prefixes:

| Method | Checkpoints | TensorBoard |
|--------|-------------|-------------|
| **CUT** | `cut/{dataset}/checkpoint-{step}` or `checkpoint-epoch-{epoch}` | `cut/{dataset}/tensorboard` |
| **Pix2Pix** | `pix2pix/{dataset}/checkpoint-epoch-{N}`, `pix2pix/{dataset}/latest` | `pix2pix/{dataset}/tensorboard` |

Browse at `https://huggingface.co/buckets/username/my-bucket`, or use the handle `hf://buckets/username/my-bucket` in code.

---

## Python Usage (standalone)

```python
from src.utils.training_utils import push_checkpoint_to_bucket, sync_tensorboard_to_bucket

# Sync a checkpoint directory
push_checkpoint_to_bucket(
    folder_path="./ckpt/facades/checkpoint-epoch-50",
    bucket_id="username/my-bucket",
    path_in_bucket="cut/facades/checkpoint-50",
)

# Sync TensorBoard logs
sync_tensorboard_to_bucket(
    tensorboard_dir="./ckpt/facades/tensorboard",
    bucket_id="username/my-bucket",
    path_in_bucket="cut/facades/tensorboard",
)
```

---

## CLI Alternative

You can also sync manually with the `hf` CLI:

```bash
hf buckets sync ./ckpt/facades hf://buckets/username/my-bucket/cut/facades
```

See the [Hugging Face Buckets guide](https://huggingface.co/docs/huggingface_hub/en/guides/buckets) for more options.
