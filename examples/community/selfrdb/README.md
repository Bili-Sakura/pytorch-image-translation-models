# SelfRDB (Community)

**Paper:** *Self-Consistent Recursive Diffusion Bridge for Medical Image Translation* (Arslan et al., Medical Image Analysis 2024)

**Source:** [icon-lab/SelfRDB](https://github.com/icon-lab/SelfRDB)

SelfRDB employs a diffusion bridge with soft-prior noise scheduling and self-consistent recursion for multi-modal medical image synthesis (T1↔T2, PD↔T1, FLAIR↔T2, CT, etc.).

## Quick start

```python
from examples.community.selfrdb import load_selfrdb_community_pipeline

pipe = load_selfrdb_community_pipeline(
    "/path/to/ixi_t1_t2.ckpt",
    device="cuda",
)
out = pipe(source_image=source_tensor, output_type="pil")
```

## Checkpoint format

Use `.ckpt` files from the [SelfRDB Model Zoo](https://github.com/icon-lab/SelfRDB#-model-zoo):

| Dataset | Task | Checkpoint |
|---------|------|------------|
| IXI | T2→T1 | [ixi_t2_t1.ckpt](https://github.com/icon-lab/SelfRDB/releases/download/v1.0.0/ixi_t2_t1.ckpt) |
| IXI | T1→T2 | [ixi_t1_t2.ckpt](https://github.com/icon-lab/SelfRDB/releases/download/v1.0.0/ixi_t1_t2.ckpt) |
| IXI | PD→T1 | [ixi_pd_t1.ckpt](https://github.com/icon-lab/SelfRDB/releases/download/v1.0.0/ixi_pd_t1.ckpt) |
| IXI | T1→PD | [ixi_t1_pd.ckpt](https://github.com/icon-lab/SelfRDB/releases/download/v1.0.0/ixi_t1_pd.ckpt) |
| BRATS | T2→T1, T1→T2, FLAIR↔T2 | See [releases](https://github.com/icon-lab/SelfRDB/releases) |
| CT | T2→CT, T1→CT | See [releases](https://github.com/icon-lab/SelfRDB/releases) |

## Input format

- Source images: `[0, 1]` range, single-channel (grayscale) or multi-channel (mean will be used).
- Shape: `(B, 1, H, W)` or PIL/numpy with `H×W` or `H×W×1`.

## Citation

```bibtex
@article{arslan2024selfconsistent,
  title={Self-Consistent Recursive Diffusion Bridge for Medical Image Translation},
  author={Fuat Arslan and Bilal Kabas and Onat Dalmaz and Muzaffer Ozbey and Tolga Çukur},
  year={2024},
  journal={arXiv:2405.06789}
}
```
