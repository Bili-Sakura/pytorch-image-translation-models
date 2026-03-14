# AlignFlow (Community)

**Paper:** [AlignFlow: Cycle Consistent Learning from Multiple Domains via Normalizing Flows](https://arxiv.org/abs/1905.12892)  
**Authors:** Aditya Grover, Christopher Chute, Rui Shu, Zhangjie Cao, Stefano Ermon (AAAI 2020)  
**Source:** [ermongroup/alignflow](https://github.com/ermongroup/alignflow)

## Overview

AlignFlow uses normalizing flow models (RealNVP) for unpaired image-to-image translation. Unlike CycleGAN, cycle-consistency is *guaranteed* by invertibility of the generators.

| Model     | Description |
|-----------|-------------|
| **CycleFlow** | Single RealNVP flow: src → tgt (forward), tgt → src (reverse). No cycle-consistency loss. |
| **Flow2Flow** | Two RealNVP flows via shared latent Z: src ↔ Z ↔ tgt. Hybrid GAN + MLE objective. |

## Quick Start

### Inference

```python
from examples.community.alignflow import load_alignflow_pipeline

pipe = load_alignflow_pipeline(
    checkpoint_dir="/path/to/alignflow-checkpoint",
    model_name="CycleFlow",  # or "Flow2Flow"
    device="cuda",
)
out = pipe(source_image=image, output_type="pil")
```

### Training

```python
from examples.community.alignflow import AlignFlowConfig, AlignFlowTrainer

cfg = AlignFlowConfig(
    model="CycleFlow",
    save_dir="./ckpts/alignflow",
    batch_size=16,
    epochs=200,
    resolution=256,
)
trainer = AlignFlowTrainer(cfg)
trainer.train(root_a="./data/trainA", root_b="./data/trainB")
```

## Checkpoint Format

Checkpoints are saved as `checkpoint-epoch-N/alignflow.pt` with:
- `model`: full state dict of the AlignFlow model (generator + discriminators)
- `epoch`: training epoch

For inference, point `checkpoint_dir` to the directory containing `alignflow.pt`.

## Citation

```bibtex
@inproceedings{grover2020alignflow,
  title={AlignFlow: Cycle Consistent Learning from Multiple Domains via Normalizing Flows},
  author={Grover, Aditya and Chute, Christopher and Shu, Rui and Cao, Zhangjie and Ermon, Stefano},
  booktitle={AAAI Conference on Artificial Intelligence},
  year={2020}
}
```
