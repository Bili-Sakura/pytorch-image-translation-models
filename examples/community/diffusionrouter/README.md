# DiffusionRouter (Community)

**Paper:** *Universal Multi-Domain Translation via Diffusion Routers*

**Source:** [kvmduc/DiffusionRouter](https://github.com/kvmduc/DiffusionRouter)

DiffusionRouter performs multi-domain translation with conditional diffusion models and optional multi-hop routing across intermediate modalities.

## Setup

Clone the original source repo locally:

```bash
git clone https://github.com/kvmduc/DiffusionRouter.git
```

## Quick start

```python
from examples.community.diffusionrouter import load_diffusionrouter_community_pipeline

pipe = load_diffusionrouter_community_pipeline(
    checkpoint_path="/path/to/model.pt",
    diffusionrouter_src_path="/path/to/DiffusionRouter",
    device="cuda",
)

# COCO class IDs: 0=color, 1=edge, 2=gray, 3=depth
out = pipe(
    source_image=pil_image,
    context_class=1,      # source domain
    target_class=3,       # target domain
    via_seq="auto",       # or "none", or explicit "0,1"
    output_type="pil",
)
```

## Notes

- The loader expects a `.pt` checkpoint from DiffusionRouter training/distillation.
- Routing follows the default chain `gray(2) ↔ color(0) ↔ edge(1) ↔ depth(3)` when `via_seq="auto"`.

