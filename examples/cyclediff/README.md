# CycleDiff

**Paper:** [CycleDiff: Cycle Diffusion Models for Unpaired Image-to-image Translation](https://arxiv.org/abs/2508.06625) (Zou et al., IEEE TIP 2026)

The full CycleDiff implementation is vendored under `src/models/cyclediff/` (``ddm``, ``unet``, ``taming``, training scripts). No external CycleDiff clone is required.

## Setup

Install project dependencies plus CycleDiff extras (``fvcore``, ``ema-pytorch``, ``accelerate``, etc. from `requirements.txt`).

## Training

Edit a config under `configs/` (paths to datasets and checkpoints are user-specific), then:

```bash
python -m examples.cyclediff.train train \
  --cfg examples/cyclediff/configs/afhq_cat2dog/translation_C_disc_timestep_ode_2.yaml
```

Pretrain VAE and single-domain LDMs first using the YAML files in the same dataset folder (`cat_ae_kl_*.yaml`, `*_ddm_const4_*.yaml`).

## Translation (dataset batch)

```bash
python -m examples.cyclediff.train translate \
  --cfg examples/cyclediff/configs/afhq_cat2dog/translation_C_disc_timestep_ode_2.yaml
```

Set `sampler.ckpt_path`, `sampler.task`, and `sampler.save_folder` in the YAML.

## Inference (Python API)

```python
from src.pipelines.cyclediff import load_cyclediff_pipeline

pipe = load_cyclediff_pipeline(
    "examples/cyclediff/configs/afhq_cat2dog/translation_C_disc_timestep_ode_2.yaml",
    ckpt_path="/path/to/model-12.pt",
    task="cat2dog",
    device="cuda",
)
out = pipe(source_image="cat.png", output_type="pil")
out.images[0].save("dog.png")
```
