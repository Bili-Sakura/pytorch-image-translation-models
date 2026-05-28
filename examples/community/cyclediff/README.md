# CycleDiff (community shim)

**Paper:** *CycleDiff: Cycle Diffusion Models for Unpaired Image-to-image Translation* (Zou et al., IEEE TIP 2026) — [arXiv:2508.06625](https://arxiv.org/abs/2508.06625), [project page](https://zoushilong1024.github.io/CycleDiff/)

**Source:** [ZouShilong1024/CycleDiff](https://github.com/ZouShilong1024/CycleDiff)

The main integration lives under **`src/pipelines/cyclediff.py`** and **`examples/cyclediff/`**. This folder re-exports the same API for backward compatibility.

## Install upstream code

```bash
git clone https://github.com/ZouShilong1024/CycleDiff.git
cd CycleDiff
pip install -r requirement.txt
```

## Use from this repository

```python
from src.pipelines.cyclediff import CycleDiffPipeline, load_cyclediff_pipeline

pipe = load_cyclediff_pipeline()  # or CycleDiffPipeline.from_pretrained(cyclediff_root="...")
pipe.run_training(cfg="./configs/cat2dog/translation_C_disc_timestep_ode_2.yaml")
pipe.run_translation(cfg="./configs/cat2dog/test_translation.yaml")
```

CLI (recommended):

```bash
python -m examples.cyclediff.train train --cfg ./configs/cat2dog/translation_C_disc_timestep_ode_2.yaml
python -m examples.cyclediff.train translate --cfg ./configs/cat2dog/test_translation.yaml
```

See also [examples/cyclediff/](../../cyclediff/) and upstream README for dataset layout and pretraining.
