# CycleDiff

**Paper:** [CycleDiff: Cycle Diffusion Models for Unpaired Image-to-image Translation](https://arxiv.org/abs/2508.06625) (Zou et al., IEEE TIP 2026)

**Upstream:** [ZouShilong1024/CycleDiff](https://github.com/ZouShilong1024/CycleDiff)

## Setup

```bash
git clone https://github.com/ZouShilong1024/CycleDiff.git
cd CycleDiff && pip install -r requirement.txt
export CYCLEDIFF_ROOT=/path/to/CycleDiff
```

## Training and translation

```bash
# Cycle LDM training (main recipe)
python -m examples.cyclediff.train train \
  --cfg ./configs/cat2dog/translation_C_disc_timestep_ode_2.yaml

# Inference / evaluation
python -m examples.cyclediff.train translate \
  --cfg ./configs/cat2dog/test_translation.yaml
```

## Python API

```python
from src.pipelines.cyclediff import load_cyclediff_pipeline

pipe = load_cyclediff_pipeline()
pipe.run_training(cfg="./configs/cat2dog/translation_C_disc_timestep_ode_2.yaml")
pipe.run_translation(cfg="./configs/cat2dog/test_translation.yaml")
```

Configs and checkpoints follow the upstream repository layout and YAML files.
