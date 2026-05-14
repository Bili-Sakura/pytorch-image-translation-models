# CycleDiff (community)

**Paper:** *CycleDiff: Cycle Diffusion Models for Unpaired Image-to-image Translation* (Zou et al., IEEE TIP 2026) — [arXiv:2508.06625](https://arxiv.org/abs/2508.06625), [project page](https://zoushilong1024.github.io/CycleDiff/)

**Source:** [ZouShilong1024/CycleDiff](https://github.com/ZouShilong1024/CycleDiff) (clone locally; this monorepo does not vendor the full upstream tree.)

**Architecture:** Two latent diffusion models (domains A and B), cycle-consistent generators, and adversarial discriminators trained with the upstream `ddm/` stack, VAE, and YAML configs.

## Install upstream code

```bash
git clone https://github.com/ZouShilong1024/CycleDiff.git
cd CycleDiff
pip install -r requirement.txt
```

Optional: add as a submodule at the monorepo root:

```bash
git submodule add https://github.com/ZouShilong1024/CycleDiff.git CycleDiff
```

## Use from this repository

**Resolve the checkout** (environment variable `CYCLEDIFF_ROOT`, or paths `./CycleDiff`, `./projects/CycleDiff`, `./external/CycleDiff` relative to the monorepo root):

```python
from examples.community.cyclediff import resolve_cyclediff_root, inject_cyclediff_sys_path

root = resolve_cyclediff_root("/path/to/CycleDiff")
inject_cyclediff_sys_path(root)  # optional; enables `import ddm` in custom scripts
```

**Run upstream training** with working directory set to the CycleDiff root:

```bash
python -m examples.community.cyclediff.train \
  --cyclediff-root /path/to/CycleDiff \
  train_uncond_ldm_cycle.py --cfg ./configs/your_dataset/translation_C_disc_timestep_ode_2.yaml
```

Follow the upstream README for dataset layout, VAE and LDM pretraining, and `accelerate launch translation_uncond_ldm_cycle.py` for testing.

## Citation

```bibtex
@article{zou2025cyclediff,
  title={CycleDiff: Cycle Diffusion Models for Unpaired Image-to-image Translation},
  author={Zou, Shilong and Huang, Yuhang and Yi, Renjiao and Zhu, Chenyang and Xu, Kai},
  journal={arXiv preprint arXiv:2508.06625},
  year={2025}
}
```

Upstream code is MIT-licensed; see the CycleDiff repository `LICENSE`.
