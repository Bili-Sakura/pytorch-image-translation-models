# CycleDiffusion (community)

**Paper:** *A Latent Space of Stochastic Diffusion Models for Zero-Shot Image Editing and Guidance* (Wu & De la Torre, ICCV 2023) — [arXiv:2210.05559](https://arxiv.org/abs/2210.05559), [project page](https://chenwu.io/project/cycle-diffusion/)

**Source:** [humansensinglab/cycle-diffusion](https://github.com/humansensinglab/cycle-diffusion) (maintained fork; original development by [ChenWu98/cycle-diffusion](https://github.com/ChenWu98/cycle-diffusion)). Clone locally; this monorepo does not vendor the full upstream tree.

**Note:** This is **not** the same method as [CycleDiff](../cyclediff/) (Zou et al., IEEE TIP 2026), which targets cycle latent diffusion for unpaired translation.

**Architecture:** Hugging Face `Trainer`-style loop in `main.py`, task configs under `config/`, models under `model/`, and CLIP-based evaluation.

## Install upstream code

```bash
git clone https://github.com/humansensinglab/cycle-diffusion.git
cd cycle-diffusion
conda env create -f environment.yml
conda activate generative_prompt
```

Follow the upstream README for CLIP, PyTorch, taming-transformers, checkpoints under `ckpts/`, and wandb setup.

Optional submodule at the monorepo root:

```bash
git submodule add https://github.com/humansensinglab/cycle-diffusion.git cycle-diffusion
```

## Use from this repository

**Resolve the checkout** (environment variable `CYCLE_DIFFUSION_ROOT`, or paths `./cycle-diffusion`, `./projects/cycle-diffusion`, `./external/cycle-diffusion` relative to the monorepo root):

```python
from examples.community.cycle_diffusion import (
    resolve_cycle_diffusion_root,
    inject_cycle_diffusion_sys_path,
)

root = resolve_cycle_diffusion_root("/path/to/cycle-diffusion")
inject_cycle_diffusion_sys_path(root)  # optional; enables `import model`, `import trainer`, etc.
```

**Run upstream** with working directory set to the clone root. Experiment configs in this fork live under `config/experiments/` (not a top-level `experiments/` folder):

```bash
python -m examples.community.cycle_diffusion.train \
  --cycle-diffusion-root /path/to/cycle-diffusion \
  main.py --cfg config/experiments/translate_text2img256_stable_diffusion_stochastic_1.cfg \
  --run_name demo --do_eval --output_dir output/demo
```

For multi-GPU, use `torch.distributed.launch` or `accelerate` as in the upstream README, still invoking `main.py` from the clone root.

## Citation

```bibtex
@inproceedings{cyclediffusion,
  title={A Latent Space of Stochastic Diffusion Models for Zero-Shot Image Editing and Guidance},
  author={Wu, Chen Henry and De la Torre, Fernando},
  booktitle={ICCV},
  year={2023},
}
```

Upstream code uses the X11 License; see the repository `LICENSE`.
