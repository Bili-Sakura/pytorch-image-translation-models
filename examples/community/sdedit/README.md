# SDEdit (community)

**Paper:** *SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations* (Meng et al., ICLR 2022) — [arXiv:2108.01073](https://arxiv.org/abs/2108.01073), [project page](https://sde-image-editing.github.io/)

**Source:** [ermongroup/SDEdit](https://github.com/ermongroup/SDEdit). Clone locally; this monorepo does not vendor the full upstream tree.

**Architecture:** VP-SDE image editing and stroke-based generation with configs under `configs/` and sampling in `runners/image_editing.py`. Pretrained checkpoints are downloaded automatically when you run sampling (see upstream README).

## Install upstream code

```bash
git clone https://github.com/ermongroup/SDEdit.git
cd SDEdit
pip install -r requirements.txt
```

Follow the upstream README for data format (`[image, mask]` tensors), example `.npy` inputs, and Colab.

Optional placement next to this repository (auto-discovered):

```bash
git clone https://github.com/ermongroup/SDEdit.git
# or: git submodule add https://github.com/ermongroup/SDEdit.git SDEdit
```

## Use from this repository

**Resolve the checkout** (environment variable `SDEDIT_ROOT`, or paths `./SDEdit`, `./sdedit`, `./projects/SDEdit`, `./external/SDEdit` relative to the monorepo root):

```python
from examples.community.sdedit import (
    resolve_sdedit_root,
    inject_sdedit_sys_path,
)

root = resolve_sdedit_root("/path/to/SDEdit")
inject_sdedit_sys_path(root)  # optional; enables `import runners`, `import models`, etc.
```

**Run upstream** with working directory set to the clone root:

```bash
python -m examples.community.sdedit.train \
  --sdedit-root /path/to/SDEdit \
  main.py --exp ./runs/ --config bedroom.yml --sample -i images \
  --npy_name lsun_bedroom1 --sample_step 3 --t 500 --ni
```

Stroke-based editing (church example from upstream):

```bash
python -m examples.community.sdedit.train \
  --sdedit-root /path/to/SDEdit \
  main.py --exp ./runs/ --config church.yml --sample -i images \
  --npy_name lsun_edit --sample_step 3 --t 500 --ni
```

## Citation

```bibtex
@inproceedings{
      meng2022sdedit,
      title={{SDE}dit: Guided Image Synthesis and Editing with Stochastic Differential Equations},
      author={Chenlin Meng and Yutong He and Yang Song and Jiaming Song and Jiajun Wu and Jun-Yan Zhu and Stefano Ermon},
      booktitle={International Conference on Learning Representations},
      year={2022},
}
```

Upstream code is licensed under the MIT License; see the repository `LICENSE`.
