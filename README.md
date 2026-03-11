# pytorch-image-translation-models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![PyPI version](https://img.shields.io/pypi/v/pytorch-image-translation-models.svg)](https://pypi.org/project/pytorch-image-translation-models/) [![Checkpoint Collections](https://img.shields.io/badge/🤗%20Checkpoints-Collection-FFD21E)](https://huggingface.co/collections/BiliSakura/image-translation-checkpoint-collections)

A PyTorch library for multi-modal image translation with diffusion bridges, GANs, and transformer backbones.

## Installation

### Install from PyPI

```bash
pip install pytorch-image-translation-models
```

### Install from source

```bash
pip install -e .
```

With optional dependencies:

```bash
# With training extras (accelerate, peft, datasets, tensorboard)
pip install -e ".[training]"

# With metrics extras (torchmetrics, lpips, torch-fidelity, scipy)
pip install -e ".[metrics]"

# Everything
pip install -e ".[all]"
```

> **Note:** PyTorch is listed as a dependency but you may want to install a specific CUDA build first.
> See [PyTorch — Get Started](https://pytorch.org/get-started/previous-versions/) for details.

## Quick Start

Examples default to `device="cuda"`. If your environment is CPU-only, replace `"cuda"` with `"cpu"`.

```python
from PIL import Image

# Baseline method (DiffuseIT) - diffusion-based image translation
from examples.baselines.diffuseit import DiffuseITPipeline

pipe = DiffuseITPipeline.from_pretrained(
    "projects/DiffuseIT",  # Clone from https://github.com/cyclomon/DiffuseIT
    timestep_respacing="100",
    skip_timesteps=80,
)
pipe.to("cuda")

source = Image.open("/path/to/source.png").convert("RGB")
# Text-guided
out = pipe(
    source_image=source,
    prompt="Black Leopard",
    source="Lion",
    use_range_restart=True,
    use_noise_aug_all=True,
    output_type="pil",
)
out.images[0].save("diffuseit_output.png")

# Alternative: UNSB baseline
from src.pipelines.unsb import UNSBPipeline
unsb = UNSBPipeline.from_pretrained(
    "path/to/UNSB-ckpt/horse2zebra",  # https://huggingface.co/BiliSakura/UNSB-ckpt
    subfolder="generator",
    scheduler_num_timesteps=5,
    scheduler_tau=0.01,
)
unsb.to("cuda")
unsb_out = unsb(source_image=source, output_type="pil")
unsb_out.images[0].save("unsb_output.png")

# Community method (E3Diff)
from examples.community.e3diff import E3DiffPipeline
e3diff = E3DiffPipeline.from_pretrained("path/to/E3Diff-ckpt/SEN12")
e3diff.to("cuda")
community_out = e3diff(source_image=source, num_inference_steps=50, output_type="pil")
community_out.images[0].save("e3diff_output.png")
```

## Documentation

All information regarding per-method checkpoint folder conventions required by `from_pretrained(...)`, as well as comprehensive package documentation, is integrated below.

| Doc | Description |
| --- | --- |
| [Checkpoint layouts](docs/checkpoint-layouts.md) | Provides detailed checkpoint folder structures, naming conventions, and requirements for each pipeline and the `from_pretrained(...)` API. |
| [Features](docs/features.md) | Documents supported models, schedulers, pipelines, data types, training methods, and evaluation metrics. |
| [Metrics README](src/metrics/README.md) | One-stop usage for paired/unpaired metrics and custom HuggingFace/local checkpoints. |
| [Examples](docs/examples.md) | Extended usage patterns and code snippets for pipelines such as I2SB, DDBM, UNSB, and Local Diffusion. |
| [Package structure](docs/package-structure.md) | Overview of the codebase organization, modules, and directories. |
| [Credits](docs/credits.md) | Citations for reference papers and third-party contributions. |

## Credits

This repository/package is primarily built upon [4th-MAVIC-T](https://github.com/Bili-Sakura/4th-MAVIC-T) by the **EarthBridge Team**:

- **Zheyuan Chen** — bilisakura@zju.edu.cn  
- **Yuanshen Guan** — guanys@mail.ustc.edu.cn  

## License

MIT
