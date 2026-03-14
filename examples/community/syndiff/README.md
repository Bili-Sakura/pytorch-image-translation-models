# SynDiff (Community)

**Paper:** *Unsupervised Medical Image Translation With Adversarial Diffusion Models* (Özbey et al., IEEE TMI 2023)

**Source:** [icon-lab/SynDiff](https://github.com/icon-lab/SynDiff)

SynDiff is an adversarial diffusion model for unsupervised medical image translation (e.g., T1↔T2 MRI). It uses NCSN++ generators with posterior sampling for cross-modality synthesis.

## Checkpoint format

SynDiff checkpoints are stored as `gen_diffusive_1_{epoch}.pth` and `gen_diffusive_2_{epoch}.pth` in the experiment folder. Pretrained models are available for:

- **IXI**: T1↔PD, PD↔T1
- **BRATS**: T1↔T2, T2↔T1

See the [official repo](https://github.com/icon-lab/SynDiff#pretrained-models) for download links.

## Quick start

```python
from examples.community.syndiff import load_syndiff_community_pipeline

pipe = load_syndiff_community_pipeline(
    checkpoint_dir="/path/to/exp_syndiff",
    direction="contrast1_to_contrast2",  # or "contrast2_to_contrast1"
    which_epoch=50,
    device="cuda",
)

# source: [B, 1, H, W] in [-1, 1], or PIL/numpy
out = pipe(source_image=source_tensor, num_inference_steps=4, output_type="pil")
```

## Pipeline parameters

| Parameter | Description |
|-----------|-------------|
| `checkpoint_dir` | Path to experiment folder (contains `gen_diffusive_1_{epoch}.pth`) |
| `direction` | `"contrast1_to_contrast2"` (uses gen_diffusive_1) or `"contrast2_to_contrast1"` |
| `which_epoch` | Epoch number of checkpoint (default: 50) |
| `image_size` | Image size (default: 256) |
| `num_channels` | Input channels (default: 2 for noise+source) |
| `num_channels_dae` | Generator base channels (default: 64) |
| `ch_mult` | Channel multipliers (default: [1,1,2,2,4,4]) |
| `num_timesteps` | Diffusion steps (default: 4) |
| `num_res_blocks` | ResBlocks per scale (default: 2) |
| `contrast1` | Name of contrast 1 (default: "T1") |
| `contrast2` | Name of contrast 2 (default: "T2") |

## Citation

```bibtex
@ARTICLE{ozbey_dalmaz_syndiff_2024,
  author={Özbey, Muzaffer and Dalmaz, Onat and Dar, Salman U. H. and Bedel, Hasan A. and Özturk, Şaban and Güngör, Alper and Çukur, Tolga},
  journal={IEEE Transactions on Medical Imaging},
  title={Unsupervised Medical Image Translation With Adversarial Diffusion Models},
  year={2023},
  volume={42},
  number={12},
  pages={3524-3539},
  doi={10.1109/TMI.2023.3290149}}
```
