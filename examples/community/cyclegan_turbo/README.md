# CycleGAN-Turbo

[One-Step Image Translation with Text-to-Image Models](https://arxiv.org/abs/2403.12036) (Parmar et al., 2024) — one-step **unpaired** translation via SD-Turbo + LoRA adversarial fine-tuning.

Upstream: [GaParmar/img2img-turbo](https://github.com/GaParmar/img2img-turbo)

## Inference

```python
from examples.community.cyclegan_turbo import load_cyclegan_turbo_pipeline

pipe = load_cyclegan_turbo_pipeline(pretrained_name="day_to_night", device="cuda")
out = pipe(source_image=image, output_type="pil")
out.images[0].save("night.png")
```

Pretrained models: `day_to_night`, `night_to_day`, `clear_to_rainy`, `rainy_to_clear`.

Custom checkpoint:

```python
pipe = load_cyclegan_turbo_pipeline(
    pretrained_path="./checkpoints/cyclegan_turbo/checkpoints/model_5000.pkl",
    device="cuda",
)
out = pipe(source_image=image, direction="a2b", prompt="driving in the night", output_type="pil")
```

## Training

Dataset layout (CycleGAN-style):

```
dataset/
  train_A/  train_B/
  test_A/   test_B/
  fixed_prompt_a.txt
  fixed_prompt_b.txt
```

```bash
python -m examples.cyclegan_turbo.train_cyclegan_turbo \
  --dataset_folder dataset/ \
  --output_dir ./outputs/cyclegan_turbo \
  --train_img_prep resize_512x512 \
  --val_img_prep resize_512x512 \
  --tracker_project_name cyclegan_turbo
```

Training requires optional packages: `vision-aided-loss`, `clean-fid`, `wandb`.

## Citation

```bibtex
@article{parmar2024one,
  title={One-Step Image Translation with Text-to-Image Models},
  author={Parmar, Gaurav and Park, Taesung and Narasimhan, Srinivasa and Zhu, Jun-Yan},
  journal={arXiv preprint arXiv:2403.12036},
  year={2024}
}
```
