# pix2pix-turbo

[One-Step Image Translation with Text-to-Image Models](https://arxiv.org/abs/2403.12036) (Parmar et al., 2024) — one-step **paired** translation via SD-Turbo + LoRA adversarial fine-tuning.

Upstream: [GaParmar/img2img-turbo](https://github.com/GaParmar/img2img-turbo)

## Inference

**Edge → image**

```python
from examples.community.pix2pix_turbo import load_pix2pix_turbo_pipeline

pipe = load_pix2pix_turbo_pipeline(pretrained_name="edge_to_image", device="cuda")
out = pipe(source_image=edge_or_photo, prompt="a blue bird", output_type="pil")
```

**Sketch → image (stochastic)**

```python
pipe = load_pix2pix_turbo_pipeline(pretrained_name="sketch_to_image_stochastic", device="cuda")
out = pipe(
    source_image=sketch,
    prompt="ethereal fantasy concept art of an asteroid",
    model_mode="sketch_to_image_stochastic",
    gamma=0.4,
    seed=42,
    output_type="pil",
)
```

Pretrained models: `edge_to_image`, `sketch_to_image_stochastic`.

## Training

Dataset layout:

```
dataset/
  train_A/  train_B/  train_prompts.json
  test_A/   test_B/   test_prompts.json
```

```bash
python -m examples.pix2pix_turbo.train_pix2pix_turbo \
  --dataset_folder dataset/ \
  --output_dir ./outputs/pix2pix_turbo
```

Training requires optional packages: `vision-aided-loss`, `clean-fid`, `clip`, `wandb`.

## Citation

```bibtex
@article{parmar2024one,
  title={One-Step Image Translation with Text-to-Image Models},
  author={Parmar, Gaurav and Park, Taesung and Narasimhan, Srinivasa and Zhu, Jun-Yan},
  journal={arXiv preprint arXiv:2403.12036},
  year={2024}
}
```
