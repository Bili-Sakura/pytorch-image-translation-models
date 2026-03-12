# MDT / LDDBM (Multimodal Distribution Translation)

**Paper:** *Towards General Modality Translation with Contrastive and Predictive Latent Diffusion Bridge* (Bosch Research, NeurIPS 2025 submission)

**Source:** [boschresearch/Multimodal-Distribution-Translation-MDT](https://github.com/boschresearch/Multimodal-Distribution-Translation-MDT)

## Architecture

MDT (LDDBM) uses a **latent diffusion bridge** for general modality translation:

- **Encoder_x / Encoder_y**: KL-VAE encoders (or task-specific encoders) map source/target images to latent space.
- **Decoder_x / Decoder_y**: KL-VAE decoders map latents back to images.
- **BridgeModel**: A transformer-based diffusion bridge translates latents from source to target domain using Karras scheduling.

Supported tasks include:
- **Super-resolution** (16×16 → 128×128): LR images to HR
- **Multi-view to 3D** (ShapeNet): Multiple 2D views to 3D reconstruction

## Installation

This community pipeline requires the upstream MDT repository to be installed:

```bash
git clone https://github.com/boschresearch/Multimodal-Distribution-Translation-MDT
cd Multimodal-Distribution-Translation-MDT
pip install -e .
pip install -r requirements.txt
```

## Quick Start

```python
from examples.community.mdt import load_mdt_community_pipeline, MDTPipeline

# Load pipeline (expects config.json + diffusion_pytorch_model.safetensors per component)
pipe = load_mdt_community_pipeline(
    checkpoint_dir="/path/to/mdt-checkpoints",
    task="sr_16_to_128",
    device="cuda",
)

# Inference: source_image is LR (16×16) for super-resolution
output = pipe(
    source_image=lr_image,
    num_inference_steps=40,
    output_type="pil",
)
output.images[0].save("sr_output.png")
```

## Checkpoint Layout (project style)

Each component uses `config.json` + `diffusion_pytorch_model.safetensors`:

```
checkpoint_dir/
  encoder_x/
    config.json
    diffusion_pytorch_model.safetensors
  encoder_y/
    config.json
    diffusion_pytorch_model.safetensors
  decoder_x/
    config.json
    diffusion_pytorch_model.safetensors
  bridge/
    config.json
    diffusion_pytorch_model.safetensors
```

For super-resolution, `decoder_y` is not used (`NoDecoder`).

**Converting from raw .pt:** Use the conversion script to produce this layout:

```bash
python -m examples.community.mdt.convert_pt_to_mdt /path/to/output \\
    --encoder-x encoder_x.pt --encoder-y encoder_y.pt \\
    --decoder-x decoder_x.pt --bridge bridge.pt
```

## Training

Training should be done in the upstream repository. See [train.py](train.py) for commands.

## Citation

```bibtex
@article{mdt2025,
  title={Towards General Modality Translation with Contrastive and Predictive Latent Diffusion Bridge},
  author={Bosch Research},
  year={2025},
  note={NeurIPS 2025 submission}
}
```
