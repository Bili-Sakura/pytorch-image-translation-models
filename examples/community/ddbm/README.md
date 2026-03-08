# DDBM Community Pipeline

DDBM (Denoising Diffusion Bridge Models) for checkpoints using the OpenAI/improved_diffusion architecture (e.g. [alexzhou907/DDBM](https://github.com/alexzhou907/DDBM), [BiliSakura/DDBM-ckpt](https://huggingface.co/BiliSakura/DDBM-ckpt)).

The standard `src` DDBM uses diffusers `UNet2DModel`, which does not match this architecture. This community package provides a compatible UNet and loader.

## Conversion (pt → unet/)

Convert raw `.pt` checkpoints to the project's unet format (config.json + safetensors):

```bash
python -m examples.community.ddbm.convert_pt_to_unet \
    /path/to/DDBM-ckpt/edges2handbags-vp \
    --checkpoint e2h_ema_0.9999_420000.pt
```

Output: `unet/config.json` + `unet/diffusion_pytorch_model.safetensors`.

## Inference

```python
from examples.community.ddbm import load_ddbm_community_pipeline

pipe = load_ddbm_community_pipeline(
    "/path/to/DDBM-ckpt/edges2handbags-vp",
    device="cuda",
)
out = pipe(source_image=image, num_inference_steps=40, output_type="pil")
out.images[0].save("output.png")
```

## Citation

```bibtex
@inproceedings{zhou2024ddbm,
  title={Denoising Diffusion Bridge Models},
  author={Zhou, Linqi and Lou, Aaron and Khanna, Samar and Ermon, Stefano},
  booktitle={ICLR},
  year={2024}
}
```
