# Checkpoint Layout Conventions

`from_pretrained(...)` expects method-specific subfolders under your checkpoint root.

| Method | Expected layout (relative to method root) |
| --- | --- |
| `DDBM`, `BBDM`, `BDBM`, `BiBBDM`, `DBIM`, `CDTSDE`, `LBM` | `unet/config.json`, `unet/diffusion_pytorch_model.safetensors`, optional `scheduler/scheduler_config.json` |
| `DDIB` | `source_unet/config.json`, `source_unet/diffusion_pytorch_model.safetensors`, `target_unet/config.json`, `target_unet/diffusion_pytorch_model.safetensors`, optional `scheduler/scheduler_config.json` |
| `I2SB` | `unet/config.json`, `unet/diffusion_pytorch_model.safetensors`, optional `scheduler_config.json` (or `scheduler/scheduler_config.json`) |
| `CUT` | `generator/config.json`, `generator/diffusion_pytorch_model.safetensors` |
| `Pix2PixHD` | `generator/config.json`, `generator/diffusion_pytorch_model.safetensors` |
| `UNSB` | `generator/config.json`, `generator/diffusion_pytorch_model.safetensors` |
| `StegoGAN` | `generator_A/config.json`, `generator_A/diffusion_pytorch_model.safetensors`, `generator_B/config.json`, `generator_B/diffusion_pytorch_model.safetensors` |
| `LocalDiffusion` | `unet/config.json`, `unet/diffusion_pytorch_model.safetensors`, optional `scheduler/scheduler_config.json` |
| Community `sar2optical` | `generator/config.json`, `generator/diffusion_pytorch_model.safetensors` |
| Community `parallel_gan` | `generator/config.json`, `generator/diffusion_pytorch_model.safetensors` |
| Community `e3diff` | `config.json`, `diffusion_pytorch_model.safetensors` |
| Community `diffuseit` | Self-contained subfolders: `imagenet256-uncond/` (unet only), `ffhq-256/` (unet + id_model); each has `unet/config.json`, `unet/diffusion_pytorch_model.safetensors` |
