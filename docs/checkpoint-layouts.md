# Checkpoint Layout Conventions

`from_pretrained(...)` expects method-specific subfolders under your checkpoint root.

All checkpoints include `config.yaml` with the full training configuration and checkpoint metadata (`epoch`, `global_step`), regardless of whether training is specified by epochs or steps.

| Method | Expected layout (relative to method root) |
| --- | --- |
| `Pix2Pix` | `generator/config.json`, `generator/diffusion_pytorch_model.safetensors`, `discriminator/config.json`, `discriminator/diffusion_pytorch_model.safetensors`, optional `training_state.pt` |
| `DDBM`, `BBDM`, `BDBM`, `BiBBDM`, `DBIM`, `CDTSDE`, `LBM` | `unet/config.json`, `unet/diffusion_pytorch_model.safetensors`, optional `scheduler/scheduler_config.json`, optional `training_state.pt` (optimizer, epoch, global_step for resume) |
| `DDIB` | `source_unet/config.json`, `source_unet/diffusion_pytorch_model.safetensors`, `target_unet/config.json`, `target_unet/diffusion_pytorch_model.safetensors`, optional `scheduler/scheduler_config.json`, optional `training_state.pt` (optimizers, epoch, global_step for resume) |
| `I2SB` | `unet/config.json`, `unet/diffusion_pytorch_model.safetensors`, optional `scheduler_config.json` (or `scheduler/scheduler_config.json`) |
| `CUT` | `generator/config.json`, `generator/diffusion_pytorch_model.safetensors`, `discriminator/diffusion_pytorch_model.safetensors`, `feature_network/diffusion_pytorch_model.safetensors`, optional `training_state.pt` (optimizers, epoch, global_step for resume) |
| `Pix2PixHD` | `generator/config.json`, `generator/diffusion_pytorch_model.safetensors`, `discriminator/diffusion_pytorch_model.safetensors`, optional `training_state.pt` (optimizers, epoch, global_step for resume) |
| `UNSB` | `generator/model.safetensors`, `discriminator/model.safetensors`, `energy_net/model.safetensors`, `feature_network/model.safetensors`, optional `training_state.pt` (optimizers, epoch, global_step for resume) |
| `StegoGAN` | `generator_A/config.json`, `generator_A/diffusion_pytorch_model.safetensors`, `generator_B/config.json`, `generator_B/diffusion_pytorch_model.safetensors` |
| `LocalDiffusion` | `model/model.safetensors`, optional `training_state.pt` (optimizer, epoch, global_step for resume) |
| `LDDBM` | `encoder_x/`, `encoder_y/`, `decoder_x/`, `bridge/` (each with `diffusion_pytorch_model.safetensors`), optional `training_state.pt` (optimizer, epoch, global_step for resume) |
| Community `sar2optical` | `generator/config.json`, `generator/diffusion_pytorch_model.safetensors` |
| Community `parallel_gan` | `generator/config.json`, `generator/diffusion_pytorch_model.safetensors` |
| Community `e3diff` | `config.json`, `diffusion_pytorch_model.safetensors` |
| Community `diffuseit` | Self-contained subfolders: `imagenet256-uncond/` (unet only), `ffhq-256/` (unet + id_model); each has `unet/config.json`, `unet/diffusion_pytorch_model.safetensors` |
