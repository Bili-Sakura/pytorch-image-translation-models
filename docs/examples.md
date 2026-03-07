# Examples

Extended usage examples for all supported methods. For a minimal quick start, see the [README](../README.md).

## GAN-based translation (Pix2Pix)

```python
import src

gen = src.UNetGenerator(in_channels=3, out_channels=3)
disc = src.PatchGANDiscriminator(in_channels=6)

from src.training import Pix2PixTrainer, TrainingConfig
config = TrainingConfig(epochs=100, device="cuda")
trainer = Pix2PixTrainer(gen, disc, config)
trainer.fit(dataloader)  # expects {"source": tensor, "target": tensor}

translator = src.ImageTranslator(gen, device="cuda")
result = translator.predict(pil_image)
```

## Diffusion bridge translation (I2SB)

```python
from src.models.unet import I2SBUNet, create_model
from src.schedulers import I2SBScheduler
from src.pipelines.i2sb import I2SBPipeline

# Create model and scheduler
model = create_model(
    image_size=256, in_channels=3, num_channels=128,
    num_res_blocks=2, attention_resolutions="32,16,8",
    condition_mode="concat",
)
scheduler = I2SBScheduler(interval=1000, beta_max=0.3)

# Inference pipeline
pipeline = I2SBPipeline(unet=model, scheduler=scheduler)
result = pipeline(source_tensor, nfe=20, output_type="pt")
```

## DDBM bridge diffusion

```python
from src.schedulers import DDBMScheduler
from src.pipelines import DDBMPipeline

scheduler = DDBMScheduler(pred_mode="vp", num_train_timesteps=40)
pipeline = DDBMPipeline(unet=my_unet, scheduler=scheduler)
result = pipeline(source_image, num_inference_steps=40, output_type="pil")
```

## BiBBDM bidirectional translation

```python
from src.schedulers import BiBBDMScheduler
from src.pipelines import BiBBDMPipeline

scheduler = BiBBDMScheduler(num_timesteps=1000, sample_step=100)
pipeline = BiBBDMPipeline(unet=my_unet, scheduler=scheduler)
# Source → Target
result = pipeline(source_tensor, direction="b2a", output_type="pt")
# Target → Source
result = pipeline(target_tensor, direction="a2b", output_type="pt")
```

## DDIB dual-model translation

```python
from src.schedulers import DDIBScheduler
from src.pipelines import DDIBPipeline

scheduler = DDIBScheduler(num_train_timesteps=1000)
pipeline = DDIBPipeline(source_unet=src_model, target_unet=tgt_model, scheduler=scheduler)
result = pipeline(source_image, num_inference_steps=250, output_type="pil")
```

## LBM flow-matching translation

```python
from src.schedulers import LBMScheduler
from src.pipelines import LBMPipeline

scheduler = LBMScheduler(num_train_timesteps=1000)
pipeline = LBMPipeline(unet=my_unet, scheduler=scheduler)
result = pipeline(source_image, num_inference_steps=1, output_type="pil")
```

## DiT backbone (SiT) for diffusion bridges

```python
from src.models.dit import SiTBackbone, SIT_CONFIGS

# Create a SiT-S/2 backbone (small, patch size 2)
depth, hidden_size, num_heads = SIT_CONFIGS["S"]
model = SiTBackbone(
    image_size=256, patch_size=2, in_channels=3,
    hidden_size=hidden_size, depth=depth, num_heads=num_heads,
    condition_mode="concat",
)
# Use as drop-in replacement for UNet in any bridge pipeline
output = model(noisy_sample, timestep, xT=source_image)
```

## UNSB unpaired translation (multi-step Schrödinger Bridge)

```python
from src.models.unsb import create_generator
from src.schedulers.unsb import UNSBScheduler
from src.pipelines.unsb import UNSBPipeline

# Create time-conditional generator and scheduler
generator = create_generator(input_nc=3, output_nc=3, ngf=64, n_blocks=9)
scheduler = UNSBScheduler(num_timesteps=5, tau=0.01)

# Inference pipeline (multi-step stochastic refinement)
pipeline = UNSBPipeline(generator=generator, scheduler=scheduler)
result = pipeline(source_image, output_type="pt")
print(result.nfe)  # 5 function evaluations
```

## UNSB training

```python
from examples.unsb.config import UNSBConfig
from examples.unsb.train_unsb import UNSBTrainer

cfg = UNSBConfig(
    input_nc=3, output_nc=3, ngf=64,
    num_timesteps=5, tau=0.01,
    lambda_GAN=1.0, lambda_SB=1.0, lambda_NCE=1.0,
    device="cuda",
)
trainer = UNSBTrainer(cfg)
# Single training step with unpaired data
losses = trainer.train_step(real_A_batch, real_B_batch)
```

## Local Diffusion hallucination-aware translation

```python
from src.models.local_diffusion import create_unet
from src.schedulers.local_diffusion import LocalDiffusionScheduler
from src.pipelines.local_diffusion import LocalDiffusionPipeline

# Create conditional U-Net and Gaussian diffusion scheduler
unet = create_unet(dim=32, channels=1, dim_mults=(1, 2, 4, 8))
scheduler = LocalDiffusionScheduler(num_train_timesteps=250, beta_schedule="sigmoid")

# Standard inference
pipeline = LocalDiffusionPipeline(unet=unet, scheduler=scheduler)
result = pipeline(cond_image, output_type="pt")

# Branch-and-fuse inference (hallucination suppression)
result = pipeline(
    cond_image, anomaly_mask=mask,
    branch_out=True, fusion_timestep=2, output_type="pt",
)
```

## Local Diffusion training

```python
from examples.local_diffusion.config import LocalDiffusionConfig
from examples.local_diffusion.train_local_diffusion import LocalDiffusionTrainer

cfg = LocalDiffusionConfig(
    dim=32, channels=1,
    num_train_timesteps=250, beta_schedule="sigmoid",
    objective="pred_x0", device="cuda",
)
trainer = LocalDiffusionTrainer(cfg)
losses = trainer.train_step(source_batch, target_batch)
```

## I2SB training with task configs

```python
from examples.i2sb.config import sar2eo_config
from examples.i2sb.trainer import I2SBTrainer

cfg = sar2eo_config(resolution=256, train_batch_size=8)
trainer = I2SBTrainer(cfg)
model = trainer.build_model()
scheduler = trainer.build_scheduler()

# Single-step loss computation
loss = I2SBTrainer.compute_training_loss(model, scheduler, source_batch, target_batch)
loss.backward()
```

## StegoGAN non-bijective translation

```python
from src.training import StegoGANTrainer, StegoGANConfig

cfg = StegoGANConfig(
    input_nc=3, output_nc=3, ngf=64,
    lambda_reg=0.3, lambda_consistency=1.0,
    resnet_layer=8, fusionblock=True,
    device="cuda",
)
trainer = StegoGANTrainer(cfg)
# Run a single training step with unpaired data
losses = trainer.train_step(real_A_batch, real_B_batch)
```
