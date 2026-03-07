# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Tests for Local Diffusion (hallucination-aware diffusion) components."""

import torch
import numpy as np
import pytest

from src.models.local_diffusion import (
    LocalDiffusionUNet,
    ConditionEncoder,
    create_unet,
)
from src.schedulers.local_diffusion import (
    LocalDiffusionScheduler,
    LocalDiffusionSchedulerOutput,
)
from src.pipelines.local_diffusion import (
    LocalDiffusionPipeline,
    LocalDiffusionPipelineOutput,
)


# Common test fixtures
DIM = 16
CHANNELS = 1
DIM_MULTS = (1, 2)
GROUPS = 4
FULL_ATTN = (False, True)
IMG_SIZE = 16
T_STEPS = 20


def _make_unet(**kwargs):
    defaults = dict(
        dim=DIM, channels=CHANNELS, dim_mults=DIM_MULTS,
        resnet_block_groups=GROUPS, full_attn=FULL_ATTN,
    )
    defaults.update(kwargs)
    return create_unet(**defaults)


# ---------------------------------------------------------------------------
# ConditionEncoder tests
# ---------------------------------------------------------------------------


class TestConditionEncoder:
    """Tests for ConditionEncoder."""

    def test_output_shape(self):
        enc = ConditionEncoder(in_channels=1, filters=(16, 32))
        x = torch.randn(1, 1, 16, 16)
        out = enc(x)
        assert out.shape[0] == 1
        assert out.shape[1] == 32  # last filter
        assert out.shape[2] == 8   # one pool: 16 / 2 = 8

    def test_multi_stage(self):
        enc = ConditionEncoder(in_channels=1, filters=(16, 32, 64))
        x = torch.randn(1, 1, 32, 32)
        out = enc(x)
        assert out.shape[1] == 64
        assert out.shape[2] == 8   # two pools: 32 / 4 = 8

    def test_rgb_input(self):
        enc = ConditionEncoder(in_channels=3, filters=(16, 32))
        x = torch.randn(1, 3, 16, 16)
        out = enc(x)
        assert out.shape[0] == 1
        assert out.shape[1] == 32


# ---------------------------------------------------------------------------
# LocalDiffusionUNet tests
# ---------------------------------------------------------------------------


class TestLocalDiffusionUNet:
    """Tests for LocalDiffusionUNet."""

    def test_output_shape(self):
        unet = _make_unet()
        x = torch.randn(1, CHANNELS, IMG_SIZE, IMG_SIZE)
        cond = torch.randn(1, CHANNELS, IMG_SIZE, IMG_SIZE)
        t = torch.tensor([5]).long()
        out = unet(x, cond, t)
        assert out.shape == (1, CHANNELS, IMG_SIZE, IMG_SIZE)

    def test_batch_size(self):
        unet = _make_unet()
        x = torch.randn(2, CHANNELS, IMG_SIZE, IMG_SIZE)
        cond = torch.randn(2, CHANNELS, IMG_SIZE, IMG_SIZE)
        t = torch.tensor([5, 10]).long()
        out = unet(x, cond, t)
        assert out.shape == (2, CHANNELS, IMG_SIZE, IMG_SIZE)

    def test_different_timesteps(self):
        unet = _make_unet()
        x = torch.randn(1, CHANNELS, IMG_SIZE, IMG_SIZE)
        cond = torch.randn(1, CHANNELS, IMG_SIZE, IMG_SIZE)
        outputs = []
        for t_val in [0, 5, 10, 15]:
            out = unet(x, cond, torch.tensor([t_val]).long())
            outputs.append(out)
        # Different timesteps should produce different outputs
        assert not torch.allclose(outputs[0], outputs[1])

    def test_rgb_channels(self):
        unet = _make_unet(channels=3)
        x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        cond = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        t = torch.tensor([5]).long()
        out = unet(x, cond, t)
        assert out.shape == (1, 3, IMG_SIZE, IMG_SIZE)

    def test_four_level(self):
        unet = create_unet(
            dim=8, channels=1, dim_mults=(1, 2, 4, 8),
            resnet_block_groups=4, full_attn=(False, False, False, True),
        )
        x = torch.randn(1, 1, 32, 32)
        cond = torch.randn(1, 1, 32, 32)
        t = torch.tensor([5]).long()
        out = unet(x, cond, t)
        assert out.shape == (1, 1, 32, 32)

    def test_custom_cond_channels(self):
        unet = _make_unet(channels=1, cond_in_channels=3)
        x = torch.randn(1, 1, IMG_SIZE, IMG_SIZE)
        cond = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        t = torch.tensor([5]).long()
        out = unet(x, cond, t)
        assert out.shape == (1, 1, IMG_SIZE, IMG_SIZE)


# ---------------------------------------------------------------------------
# Scheduler tests
# ---------------------------------------------------------------------------


class TestLocalDiffusionScheduler:
    """Tests for LocalDiffusionScheduler."""

    @pytest.fixture
    def scheduler(self):
        return LocalDiffusionScheduler(
            num_train_timesteps=T_STEPS,
            beta_schedule="sigmoid",
            objective="pred_x0",
        )

    def test_beta_shapes(self, scheduler):
        assert scheduler.betas.shape == (T_STEPS,)
        assert scheduler.alphas_cumprod.shape == (T_STEPS,)
        assert scheduler.sqrt_alphas_cumprod.shape == (T_STEPS,)

    def test_alphas_cumprod_decreasing(self, scheduler):
        for i in range(1, T_STEPS):
            assert scheduler.alphas_cumprod[i] < scheduler.alphas_cumprod[i - 1]

    def test_q_sample(self, scheduler):
        x0 = torch.randn(1, 1, IMG_SIZE, IMG_SIZE)
        noise = torch.randn_like(x0)
        t = torch.tensor([T_STEPS // 2]).long()
        xt = scheduler.q_sample(x0, t, noise)
        assert xt.shape == x0.shape

    def test_step_ddpm(self, scheduler):
        x = torch.randn(1, 1, IMG_SIZE, IMG_SIZE)
        model_out = torch.randn(1, 1, IMG_SIZE, IMG_SIZE)
        t = torch.tensor([10]).long()
        result = scheduler.step(model_out, t, x)
        assert isinstance(result, LocalDiffusionSchedulerOutput)
        assert result.prev_sample.shape == x.shape
        assert result.pred_x_start.shape == x.shape

    def test_step_ddim(self, scheduler):
        x = torch.randn(1, 1, IMG_SIZE, IMG_SIZE)
        model_out = torch.randn(1, 1, IMG_SIZE, IMG_SIZE)
        result = scheduler.ddim_step(model_out, 10, 5, x)
        assert isinstance(result, LocalDiffusionSchedulerOutput)
        assert result.prev_sample.shape == x.shape

    def test_compute_loss(self, scheduler):
        pred = torch.randn(2, 1, IMG_SIZE, IMG_SIZE)
        target = torch.randn(2, 1, IMG_SIZE, IMG_SIZE)
        t = torch.tensor([5, 10]).long()
        loss = scheduler.compute_loss(pred, target, t)
        assert loss.shape == torch.Size([])
        assert loss.item() > 0

    def test_linear_schedule(self):
        sched = LocalDiffusionScheduler(num_train_timesteps=T_STEPS, beta_schedule="linear")
        assert sched.betas.shape == (T_STEPS,)

    def test_cosine_schedule(self):
        sched = LocalDiffusionScheduler(num_train_timesteps=T_STEPS, beta_schedule="cosine")
        assert sched.betas.shape == (T_STEPS,)

    def test_pred_noise_objective(self):
        sched = LocalDiffusionScheduler(num_train_timesteps=T_STEPS, objective="pred_noise")
        x0 = torch.randn(1, 1, IMG_SIZE, IMG_SIZE)
        noise = torch.randn_like(x0)
        t = torch.tensor([5]).long()
        xt = sched.q_sample(x0, t, noise)
        x0_hat, _ = sched.model_output_to_x0_and_noise(noise, xt, t)
        assert x0_hat.shape == x0.shape

    def test_pred_v_objective(self):
        sched = LocalDiffusionScheduler(num_train_timesteps=T_STEPS, objective="pred_v")
        assert sched.loss_weight.shape == (T_STEPS,)


# ---------------------------------------------------------------------------
# Pipeline tests
# ---------------------------------------------------------------------------


class TestLocalDiffusionPipeline:
    """Tests for LocalDiffusionPipeline."""

    @pytest.fixture
    def pipeline(self):
        unet = _make_unet()
        scheduler = LocalDiffusionScheduler(
            num_train_timesteps=T_STEPS,
            beta_schedule="sigmoid",
            objective="pred_x0",
        )
        return LocalDiffusionPipeline(unet=unet, scheduler=scheduler)

    def test_pipeline_pt_output(self, pipeline):
        cond = torch.randn(1, CHANNELS, IMG_SIZE, IMG_SIZE)
        result = pipeline(cond, num_inference_steps=3, output_type="pt")
        assert isinstance(result, LocalDiffusionPipelineOutput)
        assert isinstance(result.images, torch.Tensor)
        assert result.images.shape == (1, CHANNELS, IMG_SIZE, IMG_SIZE)
        assert result.nfe == 3

    def test_pipeline_pil_output(self, pipeline):
        cond = torch.randn(1, CHANNELS, IMG_SIZE, IMG_SIZE)
        result = pipeline(cond, num_inference_steps=3, output_type="pil")
        assert isinstance(result.images, list)
        assert len(result.images) == 1

    def test_pipeline_np_output(self, pipeline):
        cond = torch.randn(1, CHANNELS, IMG_SIZE, IMG_SIZE)
        result = pipeline(cond, num_inference_steps=3, output_type="np")
        assert isinstance(result.images, np.ndarray)

    def test_pipeline_ddim(self, pipeline):
        cond = torch.randn(1, CHANNELS, IMG_SIZE, IMG_SIZE)
        result = pipeline(cond, use_ddim=True, ddim_steps=3, output_type="pt")
        assert result.images.shape == (1, CHANNELS, IMG_SIZE, IMG_SIZE)
        assert result.nfe == 3

    def test_pipeline_branching(self, pipeline):
        cond = torch.randn(1, CHANNELS, IMG_SIZE, IMG_SIZE)
        mask = torch.zeros(1, CHANNELS, IMG_SIZE, IMG_SIZE)
        mask[:, :, :8, :] = 1.0
        result = pipeline(
            cond, anomaly_mask=mask, num_inference_steps=5,
            branch_out=True, fusion_timestep=2, output_type="pt",
        )
        assert result.images.shape == (1, CHANNELS, IMG_SIZE, IMG_SIZE)
        assert result.nfe > 5  # Branching doubles NFE for some steps

    def test_pipeline_branching_ddim(self, pipeline):
        cond = torch.randn(1, CHANNELS, IMG_SIZE, IMG_SIZE)
        mask = torch.zeros(1, CHANNELS, IMG_SIZE, IMG_SIZE)
        mask[:, :, :8, :] = 1.0
        result = pipeline(
            cond, anomaly_mask=mask, use_ddim=True, ddim_steps=5,
            branch_out=True, fusion_timestep=2, output_type="pt",
        )
        assert result.images.shape == (1, CHANNELS, IMG_SIZE, IMG_SIZE)

    def test_pipeline_return_dict_false(self, pipeline):
        cond = torch.randn(1, CHANNELS, IMG_SIZE, IMG_SIZE)
        result = pipeline(cond, num_inference_steps=3, output_type="pt", return_dict=False)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_pipeline_output_range(self, pipeline):
        cond = torch.randn(1, CHANNELS, IMG_SIZE, IMG_SIZE)
        result = pipeline(cond, num_inference_steps=3, output_type="pt")
        assert result.images.min() >= -1.0
        assert result.images.max() <= 1.0


# ---------------------------------------------------------------------------
# Trainer tests
# ---------------------------------------------------------------------------


class TestLocalDiffusionTrainer:
    """Tests for LocalDiffusionTrainer."""

    def test_trainer_init(self):
        from examples.local_diffusion.config import LocalDiffusionConfig
        from examples.local_diffusion.train_local_diffusion import LocalDiffusionTrainer

        config = LocalDiffusionConfig(
            dim=DIM, channels=CHANNELS, dim_mults=DIM_MULTS,
            resnet_block_groups=GROUPS, full_attn=FULL_ATTN,
            num_train_timesteps=T_STEPS, batch_size=1,
            device="cpu",
        )
        trainer = LocalDiffusionTrainer(config)
        assert trainer.model is not None
        assert trainer.scheduler is not None

    def test_trainer_train_step(self):
        from examples.local_diffusion.config import LocalDiffusionConfig
        from examples.local_diffusion.train_local_diffusion import LocalDiffusionTrainer

        config = LocalDiffusionConfig(
            dim=DIM, channels=CHANNELS, dim_mults=DIM_MULTS,
            resnet_block_groups=GROUPS, full_attn=FULL_ATTN,
            num_train_timesteps=T_STEPS, batch_size=1,
            resolution=IMG_SIZE, device="cpu",
        )
        trainer = LocalDiffusionTrainer(config)
        source = torch.rand(1, CHANNELS, IMG_SIZE, IMG_SIZE)
        target = torch.rand(1, CHANNELS, IMG_SIZE, IMG_SIZE)
        losses = trainer.train_step(source, target)
        assert "loss" in losses
        assert np.isfinite(losses["loss"])

    def test_trainer_config_validation(self):
        from examples.local_diffusion.config import LocalDiffusionConfig

        config = LocalDiffusionConfig()
        assert config.num_train_timesteps == 250
        assert config.beta_schedule == "sigmoid"
        assert config.objective == "pred_x0"
        assert config.branch_out is True
        assert config.fusion_timestep == 2
