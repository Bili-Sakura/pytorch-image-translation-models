# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Tests for the I2SB baseline module.

Validates config builders, model forward pass, scheduler operations,
and pipeline end-to-end (with synthetic tensors — no real checkpoints needed).
"""

import pytest
import torch

from examples.i2sb.config import (
    TaskConfig,
    sar2eo_config,
    rgb2ir_config,
    sar2ir_config,
    sar2rgb_config,
)
from src.models.unet import I2SBUNet
from src.models.unet.unet_2d import create_model
from src.schedulers import I2SBScheduler, I2SBSchedulerOutput
from src.pipelines.i2sb import I2SBPipeline, I2SBPipelineOutput


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfig:
    def test_default_config(self):
        cfg = TaskConfig()
        assert cfg.interval == 1000
        assert cfg.beta_max == 0.3
        assert cfg.condition_mode == "concat"
        assert cfg.use_ema is True
        assert cfg.push_to_hub is True

    @pytest.mark.parametrize("builder,task_name", [
        (sar2eo_config, "sar2eo"),
        (rgb2ir_config, "rgb2ir"),
        (sar2ir_config, "sar2ir"),
        (sar2rgb_config, "sar2rgb"),
    ])
    def test_task_builders(self, builder, task_name):
        cfg = builder()
        assert cfg.task_name == task_name

    def test_override(self):
        cfg = sar2eo_config(train_batch_size=64)
        assert cfg.train_batch_size == 64

    @pytest.mark.parametrize("builder", [
        sar2eo_config, rgb2ir_config, sar2ir_config, sar2rgb_config,
    ])
    def test_latent_paths(self, builder):
        cfg = builder()
        assert cfg.latent_vae_path is not None

    def test_sar2rgb_has_repa_path(self):
        """Only SAR2RGB has a REPA model path (MaRS-Base-RGB for target encoding)."""
        cfg = sar2rgb_config()
        assert cfg.rep_alignment_model_path == "./models/BiliSakura/MaRS-Base-RGB"

    @pytest.mark.parametrize("builder", [sar2eo_config, rgb2ir_config, sar2ir_config])
    def test_no_repa_path_for_unsupported_tasks(self, builder):
        """SAR2EO, RGB2IR, SAR2IR have no REPA model path."""
        cfg = builder()
        assert cfg.rep_alignment_model_path is None


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestModel:
    def test_create_small_model(self):
        model = create_model(
            image_size=32,
            in_channels=1,
            num_channels=32,
            num_res_blocks=1,
            attention_resolutions="",
            condition_mode="concat",
        )
        assert isinstance(model, I2SBUNet)

    def test_forward_unconditional(self):
        model = create_model(
            image_size=32,
            in_channels=1,
            num_channels=32,
            num_res_blocks=1,
            attention_resolutions="",
            condition_mode=None,
        )
        x = torch.randn(2, 1, 32, 32)
        t = torch.tensor([0.5, 0.8])
        out = model(x, t)
        assert out.shape == x.shape

    def test_forward_conditional(self):
        model = create_model(
            image_size=32,
            in_channels=1,
            num_channels=32,
            num_res_blocks=1,
            attention_resolutions="",
            condition_mode="concat",
        )
        x = torch.randn(2, 1, 32, 32)
        cond = torch.randn(2, 1, 32, 32)
        t = torch.tensor([0.5, 0.8])
        out = model(x, t, cond=cond)
        assert out.shape == x.shape


# ---------------------------------------------------------------------------
# Scheduler tests
# ---------------------------------------------------------------------------

class TestScheduler:
    @pytest.fixture
    def scheduler(self):
        return I2SBScheduler(interval=100, beta_max=0.3)

    def test_init(self, scheduler):
        assert scheduler.interval == 100
        assert len(scheduler.std_fwd) == 100
        assert len(scheduler.mu_x0) == 100

    def test_q_sample(self, scheduler):
        x0 = torch.randn(2, 1, 8, 8)
        x1 = torch.randn(2, 1, 8, 8)
        step = torch.tensor([10, 50])
        xt = scheduler.q_sample(step, x0, x1)
        assert xt.shape == x0.shape

    def test_q_sample_ot_ode(self, scheduler):
        x0 = torch.randn(2, 1, 8, 8)
        x1 = torch.randn(2, 1, 8, 8)
        step = torch.tensor([10, 50])
        xt = scheduler.q_sample(step, x0, x1, ot_ode=True)
        assert xt.shape == x0.shape

    def test_compute_label(self, scheduler):
        x0 = torch.randn(2, 1, 8, 8)
        x1 = torch.randn(2, 1, 8, 8)
        step = torch.tensor([10, 50])
        xt = scheduler.q_sample(step, x0, x1)
        label = scheduler.compute_label(step, x0, xt)
        assert label.shape == x0.shape

    def test_compute_pred_x0(self, scheduler):
        xt = torch.randn(2, 1, 8, 8)
        net_out = torch.randn(2, 1, 8, 8)
        step = torch.tensor([10, 50])
        pred_x0 = scheduler.compute_pred_x0(step, xt, net_out)
        assert pred_x0.shape == xt.shape

    def test_p_posterior(self, scheduler):
        x_n = torch.randn(2, 1, 8, 8)
        x0 = torch.randn(2, 1, 8, 8)
        prev_sample = scheduler.p_posterior(10, 50, x_n, x0)
        assert prev_sample.shape == x_n.shape

    def test_set_timesteps(self, scheduler):
        scheduler.set_timesteps(nfe=10)
        assert scheduler.timesteps is not None
        assert len(scheduler.timesteps) == 11  # nfe + 1

    def test_step(self, scheduler):
        sample = torch.randn(2, 1, 8, 8)
        model_output = torch.randn(2, 1, 8, 8)
        result = scheduler.step(model_output, 50, 10, sample)
        assert isinstance(result, I2SBSchedulerOutput)
        assert result.prev_sample.shape == sample.shape

    def test_symmetric_beta_schedule(self, scheduler):
        """Verify the beta schedule is symmetric (mirrored)."""
        half = scheduler.interval // 2
        betas = scheduler.betas.numpy()
        assert list(betas[:half]) == pytest.approx(list(betas[half:][::-1]), abs=1e-8)


# ---------------------------------------------------------------------------
# Pipeline tests
# ---------------------------------------------------------------------------

class TestPipeline:
    @pytest.fixture
    def pipeline(self):
        model = create_model(
            image_size=32,
            in_channels=1,
            num_channels=32,
            num_res_blocks=1,
            attention_resolutions="",
            condition_mode="concat",
        )
        scheduler = I2SBScheduler(interval=100, beta_max=0.3)
        return I2SBPipeline(unet=model, scheduler=scheduler)

    def test_pipeline_pt_output(self, pipeline):
        source = torch.randn(2, 1, 32, 32)
        result = pipeline(source, nfe=5, output_type="pt")
        assert isinstance(result, I2SBPipelineOutput)
        assert result.images.shape == source.shape
        assert result.nfe == 5

    def test_pipeline_pil_output(self, pipeline):
        source = torch.randn(1, 1, 32, 32)
        result = pipeline(source, nfe=3, output_type="pil")
        assert isinstance(result.images, list)
        assert len(result.images) == 1

    def test_pipeline_np_output(self, pipeline):
        source = torch.randn(1, 1, 32, 32)
        result = pipeline(source, nfe=3, output_type="np")
        import numpy as np
        assert isinstance(result.images, np.ndarray)


# ---------------------------------------------------------------------------
# Trainer signature tests
# ---------------------------------------------------------------------------

class TestTrainerSignature:
    """Test that I2SBTrainer methods exist with correct signatures.
    
    Note: Some tests require the ``datasets`` pip package which may not be
    installed in all environments.  Those tests are skipped gracefully.
    """

    @pytest.fixture(autouse=True)
    def _require_datasets(self):
        pytest.importorskip("datasets", reason="datasets package required")

    def test_trainer_instantiation(self):
        from examples.i2sb.trainer import I2SBTrainer
        cfg = sar2eo_config()
        trainer = I2SBTrainer(cfg)
        assert trainer.cfg is cfg

    def test_build_model(self):
        from examples.i2sb.trainer import I2SBTrainer
        cfg = sar2eo_config(resolution=32, num_channels=32, attention_resolutions="")
        trainer = I2SBTrainer(cfg)
        model = trainer.build_model()
        assert isinstance(model, I2SBUNet)

    def test_build_scheduler(self):
        from examples.i2sb.trainer import I2SBTrainer
        cfg = sar2eo_config()
        trainer = I2SBTrainer(cfg)
        scheduler = trainer.build_scheduler()
        assert isinstance(scheduler, I2SBScheduler)

    def test_compute_training_loss(self):
        from examples.i2sb.trainer import I2SBTrainer
        model = create_model(
            image_size=32,
            in_channels=1,
            num_channels=32,
            num_res_blocks=1,
            attention_resolutions="",
            condition_mode="concat",
        )
        scheduler = I2SBScheduler(interval=100, beta_max=0.3)
        x0 = torch.randn(2, 1, 32, 32)
        x_T = torch.randn(2, 1, 32, 32)
        loss = I2SBTrainer.compute_training_loss(model, scheduler, x0, x_T)
        assert loss.ndim == 0  # scalar loss
        assert loss.requires_grad
