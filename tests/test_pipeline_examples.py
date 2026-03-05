# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Tests for the self-contained pipeline examples.

Validates that each pipeline (DDBM, DDIB, BiBBDM, I2SB) can be instantiated
and run end-to-end with synthetic tensors — no real checkpoints needed.
"""

import pytest
import torch
import numpy as np


# ---------------------------------------------------------------------------
# DDBM pipeline tests
# ---------------------------------------------------------------------------


class TestDDBMPipeline:
    @pytest.fixture
    def pipeline(self):
        from examples.pipelines.ddbm.pipeline import (
            DDBMPipeline,
            DDBMUNet,
            DDBMScheduler,
        )

        unet = DDBMUNet(
            image_size=32,
            in_channels=3,
            model_channels=32,
            num_res_blocks=1,
            attention_resolutions=(),
            condition_mode="concat",
        )
        scheduler = DDBMScheduler(num_train_timesteps=40)
        return DDBMPipeline(unet=unet, scheduler=scheduler)

    def test_unet_forward(self):
        from examples.pipelines.ddbm.pipeline import DDBMUNet

        unet = DDBMUNet(
            image_size=32,
            in_channels=3,
            model_channels=32,
            num_res_blocks=1,
            attention_resolutions=(),
            condition_mode="concat",
        )
        x = torch.randn(2, 3, 32, 32)
        t = torch.tensor([100.0, 200.0])
        xT = torch.randn(2, 3, 32, 32)
        out = unet(x, t, xT=xT)
        assert out.shape == (2, 3, 32, 32)

    def test_scheduler_set_timesteps(self):
        from examples.pipelines.ddbm.pipeline import DDBMScheduler

        scheduler = DDBMScheduler(num_train_timesteps=40)
        scheduler.set_timesteps(10)
        assert scheduler.sigmas is not None
        assert len(scheduler.sigmas) == 11  # steps + 1 (appended zero)

    def test_pipeline_pt_output(self, pipeline):
        from examples.pipelines.ddbm.pipeline import DDBMPipelineOutput

        source = torch.randn(1, 3, 32, 32)
        result = pipeline(source, num_inference_steps=3, output_type="pt")
        assert isinstance(result, DDBMPipelineOutput)
        assert isinstance(result.images, torch.Tensor)
        assert result.images.shape[0] == 1
        assert result.nfe > 0

    def test_pipeline_pil_output(self, pipeline):
        source = torch.randn(1, 3, 32, 32)
        result = pipeline(source, num_inference_steps=3, output_type="pil")
        assert isinstance(result.images, list)
        assert len(result.images) == 1

    def test_pipeline_np_output(self, pipeline):
        source = torch.randn(1, 3, 32, 32)
        result = pipeline(source, num_inference_steps=3, output_type="np")
        assert isinstance(result.images, np.ndarray)


# ---------------------------------------------------------------------------
# DDIB pipeline tests
# ---------------------------------------------------------------------------


class TestDDIBPipeline:
    @pytest.fixture
    def pipeline(self):
        from examples.pipelines.ddib.pipeline import (
            DDIBPipeline,
            DDIBUNet,
            DDIBScheduler,
        )

        source_unet = DDIBUNet(
            image_size=32,
            in_channels=3,
            model_channels=32,
            num_res_blocks=1,
            attention_resolutions=(),
        )
        target_unet = DDIBUNet(
            image_size=32,
            in_channels=3,
            model_channels=32,
            num_res_blocks=1,
            attention_resolutions=(),
        )
        scheduler = DDIBScheduler(num_train_timesteps=100)
        return DDIBPipeline(
            source_unet=source_unet,
            target_unet=target_unet,
            scheduler=scheduler,
        )

    def test_unet_forward(self):
        from examples.pipelines.ddib.pipeline import DDIBUNet

        unet = DDIBUNet(
            image_size=32,
            in_channels=3,
            model_channels=32,
            num_res_blocks=1,
            attention_resolutions=(),
        )
        x = torch.randn(2, 3, 32, 32)
        t = torch.tensor([100.0, 200.0])
        out = unet(x, t)
        assert out.shape == (2, 3, 32, 32)

    def test_scheduler_set_timesteps(self):
        from examples.pipelines.ddib.pipeline import DDIBScheduler

        scheduler = DDIBScheduler(num_train_timesteps=100)
        scheduler.set_timesteps(10)
        assert scheduler.timesteps is not None
        assert len(scheduler.timesteps) == 10

    def test_scheduler_ddim_step(self):
        from examples.pipelines.ddib.pipeline import DDIBScheduler

        scheduler = DDIBScheduler(num_train_timesteps=100)
        scheduler.set_timesteps(10)
        sample = torch.randn(2, 3, 8, 8)
        model_output = torch.randn(2, 3, 8, 8)
        t = torch.tensor([50, 50])
        t_prev = torch.tensor([40, 40])
        result = scheduler.ddim_step(model_output, t, t_prev, sample)
        assert result.prev_sample.shape == sample.shape

    def test_pipeline_pt_output(self, pipeline):
        from examples.pipelines.ddib.pipeline import DDIBPipelineOutput

        source = torch.randn(1, 3, 32, 32)
        result = pipeline(source, num_inference_steps=5, output_type="pt")
        assert isinstance(result, DDIBPipelineOutput)
        assert isinstance(result.images, torch.Tensor)

    def test_pipeline_pil_output(self, pipeline):
        source = torch.randn(1, 3, 32, 32)
        result = pipeline(source, num_inference_steps=5, output_type="pil")
        assert isinstance(result.images, list)
        assert len(result.images) == 1

    def test_pipeline_np_output(self, pipeline):
        source = torch.randn(1, 3, 32, 32)
        result = pipeline(source, num_inference_steps=5, output_type="np")
        assert isinstance(result.images, np.ndarray)


# ---------------------------------------------------------------------------
# BiBBDM pipeline tests
# ---------------------------------------------------------------------------


class TestBiBBDMPipeline:
    @pytest.fixture
    def pipeline(self):
        from examples.pipelines.bibbdm.pipeline import (
            BiBBDMPipeline,
            BiBBDMUNet,
            BiBBDMScheduler,
        )

        unet = BiBBDMUNet(
            image_size=32,
            in_channels=3,
            model_channels=32,
            num_res_blocks=1,
            attention_resolutions=(),
            condition_mode="concat",
        )
        scheduler = BiBBDMScheduler(
            num_timesteps=100, sample_step=10, objective="noise",
        )
        return BiBBDMPipeline(unet=unet, scheduler=scheduler)

    def test_unet_forward(self):
        from examples.pipelines.bibbdm.pipeline import BiBBDMUNet

        unet = BiBBDMUNet(
            image_size=32,
            in_channels=3,
            model_channels=32,
            num_res_blocks=1,
            attention_resolutions=(),
            condition_mode="concat",
        )
        x = torch.randn(2, 3, 32, 32)
        t = torch.tensor([50, 80])
        ctx = torch.randn(2, 3, 32, 32)
        out = unet(x, t, context=ctx)
        assert out.shape == (2, 3, 32, 32)

    def test_scheduler_register_schedule(self):
        from examples.pipelines.bibbdm.pipeline import BiBBDMScheduler

        scheduler = BiBBDMScheduler(num_timesteps=100)
        assert scheduler.m_t is not None
        assert scheduler.variance_t is not None
        assert len(scheduler.m_t) == 100

    def test_scheduler_set_timesteps(self):
        from examples.pipelines.bibbdm.pipeline import BiBBDMScheduler

        scheduler = BiBBDMScheduler(num_timesteps=100, sample_step=10)
        scheduler.set_timesteps(20)
        assert scheduler.steps is not None

    def test_pipeline_b2a(self, pipeline):
        from examples.pipelines.bibbdm.pipeline import BiBBDMPipelineOutput

        source = torch.randn(1, 3, 32, 32)
        result = pipeline(source, direction="b2a", output_type="pt")
        assert isinstance(result, BiBBDMPipelineOutput)
        assert isinstance(result.images, torch.Tensor)

    def test_pipeline_a2b(self, pipeline):
        source = torch.randn(1, 3, 32, 32)
        result = pipeline(source, direction="a2b", output_type="pt")
        assert isinstance(result.images, torch.Tensor)

    def test_pipeline_pil_output(self, pipeline):
        source = torch.randn(1, 3, 32, 32)
        result = pipeline(source, direction="b2a", output_type="pil")
        assert isinstance(result.images, list)
        assert len(result.images) == 1

    def test_pipeline_invalid_direction(self, pipeline):
        source = torch.randn(1, 3, 32, 32)
        with pytest.raises(ValueError, match="Unknown direction"):
            pipeline(source, direction="invalid", output_type="pt")


# ---------------------------------------------------------------------------
# I2SB self-contained pipeline tests
# ---------------------------------------------------------------------------


class TestI2SBSelfContainedPipeline:
    @pytest.fixture
    def pipeline(self):
        from examples.pipelines.i2sb.pipeline import (
            I2SBPipeline,
            I2SBUNet,
            I2SBScheduler,
        )

        unet = I2SBUNet(
            image_size=32,
            in_channels=3,
            model_channels=32,
            num_res_blocks=1,
            attention_resolutions=(),
            condition_mode="concat",
        )
        scheduler = I2SBScheduler(interval=100, beta_max=0.3)
        return I2SBPipeline(unet=unet, scheduler=scheduler)

    def test_unet_forward(self):
        from examples.pipelines.i2sb.pipeline import I2SBUNet

        unet = I2SBUNet(
            image_size=32,
            in_channels=3,
            model_channels=32,
            num_res_blocks=1,
            attention_resolutions=(),
            condition_mode="concat",
        )
        x = torch.randn(2, 3, 32, 32)
        t = torch.tensor([0.5, 0.8])
        cond = torch.randn(2, 3, 32, 32)
        out = unet(x, t, cond=cond)
        assert out.shape == (2, 3, 32, 32)

    def test_scheduler_symmetric_betas(self):
        from examples.pipelines.i2sb.pipeline import I2SBScheduler

        scheduler = I2SBScheduler(interval=100, beta_max=0.3)
        half = scheduler.interval // 2
        betas = scheduler.betas.numpy()
        assert list(betas[:half]) == pytest.approx(list(betas[half:][::-1]), abs=1e-8)

    def test_scheduler_set_timesteps(self):
        from examples.pipelines.i2sb.pipeline import I2SBScheduler

        scheduler = I2SBScheduler(interval=100, beta_max=0.3)
        scheduler.set_timesteps(nfe=10)
        assert scheduler.timesteps is not None
        assert len(scheduler.timesteps) == 11  # nfe + 1

    def test_pipeline_pt_output(self, pipeline):
        from examples.pipelines.i2sb.pipeline import I2SBPipelineOutput

        source = torch.randn(1, 3, 32, 32)
        result = pipeline(source, nfe=3, output_type="pt")
        assert isinstance(result, I2SBPipelineOutput)
        assert isinstance(result.images, torch.Tensor)
        assert result.nfe > 0

    def test_pipeline_pil_output(self, pipeline):
        source = torch.randn(1, 3, 32, 32)
        result = pipeline(source, nfe=3, output_type="pil")
        assert isinstance(result.images, list)
        assert len(result.images) == 1

    def test_pipeline_np_output(self, pipeline):
        source = torch.randn(1, 3, 32, 32)
        result = pipeline(source, nfe=3, output_type="np")
        assert isinstance(result.images, np.ndarray)

    def test_pipeline_ot_ode_mode(self, pipeline):
        source = torch.randn(1, 3, 32, 32)
        result = pipeline(source, nfe=3, ot_ode=True, output_type="pt")
        assert isinstance(result.images, torch.Tensor)
