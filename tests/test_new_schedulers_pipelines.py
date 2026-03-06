# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Tests for newly integrated src/ schedulers and pipelines.

Validates that all new schedulers can be instantiated and configured,
and that all new pipelines can be imported. Pipeline end-to-end tests
require model wrappers compatible with each pipeline's UNet interface.
"""

import pytest
import torch
import numpy as np


# ---------------------------------------------------------------------------
# Scheduler instantiation & basic API tests
# ---------------------------------------------------------------------------


class TestDDBMScheduler:
    def test_instantiation(self):
        from src.schedulers.ddbm import DDBMScheduler
        s = DDBMScheduler()
        assert s.config.sigma_min == 0.002

    def test_set_timesteps(self):
        from src.schedulers.ddbm import DDBMScheduler
        s = DDBMScheduler(num_train_timesteps=40)
        s.set_timesteps(10)
        assert s.sigmas is not None
        assert len(s.sigmas) == 11  # steps + 1

    def test_vp_helpers(self):
        from src.schedulers.ddbm import DDBMScheduler
        s = DDBMScheduler(pred_mode="vp")
        t = torch.tensor(0.5)
        logsnr = s._vp_logsnr(t)
        logs = s._vp_logs(t)
        assert logsnr.shape == ()
        assert logs.shape == ()


class TestBiBBDMScheduler:
    def test_instantiation(self):
        from src.schedulers.bibbdm import BiBBDMScheduler
        s = BiBBDMScheduler(num_timesteps=100)
        assert s.m_t is not None
        assert len(s.m_t) == 100

    def test_variance_schedule(self):
        from src.schedulers.bibbdm import BiBBDMScheduler
        s = BiBBDMScheduler(num_timesteps=100)
        assert s.variance_t is not None
        assert len(s.variance_t) == 100
        assert (s.variance_t >= 0).all()

    def test_set_timesteps(self):
        from src.schedulers.bibbdm import BiBBDMScheduler
        s = BiBBDMScheduler(num_timesteps=100, sample_step=10)
        s.set_timesteps(20)
        assert s.steps is not None

    def test_add_noise(self):
        from src.schedulers.bibbdm import BiBBDMScheduler
        s = BiBBDMScheduler(num_timesteps=100)
        target = torch.randn(2, 3, 8, 8)
        source = torch.randn(2, 3, 8, 8)
        t = torch.tensor([10, 50])
        xt = s.add_noise(target, source, t)
        assert xt.shape == target.shape

    def test_step_b2a(self):
        from src.schedulers.bibbdm import BiBBDMScheduler
        s = BiBBDMScheduler(num_timesteps=100, sample_step=10, objective="noise")
        x_t = torch.randn(1, 3, 8, 8)
        source = torch.randn(1, 3, 8, 8)
        model_output = torch.randn(1, 3, 8, 8)
        result = s.step_b2a(model_output, step_index=0, x_t=x_t, source=source)
        assert result.prev_sample.shape == x_t.shape


class TestDDIBScheduler:
    def test_instantiation(self):
        from src.schedulers.ddib import DDIBScheduler
        s = DDIBScheduler(num_train_timesteps=100)
        assert s.num_train_timesteps == 100

    def test_set_timesteps(self):
        from src.schedulers.ddib import DDIBScheduler
        s = DDIBScheduler(num_train_timesteps=100)
        s.set_timesteps(10)
        assert s.timesteps is not None
        assert len(s.timesteps) == 10

    def test_ddim_step(self):
        from src.schedulers.ddib import DDIBScheduler
        s = DDIBScheduler(num_train_timesteps=100)
        s.set_timesteps(10)
        sample = torch.randn(2, 3, 8, 8)
        model_output = torch.randn(2, 3, 8, 8)
        t = torch.tensor([50, 50])
        t_prev = torch.tensor([40, 40])
        result = s.ddim_step(model_output, t, t_prev, sample)
        assert result.prev_sample.shape == sample.shape

    def test_ddim_reverse_step(self):
        from src.schedulers.ddib import DDIBScheduler
        s = DDIBScheduler(num_train_timesteps=100)
        s.set_timesteps(10)
        sample = torch.randn(2, 3, 8, 8)
        model_output = torch.randn(2, 3, 8, 8)
        t = torch.tensor([40, 40])
        t_next = torch.tensor([50, 50])
        result = s.ddim_reverse_step(model_output, t, t_next, sample)
        assert result.prev_sample.shape == sample.shape

    def test_add_noise(self):
        from src.schedulers.ddib import DDIBScheduler
        s = DDIBScheduler(num_train_timesteps=100)
        original = torch.randn(2, 3, 8, 8)
        noise = torch.randn(2, 3, 8, 8)
        t = torch.tensor([10, 50])
        noisy = s.add_noise(original, noise, t)
        assert noisy.shape == original.shape


class TestBDBMScheduler:
    def test_instantiation(self):
        from src.schedulers.bdbm import BDBMScheduler
        s = BDBMScheduler(num_timesteps=100)
        assert s.m_t is not None
        assert s.steps is not None
        assert s.asc_steps is not None

    def test_set_timesteps(self):
        from src.schedulers.bdbm import BDBMScheduler
        s = BDBMScheduler(num_timesteps=100, sample_step=10)
        s.set_timesteps(20)
        assert s.steps is not None

    def test_step_b2a(self):
        from src.schedulers.bdbm import BDBMScheduler
        s = BDBMScheduler(num_timesteps=100, sample_step=10)
        x_t = torch.randn(1, 3, 8, 8)
        source = torch.randn(1, 3, 8, 8)
        model_output = torch.randn(1, 3, 8, 8)
        result = s.step_b2a(model_output, step_index=0, x_t=x_t, source=source)
        assert result.prev_sample.shape == x_t.shape

    def test_add_noise(self):
        from src.schedulers.bdbm import BDBMScheduler
        s = BDBMScheduler(num_timesteps=100)
        target = torch.randn(2, 3, 8, 8)
        source = torch.randn(2, 3, 8, 8)
        t = torch.tensor([10, 50])
        xt = s.add_noise(target, source, t)
        assert xt.shape == target.shape


class TestDBIMScheduler:
    def test_instantiation(self):
        from src.schedulers.dbim import DBIMScheduler
        s = DBIMScheduler()
        assert s.sigma_min == 0.002

    def test_set_timesteps(self):
        from src.schedulers.dbim import DBIMScheduler
        s = DBIMScheduler()
        s.set_timesteps(10)
        assert s.sigmas is not None

    def test_get_abc(self):
        from src.schedulers.dbim import DBIMScheduler
        s = DBIMScheduler(pred_mode="vp")
        t = torch.tensor([0.1, 0.5])
        a, b, c = s.get_abc(t)
        assert a.shape == t.shape
        assert b.shape == t.shape
        assert c.shape == t.shape

    def test_step(self):
        from src.schedulers.dbim import DBIMScheduler
        s = DBIMScheduler()
        s.set_timesteps(10)
        x_t = torch.randn(1, 3, 8, 8)
        x_T = torch.randn(1, 3, 8, 8)
        model_output = torch.randn(1, 3, 8, 8)
        result = s.step(model_output, timestep=0, sample=x_t, x_T=x_T)
        assert result.prev_sample.shape == x_t.shape


class TestCDTSDEScheduler:
    def test_instantiation(self):
        from src.schedulers.cdtsde import CDTSDEScheduler
        s = CDTSDEScheduler()
        assert s.train_alphas_cumprod is not None

    def test_set_timesteps(self):
        from src.schedulers.cdtsde import CDTSDEScheduler
        s = CDTSDEScheduler(num_train_timesteps=100)
        s.set_timesteps(10)
        assert s.timesteps is not None
        assert s.sigmas is not None
        assert s.lambdas is not None

    def test_add_noise(self):
        from src.schedulers.cdtsde import CDTSDEScheduler
        s = CDTSDEScheduler()
        original = torch.randn(2, 3, 8, 8)
        noise = torch.randn(2, 3, 8, 8)
        t = torch.tensor([10, 50])
        noisy = s.add_noise(original, noise, t)
        assert noisy.shape == original.shape

    def test_predict_start_from_noise(self):
        from src.schedulers.cdtsde import CDTSDEScheduler
        s = CDTSDEScheduler()
        sample = torch.randn(2, 3, 8, 8)
        noise = torch.randn(2, 3, 8, 8)
        t = torch.tensor([10, 50])
        pred = s.predict_start_from_noise(sample, t, noise)
        assert pred.shape == sample.shape


# ---------------------------------------------------------------------------
# Pipeline import tests
# ---------------------------------------------------------------------------


class TestPipelineImports:
    """Verify that all pipelines can be imported from the public API."""

    def test_import_ddbm_pipeline(self):
        from src.pipelines import DDBMPipeline, DDBMPipelineOutput
        assert DDBMPipeline is not None
        assert DDBMPipelineOutput is not None

    def test_import_bibbdm_pipeline(self):
        from src.pipelines import BiBBDMPipeline, BiBBDMPipelineOutput
        assert BiBBDMPipeline is not None
        assert BiBBDMPipelineOutput is not None

    def test_import_ddib_pipeline(self):
        from src.pipelines import DDIBPipeline, DDIBPipelineOutput
        assert DDIBPipeline is not None
        assert DDIBPipelineOutput is not None

    def test_import_bdbm_pipeline(self):
        from src.pipelines import BDBMPipeline, BDBMPipelineOutput
        assert BDBMPipeline is not None
        assert BDBMPipelineOutput is not None

    def test_import_dbim_pipeline(self):
        from src.pipelines import DBIMPipeline, DBIMPipelineOutput
        assert DBIMPipeline is not None
        assert DBIMPipelineOutput is not None

    def test_import_cdtsde_pipeline(self):
        from src.pipelines import CDTSDEPipeline, CDTSDEPipelineOutput
        assert CDTSDEPipeline is not None
        assert CDTSDEPipelineOutput is not None

    def test_import_i2sb_pipeline(self):
        from src.pipelines import I2SBPipeline, I2SBPipelineOutput
        assert I2SBPipeline is not None
        assert I2SBPipelineOutput is not None


# ---------------------------------------------------------------------------
# Top-level src import tests
# ---------------------------------------------------------------------------


class TestTopLevelExports:
    """All schedulers/pipelines are accessible from ``src``."""

    def test_all_schedulers_in_src(self):
        import src
        for name in [
            "BDBMScheduler", "BiBBDMScheduler", "CDTSDEScheduler",
            "DBIMScheduler", "DDBMScheduler", "DDIBScheduler", "I2SBScheduler",
        ]:
            assert hasattr(src, name), f"src.{name} not found"

    def test_all_pipelines_in_src(self):
        import src
        for name in [
            "BDBMPipeline", "BiBBDMPipeline", "CDTSDEPipeline",
            "DBIMPipeline", "DDBMPipeline", "DDIBPipeline", "I2SBPipeline",
        ]:
            assert hasattr(src, name), f"src.{name} not found"
