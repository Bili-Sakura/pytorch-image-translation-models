# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Tests for DiT backbones (SiTBackbone, JiTBackbone), LBM scheduler, LBM pipeline,
and diffusers UNet wrappers (BDBM, DBIM, CDTSDE, LBM).

All components are imported from ``src/`` (no duplicate code in examples/).
"""

import pytest
import torch
import numpy as np


# ---------------------------------------------------------------------------
# SiTBackbone tests
# ---------------------------------------------------------------------------


class TestSiTBackbone:
    def test_output_shape_concat(self):
        from src.models.dit.sit import SiTBackbone

        model = SiTBackbone(
            image_size=16, patch_size=2, in_channels=3,
            hidden_size=64, depth=2, num_heads=4,
            condition_mode="concat",
        )
        x = torch.randn(2, 3, 16, 16)
        t = torch.tensor([0.5, 0.8])
        xT = torch.randn(2, 3, 16, 16)
        out = model(x, t, xT=xT)
        assert out.shape == (2, 3, 16, 16)

    def test_output_shape_unconditional(self):
        from src.models.dit.sit import SiTBackbone

        model = SiTBackbone(
            image_size=16, patch_size=2, in_channels=3,
            hidden_size=64, depth=2, num_heads=4,
            condition_mode=None,
        )
        x = torch.randn(1, 3, 16, 16)
        t = torch.tensor([1.0])
        out = model(x, t)
        assert out.shape == (1, 3, 16, 16)

    def test_sit_configs(self):
        from src.models.dit.sit import SIT_CONFIGS

        assert "S" in SIT_CONFIGS
        assert "B" in SIT_CONFIGS
        assert "L" in SIT_CONFIGS
        assert "XL" in SIT_CONFIGS
        depth, hidden, heads = SIT_CONFIGS["S"]
        assert depth == 12
        assert hidden == 384
        assert heads == 6

    def test_import_from_models(self):
        from src.models import SiTBackbone, SIT_CONFIGS
        assert SiTBackbone is not None
        assert SIT_CONFIGS is not None

    def test_import_from_top_level(self):
        import src
        assert hasattr(src, "SiTBackbone")
        assert hasattr(src, "SIT_CONFIGS")

    def test_different_channels(self):
        from src.models.dit.sit import SiTBackbone

        model = SiTBackbone(
            image_size=16, patch_size=2, in_channels=1,
            hidden_size=64, depth=2, num_heads=4,
            condition_mode="concat",
        )
        x = torch.randn(1, 1, 16, 16)
        t = torch.tensor([0.5])
        xT = torch.randn(1, 1, 16, 16)
        out = model(x, t, xT=xT)
        assert out.shape == (1, 1, 16, 16)


# ---------------------------------------------------------------------------
# JiTBackbone tests
# ---------------------------------------------------------------------------


class TestJiTBackbone:
    def test_output_shape_concat(self):
        from src.models.dit.jit import JiTBackbone

        model = JiTBackbone(
            image_size=16,
            patch_size=8,
            in_channels=3,
            hidden_size=64,
            depth=2,
            num_heads=4,
            bottleneck_dim=32,
            condition_mode="concat",
        )
        x = torch.randn(2, 3, 16, 16)
        t = torch.tensor([0.5, 0.8])
        xT = torch.randn(2, 3, 16, 16)
        out = model(x, t, xT=xT)
        assert out.shape == (2, 3, 16, 16)

    def test_output_shape_unconditional(self):
        from src.models.dit.jit import JiTBackbone

        model = JiTBackbone(
            image_size=16,
            patch_size=8,
            in_channels=3,
            hidden_size=64,
            depth=2,
            num_heads=4,
            bottleneck_dim=32,
            condition_mode=None,
        )
        x = torch.randn(1, 3, 16, 16)
        t = torch.tensor([1.0])
        out = model(x, t)
        assert out.shape == (1, 3, 16, 16)

    def test_jit_configs(self):
        from src.models.dit.jit import JIT_CONFIGS

        assert "B/16" in JIT_CONFIGS
        assert "L/16" in JIT_CONFIGS
        depth, hidden, heads, bottleneck, ps = JIT_CONFIGS["B/16"]
        assert depth == 12
        assert hidden == 768
        assert heads == 12
        assert bottleneck == 128
        assert ps == 16

    def test_import_from_models(self):
        from src.models import JIT_CONFIGS, JiTBackbone

        assert JiTBackbone is not None
        assert JIT_CONFIGS is not None

    def test_import_from_top_level(self):
        import src

        assert hasattr(src, "JiTBackbone")
        assert hasattr(src, "JIT_CONFIGS")


# ---------------------------------------------------------------------------
# LBM Scheduler tests
# ---------------------------------------------------------------------------


class TestLBMScheduler:
    def test_instantiation(self):
        from src.schedulers.lbm import LBMScheduler
        s = LBMScheduler()
        assert s.num_train_timesteps == 1000
        assert s.bridge_noise_sigma == 0.001

    def test_set_timesteps(self):
        from src.schedulers.lbm import LBMScheduler
        s = LBMScheduler()
        s.set_timesteps(5)
        assert s.num_inference_steps == 5
        assert len(s.timesteps) == 5
        assert len(s.sigmas) == 5

    def test_get_sigmas(self):
        from src.schedulers.lbm import LBMScheduler
        s = LBMScheduler(num_train_timesteps=100)
        t = torch.tensor([0, 50])
        sigmas = s.get_sigmas(t, n_dim=4, device="cpu")
        assert sigmas.shape == (2, 1, 1, 1)

    def test_add_noise(self):
        from src.schedulers.lbm import LBMScheduler
        s = LBMScheduler(num_train_timesteps=100)
        x_target = torch.randn(2, 3, 8, 8)
        x_source = torch.randn(2, 3, 8, 8)
        t = torch.tensor([10, 50])
        noisy = s.add_noise(x_target, x_source, t)
        assert noisy.shape == x_target.shape

    def test_step(self):
        from src.schedulers.lbm import LBMScheduler
        s = LBMScheduler(num_train_timesteps=100)
        s.set_timesteps(5)
        sample = torch.randn(1, 3, 8, 8)
        model_output = torch.randn(1, 3, 8, 8)
        t = s.timesteps[0]
        result = s.step(model_output, t, sample)
        assert result.prev_sample.shape == sample.shape
        assert result.pred_original_sample.shape == sample.shape

    def test_sample_timesteps_uniform(self):
        from src.schedulers.lbm import LBMScheduler
        s = LBMScheduler(num_train_timesteps=100)
        t = s.sample_timesteps(10)
        assert t.shape == (10,)
        assert t.dtype == torch.long

    def test_import_from_top_level(self):
        import src
        assert hasattr(src, "LBMScheduler")
        assert hasattr(src, "LBMSchedulerOutput")


# ---------------------------------------------------------------------------
# LBM Pipeline tests (src/)
# ---------------------------------------------------------------------------


class TestLBMPipelineImports:
    def test_import_from_pipelines(self):
        from src.pipelines import LBMPipeline, LBMPipelineOutput
        assert LBMPipeline is not None
        assert LBMPipelineOutput is not None

    def test_import_from_top_level(self):
        import src
        assert hasattr(src, "LBMPipeline")
        assert hasattr(src, "LBMPipelineOutput")


# ---------------------------------------------------------------------------
# UNet wrapper + pipeline tests: BDBM
# ---------------------------------------------------------------------------


class TestBDBMPipeline:
    @pytest.fixture
    def pipeline(self):
        from src.models.unet import BDBMUNet
        from src.schedulers.bdbm import BDBMScheduler
        from src.pipelines.bdbm import BDBMPipeline

        unet = BDBMUNet(
            image_size=32, in_channels=3, model_channels=32,
            num_res_blocks=1, attention_resolutions=(), condition_mode="concat",
        )
        scheduler = BDBMScheduler(num_timesteps=100, sample_step=10, objective="noise")
        return BDBMPipeline(unet=unet, scheduler=scheduler)

    def test_unet_forward(self):
        from src.models.unet import BDBMUNet

        unet = BDBMUNet(
            image_size=32, in_channels=3, model_channels=32,
            num_res_blocks=1, attention_resolutions=(), condition_mode="concat",
        )
        x = torch.randn(2, 3, 32, 32)
        t = torch.tensor([50, 80])
        ctx = torch.randn(2, 3, 32, 32)
        out = unet(x, t, context=ctx)
        assert out.shape == (2, 3, 32, 32)

    def test_pipeline_b2a(self, pipeline):
        source = torch.randn(1, 3, 32, 32)
        result = pipeline(source, direction="b2a", output_type="pt")
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
# UNet wrapper + pipeline tests: DBIM
# ---------------------------------------------------------------------------


class TestDBIMPipeline:
    @pytest.fixture
    def pipeline(self):
        from src.models.unet.adm import DBIMUNet
        from src.schedulers.dbim import DBIMScheduler
        from src.pipelines.dbim import DBIMPipeline

        unet = DBIMUNet(
            image_size=32, in_channels=3, model_channels=32,
            num_res_blocks=1, attention_resolutions=(), condition_mode="concat",
        )
        scheduler = DBIMScheduler()
        return DBIMPipeline(unet=unet, scheduler=scheduler)

    def test_unet_forward(self):
        from src.models.unet.adm import DBIMUNet

        unet = DBIMUNet(
            image_size=32, in_channels=3, model_channels=32,
            num_res_blocks=1, attention_resolutions=(), condition_mode="concat",
        )
        x = torch.randn(2, 3, 32, 32)
        t = torch.tensor([100.0, 200.0])
        xT = torch.randn(2, 3, 32, 32)
        out = unet(x, t, xT=xT)
        assert out.shape == (2, 3, 32, 32)

    def test_pipeline_pt_output(self, pipeline):
        source = torch.randn(1, 3, 32, 32)
        result = pipeline(source, num_inference_steps=3, output_type="pt")
        assert isinstance(result.images, torch.Tensor)
        assert result.nfe > 0

    def test_pipeline_pil_output(self, pipeline):
        source = torch.randn(1, 3, 32, 32)
        result = pipeline(source, num_inference_steps=3, output_type="pil")
        assert isinstance(result.images, list)
        assert len(result.images) == 1


# ---------------------------------------------------------------------------
# UNet wrapper + pipeline tests: CDTSDE
# ---------------------------------------------------------------------------


class TestCDTSDEPipeline:
    @pytest.fixture
    def pipeline(self):
        from src.models.unet import CDTSDEUNet
        from src.schedulers.cdtsde import CDTSDEScheduler
        from src.pipelines.cdtsde import CDTSDEPipeline

        unet = CDTSDEUNet(
            image_size=32, in_channels=3, model_channels=32,
            num_res_blocks=1, attention_resolutions=(), condition_mode="concat",
        )
        scheduler = CDTSDEScheduler(num_train_timesteps=100)
        return CDTSDEPipeline(unet=unet, scheduler=scheduler)

    def test_unet_forward(self):
        from src.models.unet import CDTSDEUNet

        unet = CDTSDEUNet(
            image_size=32, in_channels=3, model_channels=32,
            num_res_blocks=1, attention_resolutions=(), condition_mode="concat",
        )
        x = torch.randn(2, 3, 32, 32)
        t = torch.tensor([10, 50])
        xT = torch.randn(2, 3, 32, 32)
        out = unet(x, t, xT=xT)
        assert out.shape == (2, 3, 32, 32)

    def test_pipeline_pt_output(self, pipeline):
        source = torch.randn(1, 3, 32, 32)
        result = pipeline(source, num_inference_steps=5, output_type="pt")
        assert isinstance(result.images, torch.Tensor)
        assert result.nfe > 0

    def test_pipeline_pil_output(self, pipeline):
        source = torch.randn(1, 3, 32, 32)
        result = pipeline(source, num_inference_steps=5, output_type="pil")
        assert isinstance(result.images, list)
        assert len(result.images) == 1


# ---------------------------------------------------------------------------
# UNet wrapper + pipeline tests: LBM
# ---------------------------------------------------------------------------


class TestLBMPipeline:
    @pytest.fixture
    def pipeline(self):
        from src.models.unet import LBMUNet
        from src.schedulers.lbm import LBMScheduler
        from src.pipelines.lbm import LBMPipeline

        unet = LBMUNet(
            image_size=32, in_channels=3, model_channels=32,
            num_res_blocks=1, attention_resolutions=(), condition_mode="concat",
        )
        scheduler = LBMScheduler()
        return LBMPipeline(unet=unet, scheduler=scheduler)

    def test_unet_forward(self):
        from src.models.unet import LBMUNet

        unet = LBMUNet(
            image_size=32, in_channels=3, model_channels=32,
            num_res_blocks=1, attention_resolutions=(), condition_mode="concat",
        )
        x = torch.randn(2, 3, 32, 32)
        t = torch.tensor([0, 100])
        cond = torch.randn(2, 3, 32, 32)
        out = unet(x, t, cond=cond)
        assert out.shape == (2, 3, 32, 32)

    def test_scheduler_set_timesteps(self):
        from src.schedulers.lbm import LBMScheduler

        scheduler = LBMScheduler()
        scheduler.set_timesteps(5)
        assert len(scheduler.timesteps) == 5

    def test_pipeline_pt_output(self, pipeline):
        source = torch.randn(1, 3, 32, 32)
        result = pipeline(source, num_inference_steps=2, output_type="pt")
        assert isinstance(result.images, torch.Tensor)
        assert result.nfe > 0

    def test_pipeline_pil_output(self, pipeline):
        source = torch.randn(1, 3, 32, 32)
        result = pipeline(source, num_inference_steps=2, output_type="pil")
        assert isinstance(result.images, list)
        assert len(result.images) == 1

    def test_pipeline_np_output(self, pipeline):
        source = torch.randn(1, 3, 32, 32)
        result = pipeline(source, num_inference_steps=2, output_type="np")
        assert isinstance(result.images, np.ndarray)
