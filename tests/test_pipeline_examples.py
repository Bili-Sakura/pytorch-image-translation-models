# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Tests for the pipeline components (models, schedulers, pipelines).

Validates that each method (DDBM, DDIB, BiBBDM, I2SB) can be instantiated
and run end-to-end with synthetic tensors — no real checkpoints needed.

All components are imported from ``src/`` (no duplicate code in examples/).
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
        from src.models.unet.diffusers_wrappers import DDBMUNet
        from src.schedulers.ddbm import DDBMScheduler
        from src.pipelines.ddbm import DDBMPipeline

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
        from src.models.unet.diffusers_wrappers import DDBMUNet

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
        from src.schedulers.ddbm import DDBMScheduler

        scheduler = DDBMScheduler(num_train_timesteps=40)
        scheduler.set_timesteps(10)
        assert scheduler.sigmas is not None
        assert len(scheduler.sigmas) == 11  # steps + 1 (appended zero)

    def test_pipeline_pt_output(self, pipeline):
        from src.pipelines.ddbm import DDBMPipelineOutput

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
        from src.models.unet.diffusers_wrappers import DDIBUNet
        from src.schedulers.ddib import DDIBScheduler
        from src.pipelines.ddib import DDIBPipeline

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
        from src.models.unet.diffusers_wrappers import DDIBUNet

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
        from src.schedulers.ddib import DDIBScheduler

        scheduler = DDIBScheduler(num_train_timesteps=100)
        scheduler.set_timesteps(10)
        assert scheduler.timesteps is not None
        assert len(scheduler.timesteps) == 10

    def test_scheduler_ddim_step(self):
        from src.schedulers.ddib import DDIBScheduler

        scheduler = DDIBScheduler(num_train_timesteps=100)
        scheduler.set_timesteps(10)
        sample = torch.randn(2, 3, 8, 8)
        model_output = torch.randn(2, 3, 8, 8)
        t = torch.tensor([50, 50])
        t_prev = torch.tensor([40, 40])
        result = scheduler.ddim_step(model_output, t, t_prev, sample)
        assert result.prev_sample.shape == sample.shape

    def test_pipeline_pt_output(self, pipeline):
        from src.pipelines.ddib import DDIBPipelineOutput

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
        from src.models.unet.diffusers_wrappers import BiBBDMUNet
        from src.schedulers.bibbdm import BiBBDMScheduler
        from src.pipelines.bibbdm import BiBBDMPipeline

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
        from src.models.unet.diffusers_wrappers import BiBBDMUNet

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
        from src.schedulers.bibbdm import BiBBDMScheduler

        scheduler = BiBBDMScheduler(num_timesteps=100)
        assert scheduler.m_t is not None
        assert scheduler.variance_t is not None
        assert len(scheduler.m_t) == 100

    def test_scheduler_set_timesteps(self):
        from src.schedulers.bibbdm import BiBBDMScheduler

        scheduler = BiBBDMScheduler(num_timesteps=100, sample_step=10)
        scheduler.set_timesteps(20)
        assert scheduler.steps is not None

    def test_pipeline_b2a(self, pipeline):
        from src.pipelines.bibbdm import BiBBDMPipelineOutput

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
# I2SB pipeline tests
# ---------------------------------------------------------------------------


class TestI2SBPipeline:
    @pytest.fixture
    def pipeline(self):
        from src.models.unet.diffusers_wrappers import I2SBDiffusersUNet
        from src.schedulers.i2sb import I2SBScheduler
        from src.pipelines.i2sb import I2SBPipeline

        unet = I2SBDiffusersUNet(
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
        from src.models.unet.diffusers_wrappers import I2SBDiffusersUNet

        unet = I2SBDiffusersUNet(
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
        from src.schedulers.i2sb import I2SBScheduler

        scheduler = I2SBScheduler(interval=100, beta_max=0.3)
        half = scheduler.interval // 2
        betas = scheduler.betas.numpy()
        assert list(betas[:half]) == pytest.approx(list(betas[half:][::-1]), abs=1e-8)

    def test_scheduler_set_timesteps(self):
        from src.schedulers.i2sb import I2SBScheduler

        scheduler = I2SBScheduler(interval=100, beta_max=0.3)
        scheduler.set_timesteps(nfe=10)
        assert scheduler.timesteps is not None
        assert len(scheduler.timesteps) == 11  # nfe + 1

    def test_pipeline_pt_output(self, pipeline):
        from src.pipelines.i2sb import I2SBPipelineOutput

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

    def test_pipeline_to_cpu(self, pipeline):
        pipeline.to("cpu")
        source = torch.randn(1, 3, 32, 32)
        result = pipeline(source, nfe=3, output_type="pt")
        assert isinstance(result.images, torch.Tensor)


# ---------------------------------------------------------------------------
# pix2pixHD baseline tests
# ---------------------------------------------------------------------------


class TestPix2PixHDBaseline:
    @pytest.fixture
    def pipeline(self):
        from src.models.pix2pixhd import Pix2PixHDGenerator
        from src.pipelines.pix2pixhd import Pix2PixHDPipeline

        gen = Pix2PixHDGenerator(input_nc=3, output_nc=3, ngf=32, n_downsampling=2, n_blocks=2)
        return Pix2PixHDPipeline(generator=gen)

    def test_generator_output_shape(self):
        from src.models.pix2pixhd import Pix2PixHDGenerator

        gen = Pix2PixHDGenerator(input_nc=3, output_nc=3, ngf=32, n_downsampling=2, n_blocks=2)
        x = torch.randn(1, 3, 64, 64)
        out = gen(x)
        assert out.shape == (1, 3, 64, 64)
        assert out.min() >= -1.0
        assert out.max() <= 1.0

    def test_pipeline_pt_output(self, pipeline):
        from src.pipelines.pix2pixhd import Pix2PixHDPipelineOutput

        x = torch.randn(1, 3, 64, 64)
        out = pipeline(source_image=x, output_type="pt")
        assert isinstance(out, Pix2PixHDPipelineOutput)
        assert isinstance(out.images, torch.Tensor)
        assert out.images.shape == (1, 3, 64, 64)

    def test_pipeline_np_output(self, pipeline):
        x = torch.randn(1, 3, 64, 64)
        out = pipeline(source_image=x, output_type="np")
        assert isinstance(out.images, np.ndarray)
        assert out.images.shape == (1, 64, 64, 3)
        assert out.images.min() >= 0.0
        assert out.images.max() <= 1.0

    def test_pipeline_pil_output(self, pipeline):
        from PIL import Image

        x = torch.randn(1, 3, 64, 64)
        out = pipeline(source_image=x, output_type="pil")
        assert isinstance(out.images, list)
        assert isinstance(out.images[0], Image.Image)
        assert out.images[0].size == (64, 64)

    def test_loader_from_checkpoint(self, tmp_path):
        from src.models.pix2pixhd import Pix2PixHDGenerator
        from src.pipelines.pix2pixhd import load_pix2pixhd_pipeline

        gen = Pix2PixHDGenerator(input_nc=3, output_nc=3, ngf=16, n_downsampling=2, n_blocks=2)
        ckpt_path = tmp_path / "latest_net_G.pth"
        torch.save(gen.state_dict(), ckpt_path)

        pipe = load_pix2pixhd_pipeline(
            ckpt_path,
            input_nc=3,
            output_nc=3,
            ngf=16,
            n_downsampling=2,
            n_blocks=2,
            device="cpu",
        )
        x = torch.randn(1, 3, 64, 64)
        out = pipe(source_image=x, output_type="pt")
        assert isinstance(out.images, torch.Tensor)
        assert out.images.shape == (1, 3, 64, 64)


# ---------------------------------------------------------------------------
# StarGAN baseline tests
# ---------------------------------------------------------------------------


class TestStarGANBaseline:
    @pytest.fixture
    def pipeline(self):
        from src.models.stargan import StarGANGenerator
        from src.pipelines.stargan import StarGANPipeline

        gen = StarGANGenerator(conv_dim=32, c_dim=5, repeat_num=2)
        return StarGANPipeline(generator=gen)

    def test_generator_output_shape(self):
        from src.models.stargan import StarGANGenerator

        gen = StarGANGenerator(conv_dim=32, c_dim=5, repeat_num=2)
        x = torch.randn(2, 3, 64, 64)
        labels = torch.randn(2, 5)
        out = gen(x, labels)
        assert out.shape == (2, 3, 64, 64)
        assert out.min() >= -1.0
        assert out.max() <= 1.0

    def test_discriminator_output_shape(self):
        from src.models.stargan import StarGANDiscriminator

        disc = StarGANDiscriminator(image_size=64, conv_dim=32, c_dim=5, repeat_num=4)
        x = torch.randn(2, 3, 64, 64)
        out_src, out_cls = disc(x)
        assert out_src.shape[0] == 2
        assert out_src.shape[1] == 1
        assert out_cls.shape == (2, 5)

    def test_pipeline_pt_output(self, pipeline):
        from src.pipelines.stargan import StarGANPipelineOutput

        x = torch.randn(1, 3, 64, 64)
        labels = torch.tensor([[1, 0, 0, 1, 0]], dtype=torch.float32)
        out = pipeline(source_image=x, target_labels=labels, output_type="pt")
        assert isinstance(out, StarGANPipelineOutput)
        assert isinstance(out.images, torch.Tensor)
        assert out.images.shape == (1, 3, 64, 64)

    def test_pipeline_np_output(self, pipeline):
        x = torch.randn(1, 3, 64, 64)
        labels = torch.tensor([[0, 1, 1, 0, 1]], dtype=torch.float32)
        out = pipeline(source_image=x, target_labels=labels, output_type="np")
        assert isinstance(out.images, np.ndarray)
        assert out.images.shape == (1, 64, 64, 3)
        assert out.images.min() >= 0.0
        assert out.images.max() <= 1.0

    def test_pipeline_pil_output(self, pipeline):
        from PIL import Image

        x = torch.randn(1, 3, 64, 64)
        labels = torch.tensor([[1, 1, 0, 0, 1]], dtype=torch.float32)
        out = pipeline(source_image=x, target_labels=labels, output_type="pil")
        assert isinstance(out.images, list)
        assert isinstance(out.images[0], Image.Image)
        assert out.images[0].size == (64, 64)
