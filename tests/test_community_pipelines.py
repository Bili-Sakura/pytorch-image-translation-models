# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Tests for community pipelines."""

import sys

import pytest
import torch


# ---------------------------------------------------------------------------
# Parallel-GAN
# ---------------------------------------------------------------------------


class TestParaGAN:
    """Tests for the Parallel-GAN translation generator."""

    def test_output_shape_3ch(self):
        from examples.community.parallel_gan import ParaGAN

        gen = ParaGAN(input_nc=3, output_nc=3, n_blocks=2)
        x = torch.randn(1, 3, 256, 256)
        out = gen(x)
        assert isinstance(out, list)
        assert len(out) == 6
        assert out[-1].shape == (1, 3, 256, 256)

    def test_output_range(self):
        from examples.community.parallel_gan import ParaGAN

        gen = ParaGAN(input_nc=3, output_nc=3, n_blocks=2)
        x = torch.randn(1, 3, 256, 256)
        out = gen(x)
        rgb = out[-1]
        assert rgb.min() >= -1.0
        assert rgb.max() <= 1.0

    def test_intermediate_features(self):
        from examples.community.parallel_gan import ParaGAN

        gen = ParaGAN(input_nc=3, output_nc=3, n_blocks=2)
        x = torch.randn(1, 3, 256, 256)
        feats = gen(x)
        # 5 intermediate + 1 final = 6
        assert len(feats) == 6
        # Each intermediate feature should be a 4D tensor
        for f in feats:
            assert f.dim() == 4


class TestResrecon:
    """Tests for the reconstruction network."""

    def test_output_shape(self):
        from examples.community.parallel_gan import Resrecon

        net = Resrecon()
        x = torch.randn(1, 3, 256, 256)
        out = net(x)
        assert isinstance(out, list)
        assert len(out) == 6
        assert out[-1].shape == (1, 3, 256, 256)

    def test_output_range(self):
        from examples.community.parallel_gan import Resrecon

        net = Resrecon()
        x = torch.randn(1, 3, 256, 256)
        out = net(x)
        rgb = out[-1]
        assert rgb.min() >= -1.0
        assert rgb.max() <= 1.0


class TestParallelGANTrainer:
    """Tests for the Parallel-GAN trainer."""

    @pytest.fixture
    def trainer_recon(self):
        """Trainer without recon_net (Stage 1 mode)."""
        from examples.community.parallel_gan import ParallelGANConfig, ParallelGANTrainer

        cfg = ParallelGANConfig(input_nc=3, output_nc=3, n_blocks=2, device="cpu")
        return ParallelGANTrainer(cfg)

    @pytest.fixture
    def trainer_trans(self):
        """Trainer with recon_net (Stage 2 mode)."""
        from examples.community.parallel_gan import (
            ParallelGANConfig,
            ParallelGANTrainer,
            Resrecon,
        )

        cfg = ParallelGANConfig(input_nc=3, output_nc=3, n_blocks=2, device="cpu")
        recon = Resrecon()
        return ParallelGANTrainer(cfg, recon_net=recon)

    def test_instantiation(self, trainer_recon):
        assert trainer_recon.netG is not None
        assert trainer_recon.netD is not None
        assert trainer_recon.recon_net is None

    def test_train_step_recon(self, trainer_recon):
        real_B = torch.randn(1, 3, 256, 256)
        losses = trainer_recon.train_step_recon(real_B)
        assert "D" in losses
        assert "G_GAN" in losses
        assert "G_L1" in losses
        assert "VGG" in losses
        assert isinstance(losses["D"], float)

    def test_train_step_auto_selects_recon(self, trainer_recon):
        """train_step without recon_net should use Stage 1."""
        real_A = torch.randn(1, 3, 256, 256)
        real_B = torch.randn(1, 3, 256, 256)
        losses = trainer_recon.train_step(real_A, real_B)
        assert "VGG" in losses  # Stage 1 signature

    def test_train_step_trans(self, trainer_trans):
        real_A = torch.randn(1, 3, 256, 256)
        real_B = torch.randn(1, 3, 256, 256)
        losses = trainer_trans.train_step_trans(real_A, real_B)
        assert "D" in losses
        assert "G_GAN" in losses
        assert "G_L1" in losses
        assert "feat" in losses
        assert isinstance(losses["feat"], float)

    def test_train_step_auto_selects_trans(self, trainer_trans):
        """train_step with recon_net should use Stage 2."""
        real_A = torch.randn(1, 3, 256, 256)
        real_B = torch.randn(1, 3, 256, 256)
        losses = trainer_trans.train_step(real_A, real_B)
        assert "feat" in losses  # Stage 2 signature

    def test_trans_requires_recon_net(self, trainer_recon):
        """Stage 2 should raise if no recon_net."""
        real_A = torch.randn(1, 3, 256, 256)
        real_B = torch.randn(1, 3, 256, 256)
        with pytest.raises(RuntimeError, match="reconstruction network"):
            trainer_recon.train_step_trans(real_A, real_B)

    def test_config_defaults(self):
        from examples.community.parallel_gan import ParallelGANConfig

        cfg = ParallelGANConfig()
        assert cfg.input_nc == 3
        assert cfg.lambda_L1 == 100.0
        assert cfg.lambda_vgg == 10.0
        assert cfg.lambda_feat == 10.0
        assert cfg.n_blocks == 6


class TestVGGLoss:
    def test_vgg_loss_scalar(self):
        from examples.community.parallel_gan import VGGLoss

        loss_fn = VGGLoss()
        x = torch.randn(1, 3, 64, 64)
        y = torch.randn(1, 3, 64, 64)
        loss = loss_fn(x, y)
        assert loss.dim() == 0
        assert loss.item() >= 0.0


class TestGANLoss:
    def test_vanilla_mode(self):
        from examples.community.parallel_gan import _GANLoss

        loss_fn = _GANLoss("vanilla")
        pred = torch.randn(2, 1, 8, 8)
        loss = loss_fn(pred, target_is_real=True)
        assert loss.dim() == 0

    def test_lsgan_mode(self):
        from examples.community.parallel_gan import _GANLoss

        loss_fn = _GANLoss("lsgan")
        pred = torch.randn(2, 1, 8, 8)
        loss = loss_fn(pred, target_is_real=False)
        assert loss.dim() == 0
        assert loss.item() >= 0.0


# ---------------------------------------------------------------------------
# E3Diff
# ---------------------------------------------------------------------------

# Shared small configuration used across all E3Diff tests to keep them fast.
_E3DIFF_KWARGS = dict(
    condition_ch=3,
    out_ch=3,
    image_size=64,
    inner_channel=16,
    channel_mults=(1, 2, 4, 8, 16),
    res_blocks=1,
    n_timestep=10,
    device="cpu",
)


class TestCPEN:
    """Tests for the Conditional Prior Enhancement Network."""

    def test_output_shapes(self):
        from examples.community.e3diff import CPEN

        cpen = CPEN(in_channel=3, base_ch=16)
        x = torch.randn(1, 3, 64, 64)
        c1, c2, c3, c4, c5 = cpen(x)
        assert c1.shape == (1, 16, 64, 64)
        assert c2.shape == (1, 32, 32, 32)
        assert c3.shape == (1, 64, 16, 16)
        assert c4.shape == (1, 128, 8, 8)
        assert c5.shape == (1, 256, 4, 4)

    def test_single_channel_input(self):
        """CPEN should accept single-channel (SAR) input."""
        from examples.community.e3diff import CPEN

        cpen = CPEN(in_channel=1, base_ch=16)
        x = torch.randn(2, 1, 64, 64)
        c1, c2, c3, c4, c5 = cpen(x)
        assert c1.shape[0] == 2


class TestE3DiffUNet:
    """Tests for the E3Diff denoising U-Net."""

    def test_output_shape(self):
        from examples.community.e3diff import E3DiffUNet

        unet = E3DiffUNet(
            out_channel=3,
            inner_channel=16,
            norm_groups=16,
            channel_mults=(1, 2, 4, 8, 16),
            res_blocks=1,
            image_size=64,
            condition_ch=3,
        )
        # Input: condition (3ch) concatenated with noisy target (3ch)
        x = torch.randn(1, 6, 64, 64)
        noise_level = torch.rand(1, 1)
        out = unet(x, noise_level)
        assert out.shape == (1, 3, 64, 64)

    def test_single_channel_condition(self):
        """UNet should accept single-channel SAR conditioning."""
        from examples.community.e3diff import E3DiffUNet

        unet = E3DiffUNet(
            out_channel=3,
            inner_channel=16,
            norm_groups=16,
            channel_mults=(1, 2, 4, 8, 16),
            res_blocks=1,
            image_size=64,
            condition_ch=1,
        )
        x = torch.randn(1, 4, 64, 64)  # 1 + 3 = 4 channels
        noise_level = torch.rand(1, 1)
        out = unet(x, noise_level)
        assert out.shape == (1, 3, 64, 64)

    def test_invalid_channel_mults(self):
        """E3DiffUNet should raise if channel_mults does not have 5 entries."""
        from examples.community.e3diff import E3DiffUNet

        with pytest.raises(ValueError, match="exactly 5"):
            E3DiffUNet(channel_mults=(1, 2, 4, 8))


class TestGaussianDiffusion:
    """Tests for the GaussianDiffusion wrapper."""

    @pytest.fixture
    def diffusion(self):
        from examples.community.e3diff import E3DiffUNet, GaussianDiffusion

        unet = E3DiffUNet(
            out_channel=3,
            inner_channel=16,
            norm_groups=16,
            channel_mults=(1, 2, 4, 8, 16),
            res_blocks=1,
            image_size=64,
            condition_ch=3,
        )
        diff = GaussianDiffusion(denoise_fn=unet, image_size=64, channels=3, xT_noise_r=0.1)
        diff.set_noise_schedule(n_timestep=10, schedule="linear", device="cpu")
        return diff

    def test_stage1_forward(self, diffusion):
        data = {"HR": torch.randn(1, 3, 64, 64), "SR": torch.randn(1, 3, 64, 64)}
        l_pix, x_start, x_pred = diffusion(data, stage=1)
        assert l_pix.dim() == 0
        assert x_start.shape == (1, 3, 64, 64)
        assert x_pred.shape == (1, 3, 64, 64)
        assert l_pix.item() >= 0.0

    def test_stage2_forward(self, diffusion):
        data = {"HR": torch.randn(1, 3, 64, 64), "SR": torch.randn(1, 3, 64, 64)}
        l_pix, x_start, x_pred = diffusion(data, stage=2)
        assert x_pred.shape == (1, 3, 64, 64)

    def test_sample_shape(self, diffusion):
        condition = torch.randn(1, 3, 64, 64)
        out = diffusion.sample(condition, n_ddim_steps=2)
        assert out.shape == (1, 3, 64, 64)


class TestFocalFrequencyLoss:
    """Tests for the Focal Frequency Loss."""

    def test_output_is_scalar(self):
        from examples.community.e3diff import FocalFrequencyLoss

        loss_fn = FocalFrequencyLoss(loss_weight=1.0)
        pred = torch.randn(2, 3, 64, 64)
        target = torch.randn(2, 3, 64, 64)
        loss = loss_fn(pred, target)
        assert loss.dim() == 0

    def test_identical_inputs_low_loss(self):
        """Loss of identical pred and target should be near zero."""
        from examples.community.e3diff import FocalFrequencyLoss

        loss_fn = FocalFrequencyLoss(loss_weight=1.0)
        x = torch.randn(2, 3, 64, 64)
        loss = loss_fn(x, x)
        assert loss.item() < 1e-4

    def test_loss_weight(self):
        from examples.community.e3diff import FocalFrequencyLoss

        pred = torch.randn(1, 3, 64, 64)
        target = torch.randn(1, 3, 64, 64)
        loss1 = FocalFrequencyLoss(loss_weight=1.0)(pred, target)
        loss2 = FocalFrequencyLoss(loss_weight=2.0)(pred, target)
        assert abs(loss2.item() - 2 * loss1.item()) < 1e-5


class TestE3DiffTrainer:
    """Tests for the two-stage E3Diff trainer."""

    @pytest.fixture
    def trainer_s1(self):
        from examples.community.e3diff import E3DiffConfig, E3DiffTrainer

        cfg = E3DiffConfig(stage=1, **_E3DIFF_KWARGS)
        return E3DiffTrainer(cfg)

    @pytest.fixture
    def trainer_s2(self):
        from examples.community.e3diff import E3DiffConfig, E3DiffTrainer

        cfg = E3DiffConfig(stage=2, lambda_gan=0.1, **_E3DIFF_KWARGS)
        return E3DiffTrainer(cfg)

    def test_stage1_train_step_keys(self, trainer_s1):
        sar = torch.randn(1, 3, 64, 64)
        opt = torch.randn(1, 3, 64, 64)
        losses = trainer_s1.train_step(sar, opt)
        assert "l_pix" in losses
        assert isinstance(losses["l_pix"], float)

    def test_stage1_with_fft_loss(self):
        from examples.community.e3diff import E3DiffConfig, E3DiffTrainer

        cfg = E3DiffConfig(stage=1, fft_weight=1.0, **_E3DIFF_KWARGS)
        trainer = E3DiffTrainer(cfg)
        sar = torch.randn(1, 3, 64, 64)
        opt = torch.randn(1, 3, 64, 64)
        losses = trainer.train_step(sar, opt)
        assert "l_pix" in losses
        assert "l_freq" in losses

    def test_stage2_train_step_keys(self, trainer_s2):
        sar = torch.randn(1, 3, 64, 64)
        opt = torch.randn(1, 3, 64, 64)
        losses = trainer_s2.train_step(sar, opt)
        assert "l_pix" in losses
        assert "l_G" in losses
        assert "l_D" in losses
        assert isinstance(losses["l_D"], float)

    def test_stage2_requires_discriminator(self):
        """Calling train_step_stage2 without stage=2 config should raise."""
        from examples.community.e3diff import E3DiffConfig, E3DiffTrainer

        cfg = E3DiffConfig(stage=1, **_E3DIFF_KWARGS)
        trainer = E3DiffTrainer(cfg)
        sar = torch.randn(1, 3, 64, 64)
        opt = torch.randn(1, 3, 64, 64)
        with pytest.raises(RuntimeError, match="stage=2"):
            trainer.train_step_stage2(sar, opt)

    def test_sample_shape(self, trainer_s1):
        sar = torch.randn(1, 3, 64, 64)
        pred = trainer_s1.sample(sar, n_ddim_steps=2)
        assert pred.shape == (1, 3, 64, 64)

    def test_sample_range(self, trainer_s1):
        """Sampled output should be clipped to [-1, 1]."""
        sar = torch.randn(1, 3, 64, 64)
        pred = trainer_s1.sample(sar, n_ddim_steps=2)
        assert pred.min().item() >= -1.1  # slight tolerance for float precision
        assert pred.max().item() <= 1.1

    def test_config_defaults(self):
        from examples.community.e3diff import E3DiffConfig

        cfg = E3DiffConfig()
        assert cfg.stage == 1
        assert cfg.condition_ch == 3
        assert cfg.out_ch == 3
        assert cfg.n_timestep == 1000
        assert cfg.channel_mults == (1, 2, 4, 8, 16)

    def test_instantiation_stage1(self, trainer_s1):
        assert trainer_s1.diffusion is not None
        assert trainer_s1.netD is None  # no discriminator in Stage 1

    def test_instantiation_stage2(self, trainer_s2):
        assert trainer_s2.diffusion is not None
        assert trainer_s2.netD is not None


# ---------------------------------------------------------------------------
# OpenEarthMap-SAR
# ---------------------------------------------------------------------------


class TestOpenEarthMapSARGenerator:
    """Tests for the OpenEarthMap-SAR CUT generator."""

    def test_output_shape(self):
        from examples.community.openearthmap_sar import OpenEarthMapSARGenerator

        gen = OpenEarthMapSARGenerator(in_channels=3, out_channels=3, base_filters=64, n_blocks=2)
        x = torch.randn(1, 3, 256, 256)
        out = gen(x)
        assert out.shape == (1, 3, 256, 256)

    def test_output_range(self):
        from examples.community.openearthmap_sar import OpenEarthMapSARGenerator

        gen = OpenEarthMapSARGenerator(in_channels=3, out_channels=3, base_filters=64, n_blocks=2)
        x = torch.randn(1, 3, 256, 256)
        out = gen(x)
        assert out.min() >= -1.0
        assert out.max() <= 1.0

    def test_antialias_variants(self):
        """Generator should work with both antialias and no-antialias configs."""
        from examples.community.openearthmap_sar import OpenEarthMapSARGenerator

        gen_aa = OpenEarthMapSARGenerator(no_antialias=False, no_antialias_up=False, n_blocks=2)
        gen_no_aa = OpenEarthMapSARGenerator(no_antialias=True, no_antialias_up=True, n_blocks=2)
        x = torch.randn(1, 3, 64, 64)
        out_aa = gen_aa(x)
        out_no_aa = gen_no_aa(x)
        assert out_aa.shape == out_no_aa.shape == (1, 3, 64, 64)


# ---------------------------------------------------------------------------
# SAR2Optical
# ---------------------------------------------------------------------------


class TestSAR2OpticalGenerator:
    """Tests for SAR2Optical pix2pix generator."""

    def test_output_shape(self):
        from examples.community.sar2optical import SAR2OpticalGenerator

        gen = SAR2OpticalGenerator(c_in=3, c_out=3)
        x = torch.randn(2, 3, 256, 256)
        out = gen(x)
        assert out.shape == (2, 3, 256, 256)

    def test_output_range(self):
        from examples.community.sar2optical import SAR2OpticalGenerator

        gen = SAR2OpticalGenerator(c_in=3, c_out=3)
        x = torch.randn(2, 3, 256, 256)
        out = gen(x)
        assert out.min() >= -1.0
        assert out.max() <= 1.0


class TestSAR2OpticalTrainer:
    """Tests for SAR2Optical trainer."""

    def test_train_step_keys(self):
        from examples.community.sar2optical import SAR2OpticalConfig, SAR2OpticalTrainer

        trainer = SAR2OpticalTrainer(SAR2OpticalConfig(c_in=3, c_out=3, device="cpu"))
        x = torch.randn(2, 3, 256, 256)
        y = torch.randn(2, 3, 256, 256)
        losses = trainer.train_step(x, y)
        assert "loss_D" in losses
        assert "loss_G" in losses
        assert "loss_G_GAN" in losses
        assert "loss_G_L1" in losses


# ---------------------------------------------------------------------------
# CycleGAN (junyanz/pytorch-CycleGAN-and-pix2pix)
# ---------------------------------------------------------------------------


class TestCycleGANGenerator:
    def test_output_shape_and_range(self):
        from examples.community.cyclegan import create_generator

        gen = create_generator(netG="resnet_6blocks", norm="instance")
        x = torch.randn(2, 3, 64, 64)
        out = gen(x)
        assert out.shape == (2, 3, 64, 64)
        assert out.min() >= -1.0
        assert out.max() <= 1.0


class TestCycleGANTrainer:
    def test_train_step_keys(self):
        from examples.community.cyclegan import CycleGANConfig, CycleGANTrainer

        cfg = CycleGANConfig(resolution=64, lambda_identity=0.0, device="cpu")
        trainer = CycleGANTrainer(cfg)
        a = torch.rand(1, 3, 64, 64)
        b = torch.rand(1, 3, 64, 64)
        losses = trainer.train_step(a, b)
        assert "loss_G" in losses
        assert "loss_D_A" in losses
        assert "loss_D_B" in losses


class TestCycleGANPipeline:
    def test_forward_a2b(self):
        from examples.community.cyclegan import create_generator
        from src.pipelines.cyclegan import CycleGANPipeline

        gen = create_generator(netG="resnet_9blocks", norm="instance")
        pipe = CycleGANPipeline(generator_a=gen)
        x = torch.rand(1, 3, 64, 64) * 2 - 1
        out = pipe(source_image=x, output_type="pt")
        assert out.images.shape == (1, 3, 64, 64)


# ---------------------------------------------------------------------------
# pix2pix (junyanz/pytorch-CycleGAN-and-pix2pix)
# ---------------------------------------------------------------------------


class TestPix2PixCommunityGenerator:
    def test_output_shape_and_range(self):
        from examples.community.pix2pix import create_generator

        gen = create_generator(netG="unet_256", norm="batch", use_dropout=True)
        x = torch.randn(2, 3, 64, 64)
        out = gen(x)
        assert out.shape == (2, 3, 64, 64)
        assert out.min() >= -1.0
        assert out.max() <= 1.0


class TestPix2PixCommunityPipeline:
    def test_forward(self):
        from examples.community.pix2pix import create_generator
        from src.pipelines.pix2pix import Pix2PixPipeline

        gen = create_generator(netG="unet_256", norm="batch")
        pipe = Pix2PixPipeline(generator=gen)
        x = torch.rand(1, 3, 64, 64) * 2 - 1
        out = pipe(source_image=x, output_type="pt")
        assert out.images.shape == (1, 3, 64, 64)


# ---------------------------------------------------------------------------
# DiffusionPipeline-based inference tests
# ---------------------------------------------------------------------------


class TestParallelGANPipeline:
    """Tests for the ParallelGANPipeline (DiffusionPipeline subclass)."""

    @pytest.fixture
    def pipeline(self):
        from examples.community.parallel_gan import ParaGAN, ParallelGANPipeline

        gen = ParaGAN(input_nc=3, output_nc=3, n_blocks=2)
        return ParallelGANPipeline(generator=gen)

    def test_inherits_diffusion_pipeline(self):
        from diffusers import DiffusionPipeline
        from examples.community.parallel_gan import ParallelGANPipeline

        assert issubclass(ParallelGANPipeline, DiffusionPipeline)

    def test_call_output_pt(self, pipeline):
        from examples.community.parallel_gan import ParallelGANPipelineOutput

        x = torch.randn(1, 3, 256, 256)
        out = pipeline(source_image=x, output_type="pt")
        assert isinstance(out, ParallelGANPipelineOutput)
        assert isinstance(out.images, torch.Tensor)
        assert out.images.shape == (1, 3, 256, 256)
        assert out.images.min() >= -1.0
        assert out.images.max() <= 1.0

    def test_call_output_np(self, pipeline):
        import numpy as np

        x = torch.randn(1, 3, 256, 256)
        out = pipeline(source_image=x, output_type="np")
        assert isinstance(out.images, np.ndarray)
        assert out.images.shape == (1, 256, 256, 3)
        assert out.images.min() >= 0.0
        assert out.images.max() <= 1.0

    def test_call_output_pil(self, pipeline):
        from PIL import Image

        x = torch.randn(1, 3, 256, 256)
        out = pipeline(source_image=x, output_type="pil")
        assert isinstance(out.images, list)
        assert isinstance(out.images[0], Image.Image)
        assert out.images[0].size == (256, 256)

    def test_call_return_tuple(self, pipeline):
        x = torch.randn(1, 3, 256, 256)
        out = pipeline(source_image=x, output_type="pt", return_dict=False)
        assert isinstance(out, tuple)
        assert isinstance(out[0], torch.Tensor)

    def test_device_property(self, pipeline):
        assert pipeline.device == torch.device("cpu")

    def test_dtype_property(self, pipeline):
        assert pipeline.dtype == torch.float32


class TestE3DiffPipeline:
    """Tests for the E3DiffPipeline (DiffusionPipeline subclass)."""

    @pytest.fixture
    def pipeline(self):
        from examples.community.e3diff import E3DiffPipeline, E3DiffUNet, GaussianDiffusion

        unet = E3DiffUNet(
            out_channel=3,
            inner_channel=16,
            norm_groups=16,
            channel_mults=(1, 2, 4, 8, 16),
            res_blocks=1,
            image_size=64,
            condition_ch=3,
        )
        diff = GaussianDiffusion(
            denoise_fn=unet, image_size=64, channels=3, xT_noise_r=0.1
        )
        diff.set_noise_schedule(n_timestep=10, schedule="linear", device="cpu")
        return E3DiffPipeline(diffusion=diff)

    def test_inherits_diffusion_pipeline(self):
        from diffusers import DiffusionPipeline
        from examples.community.e3diff import E3DiffPipeline

        assert issubclass(E3DiffPipeline, DiffusionPipeline)

    def test_call_output_pt(self, pipeline):
        from examples.community.e3diff import E3DiffPipelineOutput

        x = torch.randn(1, 3, 64, 64)
        # Use only 2 steps for test speed; production typically uses 50.
        out = pipeline(source_image=x, num_inference_steps=2, output_type="pt")
        assert isinstance(out, E3DiffPipelineOutput)
        assert isinstance(out.images, torch.Tensor)
        assert out.images.shape == (1, 3, 64, 64)
        assert out.nfe == 2

    def test_call_output_np(self, pipeline):
        import numpy as np

        x = torch.randn(1, 3, 64, 64)
        out = pipeline(source_image=x, num_inference_steps=2, output_type="np")
        assert isinstance(out.images, np.ndarray)
        assert out.images.shape == (1, 64, 64, 3)
        assert out.images.min() >= 0.0
        assert out.images.max() <= 1.0

    def test_call_output_pil(self, pipeline):
        from PIL import Image

        x = torch.randn(1, 3, 64, 64)
        out = pipeline(source_image=x, num_inference_steps=2, output_type="pil")
        assert isinstance(out.images, list)
        assert isinstance(out.images[0], Image.Image)
        assert out.images[0].size == (64, 64)

    def test_call_return_tuple(self, pipeline):
        x = torch.randn(1, 3, 64, 64)
        out = pipeline(source_image=x, num_inference_steps=2, output_type="pt", return_dict=False)
        assert isinstance(out, tuple)
        assert len(out) == 2
        assert isinstance(out[0], torch.Tensor)
        assert out[1] == 2  # nfe

    def test_device_property(self, pipeline):
        assert pipeline.device == torch.device("cpu")

    def test_dtype_property(self, pipeline):
        assert pipeline.dtype == torch.float32


class TestOpenEarthMapSARPipeline:
    """Tests for CUTPipeline with OpenEarthMapSARGenerator (OpenEarthMap-SAR)."""

    @pytest.fixture
    def pipeline(self):
        from examples.community.openearthmap_sar import OpenEarthMapSARGenerator
        from src.pipelines.cut import CUTPipeline

        gen = OpenEarthMapSARGenerator(in_channels=3, out_channels=3, base_filters=64, n_blocks=2)
        return CUTPipeline(generator=gen)

    def test_inherits_diffusion_pipeline(self):
        from diffusers import DiffusionPipeline
        from src.pipelines.cut import CUTPipeline

        assert issubclass(CUTPipeline, DiffusionPipeline)

    def test_call_output_pt(self, pipeline):
        from src.pipelines.cut import CUTPipelineOutput

        x = torch.randn(1, 3, 256, 256)
        out = pipeline(source_image=x, output_type="pt")
        assert isinstance(out, CUTPipelineOutput)
        assert isinstance(out.images, torch.Tensor)
        assert out.images.shape == (1, 3, 256, 256)
        assert out.images.min() >= -1.0
        assert out.images.max() <= 1.0

    def test_call_output_pil(self, pipeline):
        from PIL import Image

        x = torch.randn(1, 3, 256, 256)
        out = pipeline(source_image=x, output_type="pil")
        assert isinstance(out.images, list)
        assert isinstance(out.images[0], Image.Image)
        assert out.images[0].size == (256, 256)

    def test_device_property(self, pipeline):
        assert pipeline.device == torch.device("cpu")

    def test_dtype_property(self, pipeline):
        assert pipeline.dtype == torch.float32


class TestSAR2OpticalPipeline:
    """Tests for the SAR2Optical DiffusionPipeline wrapper."""

    @pytest.fixture
    def pipeline(self):
        from examples.community.sar2optical import SAR2OpticalGenerator, SAR2OpticalPipeline

        gen = SAR2OpticalGenerator(c_in=3, c_out=3)
        return SAR2OpticalPipeline(generator=gen)

    def test_inherits_diffusion_pipeline(self):
        from diffusers import DiffusionPipeline
        from examples.community.sar2optical import SAR2OpticalPipeline

        assert issubclass(SAR2OpticalPipeline, DiffusionPipeline)

    def test_call_output_pt(self, pipeline):
        from examples.community.sar2optical import SAR2OpticalPipelineOutput

        x = torch.randn(2, 3, 256, 256)
        out = pipeline(source_image=x, output_type="pt")
        assert isinstance(out, SAR2OpticalPipelineOutput)
        assert isinstance(out.images, torch.Tensor)
        assert out.images.shape == (2, 3, 256, 256)
        assert out.images.min() >= -1.0
        assert out.images.max() <= 1.0


# ---------------------------------------------------------------------------
# LDDBM (Latent Diffusion Bridge Model)
# ---------------------------------------------------------------------------


class TestLDDBMPipeline:
    """Tests for the LDDBM pipeline (examples/lddbm, built-in)."""

    def test_lddbm_imports(self):
        from examples.lddbm import LDDBMPipeline, LDDBMPipelineOutput, load_lddbm_pipeline

        assert LDDBMPipeline is not None
        assert LDDBMPipelineOutput is not None
        assert callable(load_lddbm_pipeline)

    def test_lddbm_import_from_src(self):
        from src.pipelines.lddbm import LDDBMPipeline, LDDBMPipelineOutput, load_lddbm_pipeline

        assert LDDBMPipeline is not None
        assert LDDBMPipelineOutput is not None
        assert callable(load_lddbm_pipeline)

    def test_load_lddbm_nonexistent_raises(self):
        """load_lddbm_pipeline raises FileNotFoundError when checkpoint dir is missing."""
        from examples.lddbm import load_lddbm_pipeline

        with pytest.raises((FileNotFoundError, OSError)) as exc_info:
            load_lddbm_pipeline("/tmp/nonexistent-lddbm-checkpoints", device="cpu")
        assert "encoder_x" in str(exc_info.value) or "missing" in str(exc_info.value).lower() or "no such file" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# SynDiff (ICON Lab, IEEE TMI 2023)
# ---------------------------------------------------------------------------


class TestSynDiffPipeline:
    """Tests for the SynDiff community pipeline."""

    def test_syndiff_imports(self):
        from examples.community.syndiff import (
            SynDiffPipeline,
            SynDiffPipelineOutput,
            load_syndiff_community_pipeline,
        )

        assert SynDiffPipeline is not None
        assert SynDiffPipelineOutput is not None
        assert callable(load_syndiff_community_pipeline)

    def test_load_syndiff_nonexistent_checkpoint_raises(self):
        """load_syndiff_community_pipeline raises when checkpoint dir is missing."""
        from examples.community.syndiff import load_syndiff_community_pipeline

        with pytest.raises(FileNotFoundError):
            load_syndiff_community_pipeline(
                "/tmp/nonexistent-syndiff-exp",
                device="cpu",
            )


# ---------------------------------------------------------------------------
# SelfRDB (ICON Lab, Medical Image Analysis 2024)
# ---------------------------------------------------------------------------


class TestSelfRDBPipeline:
    """Tests for the SelfRDB community pipeline."""

    def test_selfrdb_imports(self):
        from examples.community.selfrdb import (
            SelfRDBPipeline,
            SelfRDBPipelineOutput,
            load_selfrdb_community_pipeline,
        )

        assert SelfRDBPipeline is not None
        assert SelfRDBPipelineOutput is not None
        assert callable(load_selfrdb_community_pipeline)

    def test_load_selfrdb_nonexistent_checkpoint_raises(self):
        """load_selfrdb_community_pipeline raises when checkpoint file is missing."""
        from examples.community.selfrdb import load_selfrdb_community_pipeline

        with pytest.raises(FileNotFoundError):
            load_selfrdb_community_pipeline(
                "/tmp/nonexistent-selfrdb.ckpt",
                device="cpu",
            )


# ---------------------------------------------------------------------------
# DiffuseIT (Kwon & Ye, ICLR 2023) – bundled implementation, no external repo
# ---------------------------------------------------------------------------


class TestDiffuseITPipeline:
    """Tests for the DiffuseIT community pipeline."""

    def test_diffuseit_imports(self):
        from examples.community.diffuseit import (
            DiffuseITPipeline,
            DiffuseITPipelineOutput,
            load_diffuseit_community_pipeline,
        )

        assert DiffuseITPipeline is not None
        assert DiffuseITPipelineOutput is not None
        assert callable(load_diffuseit_community_pipeline)

    def test_load_diffuseit_nonexistent_checkpoint_raises(self):
        """load_diffuseit_community_pipeline raises when checkpoint or bundled source is missing."""
        from examples.community.diffuseit import load_diffuseit_community_pipeline

        with pytest.raises((FileNotFoundError, RuntimeError, OSError)):
            load_diffuseit_community_pipeline(
                "/tmp/nonexistent-diffuseit-ckpt",
                device="cpu",
            )


# ---------------------------------------------------------------------------
# DiffusionRouter (kvmduc)
# ---------------------------------------------------------------------------


class TestDiffusionRouterPipeline:
    """Tests for the DiffusionRouter community pipeline."""

    def test_diffusionrouter_imports(self):
        from examples.community.diffusionrouter import (
            DIFFUSIONROUTER_CLASS_NAMES,
            DIFFUSIONROUTER_DEFAULT_CHAIN,
            DiffusionRouterConfig,
            DiffusionRouterPipeline,
            DiffusionRouterPipelineOutput,
            load_diffusionrouter_community_pipeline,
        )

        assert DIFFUSIONROUTER_CLASS_NAMES is not None
        assert DIFFUSIONROUTER_DEFAULT_CHAIN is not None
        assert DiffusionRouterConfig is not None
        assert DiffusionRouterPipeline is not None
        assert DiffusionRouterPipelineOutput is not None
        assert callable(load_diffusionrouter_community_pipeline)

    def test_load_diffusionrouter_nonexistent_checkpoint_raises(self):
        """load_diffusionrouter_community_pipeline raises when checkpoint is missing."""
        from examples.community.diffusionrouter import load_diffusionrouter_community_pipeline

        with pytest.raises(FileNotFoundError):
            load_diffusionrouter_community_pipeline(
                "/tmp/nonexistent-diffusionrouter.pt",
                device="cpu",
            )


# ---------------------------------------------------------------------------
# EGSDE (Bili-Sakura/EGSDE-diffusers)
# ---------------------------------------------------------------------------


class TestEGSDEPipeline:
    """Tests for the EGSDE community pipeline wrapper."""

    def test_egsde_imports(self):
        from examples.community.egsde import (
            EGSDE_TASKS,
            EGSDEPipeline,
            EGSDEPipelineOutput,
            load_egsde_community_pipeline,
        )

        assert "cat2dog" in EGSDE_TASKS
        assert EGSDEPipeline is not None
        assert EGSDEPipelineOutput is not None
        assert callable(load_egsde_community_pipeline)

    def test_prepare_source_batch_resize(self):
        from examples.community.egsde.pipeline import _prepare_source_batch

        x = torch.rand(1, 3, 64, 64)
        y = _prepare_source_batch(x, image_size=128, device=torch.device("cpu"), dtype=torch.float32)
        assert y.shape == (1, 3, 128, 128)
        assert y.min() >= -1.0
        assert y.max() <= 1.0

    def test_normalize_state_dict_strips_module_prefix(self):
        from examples.community.egsde.pipeline import _normalize_state_dict

        d = {"module.w": torch.zeros(1)}
        out = _normalize_state_dict(d)
        assert list(out.keys()) == ["w"]

    def test_ensure_egsde_path_invalid_raises(self, tmp_path):
        from examples.community.egsde.pipeline import _ensure_egsde_path

        with pytest.raises(FileNotFoundError):
            _ensure_egsde_path(tmp_path / "not-egsde")

    def test_load_egsde_without_task_requires_fields(self, tmp_path):
        from examples.community.egsde import load_egsde_community_pipeline

        fake = tmp_path / "EGSDE-diffusers"
        (fake / "runners").mkdir(parents=True)
        (fake / "guided_diffusion").mkdir(parents=True)
        (fake / "runners" / "egsde.py").write_text("# stub\n")
        (fake / "guided_diffusion" / "script_util.py").write_text("# stub\n")

        with pytest.raises(ValueError, match="task"):
            load_egsde_community_pipeline(fake, task=None, device="cpu")

    def test_load_egsde_bad_task_raises(self, tmp_path):
        from examples.community.egsde import load_egsde_community_pipeline

        fake = tmp_path / "EGSDE-diffusers"
        (fake / "runners").mkdir(parents=True)
        (fake / "guided_diffusion").mkdir(parents=True)
        (fake / "runners" / "egsde.py").write_text("# stub\n")
        (fake / "guided_diffusion" / "script_util.py").write_text("# stub\n")

        with pytest.raises(ValueError, match="Unknown EGSDE task"):
            load_egsde_community_pipeline(fake, task="unknown_task", device="cpu")


# ---------------------------------------------------------------------------
# CycleDiff (Zou et al., IEEE TIP 2026)
# ---------------------------------------------------------------------------


class TestCycleDiffIntegration:
    """Tests for the CycleDiff community bridge (local checkout)."""

    def test_cyclediff_imports(self):
        from examples.community.cyclediff import (
            CYCLEDIFF_REPO_URL,
            inject_cyclediff_sys_path,
            resolve_cyclediff_root,
        )

        assert CYCLEDIFF_REPO_URL.endswith("CycleDiff")
        assert callable(resolve_cyclediff_root)
        assert callable(inject_cyclediff_sys_path)

    def test_resolve_cyclediff_explicit_missing_raises(self):
        from examples.community.cyclediff import resolve_cyclediff_root

        with pytest.raises(FileNotFoundError):
            resolve_cyclediff_root("/tmp/nonexistent-cyclediff-checkout-xyz")

    def test_resolve_and_inject_with_minimal_fake_tree(self, tmp_path):
        from examples.community.cyclediff import inject_cyclediff_sys_path, resolve_cyclediff_root

        (tmp_path / "train_uncond_ldm_cycle.py").write_text("# marker\n")
        (tmp_path / "translation_uncond_ldm_cycle.py").write_text("# marker\n")
        (tmp_path / "ddm").mkdir()
        (tmp_path / "ddm" / "__init__.py").write_text("")
        root = resolve_cyclediff_root(tmp_path)
        assert root == tmp_path.resolve()
        inject_cyclediff_sys_path(tmp_path)
        assert str(tmp_path.resolve()) in sys.path

    def test_train_main_runs_upstream_script_stub(self, tmp_path):
        from examples.community.cyclediff.train import main

        (tmp_path / "translation_uncond_ldm_cycle.py").write_text("# marker\n")
        (tmp_path / "ddm").mkdir()
        (tmp_path / "ddm" / "__init__.py").write_text("")
        (tmp_path / "train_uncond_ldm_cycle.py").write_text("import sys\nsys.exit(0)\n")
        rc = main(["--cyclediff-root", str(tmp_path), "train_uncond_ldm_cycle.py"])
        assert rc == 0


# ---------------------------------------------------------------------------
# CycleDiffusion (Wu & De la Torre, ICCV 2023)
# ---------------------------------------------------------------------------


class TestCycleDiffusionIntegration:
    """Tests for the humansensinglab/cycle-diffusion community bridge (local checkout)."""

    def test_cycle_diffusion_imports(self):
        from examples.community.cycle_diffusion import (
            CYCLE_DIFFUSION_REPO_URL,
            inject_cycle_diffusion_sys_path,
            resolve_cycle_diffusion_root,
        )

        assert "humansensinglab/cycle-diffusion" in CYCLE_DIFFUSION_REPO_URL
        assert callable(resolve_cycle_diffusion_root)
        assert callable(inject_cycle_diffusion_sys_path)

    def test_resolve_cycle_diffusion_explicit_missing_raises(self):
        from examples.community.cycle_diffusion import resolve_cycle_diffusion_root

        with pytest.raises(FileNotFoundError):
            resolve_cycle_diffusion_root("/tmp/nonexistent-cycle-diffusion-checkout-xyz")

    def test_resolve_and_inject_with_minimal_fake_tree(self, tmp_path):
        from examples.community.cycle_diffusion import inject_cycle_diffusion_sys_path, resolve_cycle_diffusion_root

        (tmp_path / "main.py").write_text("# marker\n")
        (tmp_path / "trainer").mkdir()
        (tmp_path / "trainer" / "trainer.py").write_text("# marker\n")
        (tmp_path / "utils").mkdir()
        (tmp_path / "utils" / "config_utils.py").write_text("# marker\n")
        (tmp_path / "model").mkdir()
        (tmp_path / "model" / "__init__.py").write_text("")
        root = resolve_cycle_diffusion_root(tmp_path)
        assert root == tmp_path.resolve()
        inject_cycle_diffusion_sys_path(tmp_path)
        assert str(tmp_path.resolve()) in sys.path

    def test_train_main_runs_upstream_script_stub(self, tmp_path):
        from examples.community.cycle_diffusion.train import main

        (tmp_path / "main.py").write_text("import sys\nsys.exit(0)\n")
        (tmp_path / "trainer").mkdir()
        (tmp_path / "trainer" / "trainer.py").write_text("# marker\n")
        (tmp_path / "utils").mkdir()
        (tmp_path / "utils" / "config_utils.py").write_text("# marker\n")
        (tmp_path / "model").mkdir()
        (tmp_path / "model" / "__init__.py").write_text("")
        rc = main(["--cycle-diffusion-root", str(tmp_path), "main.py"])
        assert rc == 0


# ---------------------------------------------------------------------------
# SDEdit (Meng et al., ICLR 2022)
# ---------------------------------------------------------------------------


class TestSDEditIntegration:
    """Tests for the ermongroup/SDEdit community bridge (local checkout)."""

    def test_sdedit_imports(self):
        from examples.community.sdedit import (
            SDEDIT_REPO_URL,
            inject_sdedit_sys_path,
            resolve_sdedit_root,
        )

        assert "ermongroup/SDEdit" in SDEDIT_REPO_URL
        assert callable(resolve_sdedit_root)
        assert callable(inject_sdedit_sys_path)

    def test_resolve_sdedit_explicit_missing_raises(self):
        from examples.community.sdedit import resolve_sdedit_root

        with pytest.raises(FileNotFoundError):
            resolve_sdedit_root("/tmp/nonexistent-sdedit-checkout-xyz")

    def test_resolve_and_inject_with_minimal_fake_tree(self, tmp_path):
        from examples.community.sdedit import inject_sdedit_sys_path, resolve_sdedit_root

        (tmp_path / "main.py").write_text("# marker\n")
        (tmp_path / "runners").mkdir()
        (tmp_path / "runners" / "image_editing.py").write_text("# marker\n")
        (tmp_path / "models").mkdir()
        (tmp_path / "models" / "diffusion.py").write_text("# marker\n")
        (tmp_path / "configs").mkdir()
        (tmp_path / "configs" / "bedroom.yml").write_text("data:\n  dataset: dummy\n")
        root = resolve_sdedit_root(tmp_path)
        assert root == tmp_path.resolve()
        inject_sdedit_sys_path(tmp_path)
        assert str(tmp_path.resolve()) in sys.path

    def test_train_main_runs_upstream_script_stub(self, tmp_path):
        from examples.community.sdedit.train import main

        (tmp_path / "main.py").write_text("import sys\nsys.exit(0)\n")
        (tmp_path / "runners").mkdir()
        (tmp_path / "runners" / "image_editing.py").write_text("# marker\n")
        (tmp_path / "models").mkdir()
        (tmp_path / "models" / "diffusion.py").write_text("# marker\n")
        (tmp_path / "configs").mkdir()
        (tmp_path / "configs" / "bedroom.yml").write_text("data:\n  dataset: dummy\n")
        rc = main(["--sdedit-root", str(tmp_path), "main.py"])
        assert rc == 0


class TestHnegSRCCommunity:
    """Tests for the Hneg-SRC community pipeline."""

    def test_imports(self):
        from examples.community.hneg_src import (
            HnegSRCConfig,
            HnegSRCTrainer,
            HnegSRCPipeline,
            SRCLoss,
            PatchHDCELoss,
            load_hneg_src_pipeline,
        )

        assert callable(HnegSRCTrainer)
        assert callable(load_hneg_src_pipeline)

    def test_src_loss_forward(self):
        from src.models.hneg_src import SRCLoss

        crit = SRCLoss(num_patches=4, lambda_src=0.05)
        feat_q = torch.randn(4, 64)
        feat_k = torch.randn(4, 64)
        loss, weight = crit(feat_q, feat_k, epoch=0)
        assert loss.ndim == 0
        assert weight.shape == (1, 4, 4)

    def test_hdce_loss_forward(self):
        from src.models.hneg_src import PatchHDCELoss

        crit = PatchHDCELoss(batch_size=1, lambda_hdce=0.1)
        feat_q = torch.randn(4, 64)
        feat_k = torch.randn(4, 64)
        weight = torch.ones(1, 4, 4)
        loss = crit(feat_q, feat_k, weight)
        assert loss.ndim == 0

    def test_trainer_init(self):
        from examples.hneg_src import HnegSRCConfig, HnegSRCTrainer

        cfg = HnegSRCConfig(device="cpu", batch_size=1, num_patches=4, nce_layers="0,4")
        trainer = HnegSRCTrainer(cfg)
        assert trainer.netG is not None
        assert len(trainer.src_criteria) == 2

    def test_load_pipeline_missing_checkpoint(self, tmp_path):
        from examples.community.hneg_src import load_hneg_src_pipeline

        with pytest.raises(FileNotFoundError):
            load_hneg_src_pipeline(tmp_path / "missing")

    def test_hneg_src_pipeline_is_cut_pipeline(self):
        from diffusers import DiffusionPipeline
        from src.models.cut import CUTGenerator
        from src.pipelines.cut import CUTPipeline
        from src.pipelines.cut import HnegSRCPipeline

        gen = CUTGenerator(input_nc=3, output_nc=3, n_blocks=2)
        pipe = HnegSRCPipeline(generator=gen)
        assert isinstance(pipe, CUTPipeline)
        assert issubclass(HnegSRCPipeline, DiffusionPipeline)


class TestNEGCUTCommunity:
    """Tests for the NEGCUT community pipeline."""

    def test_imports(self):
        from examples.community.negcut import (
            NEGCUTConfig,
            NEGCUTTrainer,
            NEGCUTPipeline,
            LearnedPatchNCELoss,
            NegativeGenerator,
            load_negcut_pipeline,
        )

        assert callable(NEGCUTTrainer)
        assert callable(load_negcut_pipeline)

    def test_learned_patch_nce_loss_forward(self):
        from src.models.negcut import LearnedPatchNCELoss

        crit = LearnedPatchNCELoss(batch_size=1, lambda_nce=1.0)
        feat_q = torch.randn(4, 64)
        feat_k = torch.randn(4, 64)
        neg = torch.randn(4, 64)
        loss = crit(feat_q, feat_k, neg)
        assert loss.ndim == 0

    def test_negative_generator_forward(self):
        from src.models.negcut import NegativeGenerator

        gen = NegativeGenerator(use_conv=False, num_patches=4, nc=32, z_dim=8)
        feats = [torch.randn(2, 32, 8, 8)]
        out = gen(feats, num_patches=4)
        assert len(out) == 1
        assert out[0].shape == (2 * 4, 32)

    def test_trainer_init(self):
        from examples.negcut import NEGCUTConfig, NEGCUTTrainer

        cfg = NEGCUTConfig(device="cpu", batch_size=1, num_patches=4, nce_layers="0,4")
        trainer = NEGCUTTrainer(cfg)
        assert trainer.netG is not None
        assert len(trainer.nce_criteria) == 2

    def test_load_pipeline_missing_checkpoint(self, tmp_path):
        from examples.community.negcut import load_negcut_pipeline

        with pytest.raises(FileNotFoundError):
            load_negcut_pipeline(tmp_path / "missing")

    def test_negcut_pipeline_is_cut_pipeline(self):
        from diffusers import DiffusionPipeline
        from src.models.cut import CUTGenerator
        from src.pipelines.cut import CUTPipeline
        from src.pipelines.cut import NEGCUTPipeline

        gen = CUTGenerator(input_nc=3, output_nc=3, n_blocks=2)
        pipe = NEGCUTPipeline(generator=gen)
        assert isinstance(pipe, CUTPipeline)
        assert issubclass(NEGCUTPipeline, DiffusionPipeline)


class TestDecentCommunity:
    """Tests for the Decent community pipeline."""

    def test_imports(self):
        from examples.community.decent import (
            BNAFModel,
            DecentConfig,
            DecentPipeline,
            DecentTrainer,
            FlowConfig,
            PatchDensityEstimator,
            load_decent_pipeline,
        )

        assert callable(DecentTrainer)
        assert callable(load_decent_pipeline)
        assert callable(BNAFModel)

    def test_bnaf_log_probs(self):
        from src.models.decent import BNAFModel

        flow = BNAFModel(8, n_flows=1, n_layers=0, hidden_dim=8)
        x = torch.randn(4, 8)
        log_probs = flow.log_probs(x)
        assert log_probs.shape == (4,)

    def test_density_estimator_lazy_init(self):
        from src.models.decent import FlowConfig, PatchDensityEstimator

        estimator = PatchDensityEstimator(FlowConfig(flow_type="bnaf", flow_blocks=1))
        feats = [torch.randn(2, 16, 8, 8), torch.randn(2, 32, 4, 4)]
        log_probs, feat_lens, patch_ids = estimator(feats, num_patches=4, detach=True)
        assert len(log_probs) == 2
        assert len(feat_lens) == 2

    def test_density_changing_loss(self):
        from src.models.decent import compute_density_changing_loss

        log_a = [torch.randn(8), torch.randn(8)]
        log_b = [torch.randn(8), torch.randn(8)]
        feat_lens = [torch.tensor(16.0), torch.tensor(32.0)]
        loss = compute_density_changing_loss(
            log_a, log_b, feat_lens, batch_size=2, num_patches=4, var_all=False
        )
        assert loss.ndim == 0

    def test_trainer_init(self):
        from examples.decent import DecentConfig, DecentTrainer

        cfg = DecentConfig(device="cpu", batch_size=1, num_patches=4, var_layers="0,4")
        trainer = DecentTrainer(cfg)
        assert trainer.netG is not None
        assert trainer.netF_A is not None
        assert len(trainer.var_layers) == 2

    def test_load_pipeline_missing_checkpoint(self, tmp_path):
        from examples.community.decent import load_decent_pipeline

        with pytest.raises(FileNotFoundError):
            load_decent_pipeline(tmp_path / "missing")

    def test_decent_pipeline_is_cut_pipeline(self):
        from diffusers import DiffusionPipeline
        from src.models.cut import CUTGenerator
        from src.pipelines.cut import CUTPipeline
        from src.pipelines.cut import DecentPipeline

        gen = CUTGenerator(input_nc=3, output_nc=3, n_blocks=2)
        pipe = DecentPipeline(generator=gen)
        assert isinstance(pipe, CUTPipeline)
        assert issubclass(DecentPipeline, DiffusionPipeline)


class TestFLSeSimCommunity:
    """Tests for the F-LSeSim community pipeline."""

    def test_imports(self):
        from examples.community.flsesim import (
            FLSeSimConfig,
            FLSeSimTrainer,
            FLSeSimPipeline,
            SpatialCorrelativeLoss,
            VGG16FeatureExtractor,
            load_flsesim_pipeline,
        )

        assert callable(FLSeSimTrainer)
        assert callable(load_flsesim_pipeline)

    def test_spatial_loss_forward(self):
        from src.models.flsesim import SpatialCorrelativeLoss

        crit = SpatialCorrelativeLoss(
            patch_nums=4,
            patch_size=8,
            use_norm=True,
            use_conv=False,
        )
        feat_src = torch.randn(2, 32, 16, 16)
        feat_tgt = torch.randn(2, 32, 16, 16)
        loss = crit.loss(feat_src, feat_tgt, None, layer=0)
        assert loss.ndim == 0

    def test_compute_spatial_correlative_loss(self):
        from src.models.flsesim import (
            SpatialCorrelativeLoss,
            VGG16FeatureExtractor,
            compute_spatial_correlative_loss,
        )

        net = VGG16FeatureExtractor()
        crit = SpatialCorrelativeLoss(patch_nums=4, patch_size=8, use_conv=False)
        src = torch.rand(1, 3, 64, 64)
        tgt = torch.rand(1, 3, 64, 64)
        loss = compute_spatial_correlative_loss(net, crit, src, tgt, None, [4, 7])
        assert loss.ndim == 0

    def test_trainer_init(self):
        from examples.flsesim import FLSeSimConfig, FLSeSimTrainer

        cfg = FLSeSimConfig(device="cpu", batch_size=1, patch_nums=4, patch_size=8)
        trainer = FLSeSimTrainer(cfg)
        assert trainer.netG is not None
        assert len(trainer.attn_layers) == 3

    def test_load_pipeline_missing_checkpoint(self, tmp_path):
        from examples.community.flsesim import load_flsesim_pipeline

        with pytest.raises(FileNotFoundError):
            load_flsesim_pipeline(tmp_path / "missing")

    def test_flsesim_pipeline_is_cut_pipeline(self):
        from diffusers import DiffusionPipeline
        from src.models.cut import CUTGenerator
        from src.pipelines.cut import CUTPipeline
        from src.pipelines.cut import FLSeSimPipeline

        gen = CUTGenerator(input_nc=3, output_nc=3, n_blocks=2)
        pipe = FLSeSimPipeline(generator=gen)
        assert isinstance(pipe, CUTPipeline)
        assert issubclass(FLSeSimPipeline, DiffusionPipeline)


class TestCycleGANTurboCommunity:
    """Tests for CycleGAN-Turbo community pipeline."""

    def test_imports(self):
        from examples.community.cyclegan_turbo import (
            CycleGANTurbo,
            CycleGANTurboConfig,
            CycleGANTurboPipeline,
            CycleGANTurboTrainer,
            load_cyclegan_turbo_pipeline,
            PRETRAINED_CYCLEGAN_TURBO,
        )

        assert callable(load_cyclegan_turbo_pipeline)
        assert "day_to_night" in PRETRAINED_CYCLEGAN_TURBO

    def test_build_transform(self):
        from src.models.img2img_turbo import build_transform

        t = build_transform("resize_512x512")
        assert callable(t)

    def test_load_pipeline_requires_one_source(self):
        from examples.community.cyclegan_turbo import load_cyclegan_turbo_pipeline

        with pytest.raises(ValueError):
            load_cyclegan_turbo_pipeline()

    def test_pipeline_is_diffusion_pipeline(self):
        from diffusers import DiffusionPipeline
        from src.pipelines.cyclegan_turbo import CycleGANTurboPipeline

        assert issubclass(CycleGANTurboPipeline, DiffusionPipeline)


class TestPix2PixTurboCommunity:
    """Tests for pix2pix-turbo community pipeline."""

    def test_imports(self):
        from examples.community.pix2pix_turbo import (
            Pix2PixTurbo,
            Pix2PixTurboConfig,
            Pix2PixTurboPipeline,
            Pix2PixTurboTrainer,
            load_pix2pix_turbo_pipeline,
            PRETRAINED_PIX2PIX_TURBO,
            canny_from_pil,
        )

        assert callable(load_pix2pix_turbo_pipeline)
        assert callable(canny_from_pil)
        assert "edge_to_image" in PRETRAINED_PIX2PIX_TURBO

    def test_twin_conv(self):
        from src.models.img2img_turbo import TwinConv
        import torch.nn as nn

        conv = TwinConv(nn.Conv2d(3, 3, 1), nn.Conv2d(3, 3, 1))
        assert conv.r is None

    def test_load_pipeline_requires_one_source(self):
        from examples.community.pix2pix_turbo import load_pix2pix_turbo_pipeline

        with pytest.raises(ValueError):
            load_pix2pix_turbo_pipeline()

    def test_pipeline_is_diffusion_pipeline(self):
        from diffusers import DiffusionPipeline
        from src.pipelines.pix2pix_turbo import Pix2PixTurboPipeline

        assert issubclass(Pix2PixTurboPipeline, DiffusionPipeline)
