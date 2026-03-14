# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Tests for community pipelines."""

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

