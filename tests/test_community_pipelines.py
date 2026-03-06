# Copyright (c) 2026 EarthBridge Team.

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
