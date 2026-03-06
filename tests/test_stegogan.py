# Copyright (c) 2026 EarthBridge Team.

"""Tests for the StegoGAN model components."""

import pytest
import torch

from src.models.stegogan.generators import (
    ResnetMaskV1Generator,
    ResnetMaskV3Generator,
)
from src.models.stegogan.networks import (
    NetMatchability,
    ResnetBlock,
    SoftClamp,
    mask_generate,
)
from src.training.stegogan_trainer import StegoGANConfig, StegoGANTrainer


# ---------------------------------------------------------------------------
# Network building blocks
# ---------------------------------------------------------------------------


class TestSoftClamp:
    def test_values_in_range(self):
        m = SoftClamp(alpha=0.001)
        x = torch.tensor([0.0, 0.5, 1.0])
        out = m(x)
        assert out.shape == x.shape
        assert torch.allclose(out, x, atol=1e-6)

    def test_values_outside_range(self):
        m = SoftClamp(alpha=0.001)
        x = torch.tensor([-1.0, 2.0])
        out = m(x)
        # Outside range, soft-clamping shrinks towards boundary
        assert out[0].item() < 0.0  # slightly negative due to alpha
        assert out[1].item() > 1.0  # slightly above 1


class TestResnetBlock:
    def test_output_shape(self):
        block = ResnetBlock(dim=64)
        x = torch.randn(2, 64, 32, 32)
        out = block(x)
        assert out.shape == (2, 64, 32, 32)

    def test_skip_connection(self):
        block = ResnetBlock(dim=64)
        x = torch.zeros(1, 64, 16, 16)
        out = block(x)
        # With zero input, skip connection ensures output is nonzero
        # only from the conv block
        assert out.shape == x.shape


class TestNetMatchability:
    def test_output_shape(self):
        net = NetMatchability(input_dim=256, out_dim=256)
        x = torch.randn(2, 256, 8, 8)
        out = net(x)
        assert out.shape == (2, 256, 8, 8)

    def test_output_range(self):
        net = NetMatchability(input_dim=64, out_dim=32)
        x = torch.randn(1, 64, 16, 16)
        out = net(x)
        # Sigmoid output should be in [0, 1]
        assert out.min() >= 0.0
        assert out.max() <= 1.0


class TestMaskGenerate:
    def test_mask_generate(self):
        net = NetMatchability(input_dim=64, out_dim=64)
        feat = torch.randn(2, 64, 8, 8)
        output, discarded, mask_sum = mask_generate(feat, net)
        assert output.shape == feat.shape
        assert discarded.shape == feat.shape
        assert mask_sum.shape == (2, 1, 8, 8)

    def test_output_and_discarded_sum_to_input(self):
        net = NetMatchability(input_dim=64, out_dim=64)
        feat = torch.randn(1, 64, 8, 8)
        output, discarded, _ = mask_generate(feat, net)
        reconstructed = output + discarded
        assert torch.allclose(reconstructed, feat, atol=1e-6)


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------


class TestResnetMaskV1Generator:
    def test_output_shape(self):
        gen = ResnetMaskV1Generator(
            input_nc=3, output_nc=3, ngf=16, n_blocks=3, resnet_layer=1
        )
        x = torch.randn(1, 3, 64, 64)
        out = gen(x)
        assert out.shape == (1, 3, 64, 64)

    def test_output_range(self):
        gen = ResnetMaskV1Generator(
            input_nc=3, output_nc=3, ngf=16, n_blocks=3, resnet_layer=1
        )
        x = torch.randn(1, 3, 64, 64)
        out = gen(x)
        # Tanh output should be in [-1, 1]
        assert out.min() >= -1.0
        assert out.max() <= 1.0

    def test_with_extra_feature(self):
        gen = ResnetMaskV1Generator(
            input_nc=3, output_nc=3, ngf=16, n_blocks=3, resnet_layer=1
        )
        x = torch.randn(1, 3, 64, 64)
        # Extra feature must match intermediate spatial/channel dims
        extra = torch.randn(1, 64, 16, 16)  # ngf * 4 = 64, spatial 64/4=16
        out = gen(x, extra_feature=extra)
        assert out.shape == (1, 3, 64, 64)

    def test_resnet_layer_minus_one(self):
        gen = ResnetMaskV1Generator(
            input_nc=3, output_nc=3, ngf=16, n_blocks=3, resnet_layer=-1
        )
        x = torch.randn(1, 3, 64, 64)
        out = gen(x)
        assert out.shape == (1, 3, 64, 64)


class TestResnetMaskV3Generator:
    def test_output_shape_and_mask(self):
        gen = ResnetMaskV3Generator(
            input_nc=3, output_nc=3, ngf=16, n_blocks=3,
            input_dim=64, out_dim=64, resnet_layer=1,
        )
        x = torch.randn(1, 3, 64, 64)
        out, feat_disc, mask_sum = gen(x)
        assert out.shape == (1, 3, 64, 64)
        assert mask_sum.shape[0] == 1
        assert mask_sum.shape[1] == 1  # single-channel mask

    def test_output_range(self):
        gen = ResnetMaskV3Generator(
            input_nc=3, output_nc=3, ngf=16, n_blocks=3,
            input_dim=64, out_dim=64, resnet_layer=1,
        )
        x = torch.randn(1, 3, 64, 64)
        out, _, _ = gen(x)
        assert out.min() >= -1.0
        assert out.max() <= 1.0


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class TestStegoGANTrainer:
    @pytest.fixture
    def trainer(self):
        cfg = StegoGANConfig(
            input_nc=3,
            output_nc=3,
            ngf=16,
            ndf=16,
            n_layers_D=2,
            resnet_layer=1,
            mask_group=64,
            device="cpu",
        )
        return StegoGANTrainer(cfg)

    def test_instantiation(self, trainer):
        assert trainer.netG_A is not None
        assert trainer.netG_B is not None
        assert trainer.netD_A is not None
        assert trainer.netD_B is not None

    def test_forward(self, trainer):
        real_A = torch.randn(1, 3, 64, 64)
        real_B = torch.randn(1, 3, 64, 64)
        fwd = trainer.forward(real_A, real_B)
        assert "fake_A" in fwd
        assert "fake_B" in fwd
        assert "rec_A" in fwd
        assert "rec_B" in fwd
        assert "latent_real_B_mask_up" in fwd

    def test_compute_G_loss(self, trainer):
        real_A = torch.randn(1, 3, 64, 64)
        real_B = torch.randn(1, 3, 64, 64)
        fwd = trainer.forward(real_A, real_B)
        losses = trainer.compute_G_loss(fwd)
        assert "total" in losses
        assert "G_A" in losses
        assert "cycle_A" in losses
        assert "consistency_B" in losses
        assert "reg" in losses
        assert losses["total"].requires_grad

    def test_train_step(self, trainer):
        real_A = torch.randn(1, 3, 64, 64)
        real_B = torch.randn(1, 3, 64, 64)
        loss_dict = trainer.train_step(real_A, real_B)
        assert "G_total" in loss_dict
        assert "D_A" in loss_dict
        assert "D_B" in loss_dict
        assert isinstance(loss_dict["G_total"], float)

    def test_config_defaults(self):
        cfg = StegoGANConfig()
        assert cfg.lambda_A == 10.0
        assert cfg.lambda_B == 10.0
        assert cfg.lambda_reg == 0.2
        assert cfg.lambda_consistency == 3.0
        assert cfg.resnet_layer == 8
        assert cfg.fusionblock is True
