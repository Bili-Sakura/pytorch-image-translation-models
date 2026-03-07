# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Tests for UNSB (Unpaired Neural Schrödinger Bridge) components."""

import torch
import numpy as np
import pytest

from src.models.unsb import (
    UNSBGenerator,
    UNSBDiscriminator,
    UNSBEnergyNet,
    PatchSampleMLP,
    GANLoss,
    PatchNCELoss,
    create_generator,
    create_discriminator,
    create_energy_net,
    create_patch_sample_mlp,
)
from src.schedulers.unsb import UNSBScheduler, UNSBSchedulerOutput
from src.pipelines.unsb import UNSBPipeline, UNSBPipelineOutput


# ---------------------------------------------------------------------------
# Generator tests
# ---------------------------------------------------------------------------


class TestUNSBGenerator:
    """Tests for UNSBGenerator."""

    def test_output_shape(self):
        gen = create_generator(input_nc=3, output_nc=3, ngf=64, n_blocks=4, n_mlp=2)
        x = torch.randn(1, 3, 32, 32)
        t = torch.zeros(1).long()
        z = torch.randn(1, 256)
        y = gen(x, t, z)
        assert y.shape == (1, 3, 32, 32)

    def test_output_range(self):
        gen = create_generator(input_nc=3, output_nc=3, ngf=64, n_blocks=4, n_mlp=2)
        x = torch.randn(1, 3, 32, 32)
        t = torch.zeros(1).long()
        z = torch.randn(1, 256)
        y = gen(x, t, z)
        assert y.min() >= -1.0 and y.max() <= 1.0, "Tanh output should be in [-1, 1]"

    def test_batch_size(self):
        gen = create_generator(input_nc=3, output_nc=3, ngf=64, n_blocks=4, n_mlp=2)
        x = torch.randn(2, 3, 32, 32)
        t = torch.zeros(2).long()
        z = torch.randn(2, 256)
        y = gen(x, t, z)
        assert y.shape == (2, 3, 32, 32)

    def test_different_timesteps(self):
        gen = create_generator(input_nc=3, output_nc=3, ngf=64, n_blocks=4, n_mlp=2)
        x = torch.randn(1, 3, 32, 32)
        z = torch.randn(1, 256)
        for t_val in [0, 1, 2, 3, 4]:
            t = torch.tensor([t_val]).long()
            y = gen(x, t, z)
            assert y.shape == (1, 3, 32, 32)

    def test_encode_only(self):
        gen = create_generator(input_nc=3, output_nc=3, ngf=64, n_blocks=4, n_mlp=2)
        x = torch.randn(1, 3, 32, 32)
        t = torch.zeros(1).long()
        z = torch.randn(1, 256)
        feats = gen(x, t, z, layers=[0, 4], encode_only=True)
        assert isinstance(feats, list)
        assert len(feats) > 0

    def test_single_channel(self):
        gen = create_generator(input_nc=1, output_nc=1, ngf=64, n_blocks=4, n_mlp=2)
        x = torch.randn(1, 1, 32, 32)
        t = torch.zeros(1).long()
        z = torch.randn(1, 256)
        y = gen(x, t, z)
        assert y.shape == (1, 1, 32, 32)

    def test_with_antialias(self):
        gen = create_generator(
            input_nc=3, output_nc=3, ngf=64, n_blocks=4, n_mlp=2,
            no_antialias=True, no_antialias_up=True,
        )
        x = torch.randn(1, 3, 32, 32)
        t = torch.zeros(1).long()
        z = torch.randn(1, 256)
        y = gen(x, t, z)
        assert y.shape == (1, 3, 32, 32)


# ---------------------------------------------------------------------------
# Discriminator tests
# ---------------------------------------------------------------------------


class TestUNSBDiscriminator:
    """Tests for UNSBDiscriminator."""

    def test_output_shape(self):
        disc = create_discriminator(input_nc=3, ndf=64, n_layers=3)
        x = torch.randn(1, 3, 32, 32)
        t = torch.zeros(1).long()
        y = disc(x, t)
        assert y.ndim == 4
        assert y.shape[0] == 1
        assert y.shape[1] == 1  # single channel output

    def test_time_conditioning(self):
        disc = create_discriminator(input_nc=3, ndf=64, n_layers=3)
        x = torch.randn(1, 3, 32, 32)
        y0 = disc(x, torch.tensor([0]).long())
        y1 = disc(x, torch.tensor([1]).long())
        # Different timesteps should produce different outputs
        assert not torch.allclose(y0, y1)

    def test_batch_size(self):
        disc = create_discriminator(input_nc=3, ndf=64, n_layers=3)
        x = torch.randn(2, 3, 32, 32)
        t = torch.zeros(2).long()
        y = disc(x, t)
        assert y.shape[0] == 2


# ---------------------------------------------------------------------------
# Energy network tests
# ---------------------------------------------------------------------------


class TestUNSBEnergyNet:
    """Tests for UNSBEnergyNet."""

    def test_output_shape(self):
        enet = create_energy_net(input_nc=3, ndf=64, n_layers=3)
        pair = torch.randn(1, 6, 32, 32)
        t = torch.zeros(1).long()
        y = enet(pair, t, pair)
        assert y.ndim == 4
        assert y.shape[0] == 1

    def test_self_pair(self):
        enet = create_energy_net(input_nc=3, ndf=64, n_layers=3)
        pair = torch.randn(1, 6, 32, 32)
        t = torch.zeros(1).long()
        y = enet(pair, t)
        assert y.shape[0] == 1

    def test_contrastive_pairs(self):
        enet = create_energy_net(input_nc=3, ndf=64, n_layers=3)
        pair1 = torch.randn(1, 6, 32, 32)
        pair2 = torch.randn(1, 6, 32, 32)
        t = torch.zeros(1).long()
        y = enet(pair1, t, pair2)
        assert y.shape[0] == 1


# ---------------------------------------------------------------------------
# PatchSampleMLP tests
# ---------------------------------------------------------------------------


class TestPatchSampleMLP:
    """Tests for PatchSampleMLP."""

    def test_lazy_init(self):
        mlp = create_patch_sample_mlp(use_mlp=True, nc=128)
        assert not mlp.mlp_init
        feats = [torch.randn(1, 64, 8, 8), torch.randn(1, 128, 4, 4)]
        out_feats, ids = mlp(feats, num_patches=16, patch_ids=None)
        assert mlp.mlp_init
        assert len(out_feats) == 2
        assert len(ids) == 2

    def test_deterministic_patch_ids(self):
        mlp = create_patch_sample_mlp(use_mlp=True, nc=128)
        feats = [torch.randn(1, 64, 8, 8)]
        _, ids = mlp(feats, num_patches=16, patch_ids=None)
        out2, _ = mlp(feats, num_patches=16, patch_ids=ids)
        assert out2[0].shape[1] == 128  # nc dimension


# ---------------------------------------------------------------------------
# Loss tests
# ---------------------------------------------------------------------------


class TestGANLoss:
    """Tests for UNSB GANLoss."""

    def test_lsgan(self):
        loss = GANLoss(gan_mode="lsgan")
        pred = torch.randn(4, 1, 4, 4)
        loss_real = loss(pred, True)
        loss_fake = loss(pred, False)
        assert loss_real.shape == torch.Size([])
        assert loss_fake.shape == torch.Size([])

    def test_vanilla(self):
        loss = GANLoss(gan_mode="vanilla")
        pred = torch.randn(4, 1, 4, 4)
        loss_val = loss(pred, True)
        assert loss_val.shape == torch.Size([])


class TestPatchNCELoss:
    """Tests for PatchNCE loss."""

    def test_output_shape(self):
        nce = PatchNCELoss(nce_T=0.07, batch_size=1)
        feat_q = torch.randn(256, 128)
        feat_k = torch.randn(256, 128)
        loss = nce(feat_q, feat_k)
        assert loss.shape[0] == 256


# ---------------------------------------------------------------------------
# Scheduler tests
# ---------------------------------------------------------------------------


class TestUNSBScheduler:
    """Tests for UNSBScheduler."""

    def test_time_schedule(self):
        scheduler = UNSBScheduler(num_timesteps=5, tau=0.01)
        times = scheduler.times
        assert times.shape[0] == 6  # T+1 entries (including t=0)
        assert times[0] == 0.0
        assert times[-1] > 0.0
        # Times should be monotonically increasing
        for i in range(1, len(times)):
            assert times[i] > times[i - 1]

    def test_step(self):
        scheduler = UNSBScheduler(num_timesteps=5, tau=0.01)
        sample = torch.randn(1, 3, 32, 32)
        model_output = torch.randn(1, 3, 32, 32)
        result = scheduler.step(model_output, timestep_idx=1, sample=sample)
        assert isinstance(result, UNSBSchedulerOutput)
        assert result.prev_sample.shape == (1, 3, 32, 32)

    def test_last_step(self):
        scheduler = UNSBScheduler(num_timesteps=5, tau=0.01)
        sample = torch.randn(1, 3, 32, 32)
        model_output = torch.randn(1, 3, 32, 32)
        # Last step should return model output directly
        result = scheduler.step(model_output, timestep_idx=5, sample=sample)
        assert torch.equal(result.prev_sample, model_output)


# ---------------------------------------------------------------------------
# Pipeline tests
# ---------------------------------------------------------------------------


class TestUNSBPipeline:
    """Tests for UNSBPipeline."""

    @pytest.fixture
    def pipeline(self):
        gen = create_generator(input_nc=3, output_nc=3, ngf=64, n_blocks=4, n_mlp=2)
        scheduler = UNSBScheduler(num_timesteps=3, tau=0.01)
        return UNSBPipeline(generator=gen, scheduler=scheduler)

    def test_pipeline_pt_output(self, pipeline):
        source = torch.randn(1, 3, 32, 32)
        result = pipeline(source, output_type="pt")
        assert isinstance(result, UNSBPipelineOutput)
        assert isinstance(result.images, torch.Tensor)
        assert result.images.shape == (1, 3, 32, 32)
        assert result.nfe == 3

    def test_pipeline_pil_output(self, pipeline):
        source = torch.randn(1, 3, 32, 32)
        result = pipeline(source, output_type="pil")
        assert isinstance(result.images, list)
        assert len(result.images) == 1

    def test_pipeline_np_output(self, pipeline):
        source = torch.randn(1, 3, 32, 32)
        result = pipeline(source, output_type="np")
        assert isinstance(result.images, np.ndarray)
        assert result.images.shape == (1, 32, 32, 3)

    def test_pipeline_custom_nfe(self, pipeline):
        source = torch.randn(1, 3, 32, 32)
        result = pipeline(source, num_timesteps=2, output_type="pt")
        assert result.nfe == 2

    def test_pipeline_return_dict_false(self, pipeline):
        source = torch.randn(1, 3, 32, 32)
        result = pipeline(source, output_type="pt", return_dict=False)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_pipeline_output_range(self, pipeline):
        source = torch.randn(1, 3, 32, 32)
        result = pipeline(source, output_type="pt")
        assert result.images.min() >= -1.0
        assert result.images.max() <= 1.0


# ---------------------------------------------------------------------------
# Trainer tests
# ---------------------------------------------------------------------------


class TestUNSBTrainer:
    """Tests for UNSBTrainer."""

    def test_trainer_init(self):
        from examples.unsb.config import UNSBConfig
        from examples.unsb.train_unsb import UNSBTrainer

        config = UNSBConfig(
            input_nc=3, output_nc=3, ngf=64, ndf=64,
            n_blocks=4, n_mlp=2, n_layers_D=2,
            num_timesteps=2, batch_size=1,
            device="cpu",
        )
        trainer = UNSBTrainer(config)
        assert trainer.netG is not None
        assert trainer.netD is not None
        assert trainer.netE is not None
        assert trainer.netF is not None

    def test_trainer_train_step(self):
        from examples.unsb.config import UNSBConfig
        from examples.unsb.train_unsb import UNSBTrainer

        config = UNSBConfig(
            input_nc=3, output_nc=3, ngf=64, ndf=64,
            n_blocks=4, n_mlp=2, n_layers_D=2,
            num_timesteps=2, batch_size=1,
            nce_layers="0,4,8,12",  # Smaller layers for n_blocks=4
            device="cpu",
        )
        trainer = UNSBTrainer(config)
        # Inputs in [0, 1] (the trainer scales to [-1, 1])
        real_A = torch.rand(1, 3, 32, 32)
        real_B = torch.rand(1, 3, 32, 32)
        losses = trainer.train_step(real_A, real_B)
        assert "loss_D" in losses
        assert "loss_E" in losses
        assert "loss_G" in losses
        assert "loss_SB" in losses
        assert "loss_NCE" in losses
        assert all(np.isfinite(v) for v in losses.values())

    def test_trainer_config_validation(self):
        from examples.unsb.config import UNSBConfig

        config = UNSBConfig()
        assert config.num_timesteps == 5
        assert config.tau == 0.01
        assert config.lambda_SB == 1.0
        assert config.lambda_GAN == 1.0
        assert config.lambda_NCE == 1.0
