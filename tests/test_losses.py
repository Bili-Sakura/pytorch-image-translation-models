# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Tests for loss functions."""

import torch

from src.losses.adversarial import GANLoss


class TestGANLoss:
    def test_vanilla(self):
        loss = GANLoss(mode="vanilla")
        pred = torch.randn(2, 1, 4, 4)
        val = loss(pred, target_is_real=True)
        assert val.dim() == 0  # scalar
        assert val.item() >= 0

    def test_lsgan(self):
        loss = GANLoss(mode="lsgan")
        pred = torch.randn(2, 1, 4, 4)
        val = loss(pred, target_is_real=False)
        assert val.dim() == 0
        assert val.item() >= 0

    def test_hinge_discriminator(self):
        loss = GANLoss(mode="hinge")
        pred = torch.randn(2, 1, 4, 4)
        val_real = loss(pred, target_is_real=True, for_discriminator=True)
        val_fake = loss(pred, target_is_real=False, for_discriminator=True)
        assert val_real.dim() == 0
        assert val_fake.dim() == 0

    def test_hinge_generator(self):
        loss = GANLoss(mode="hinge")
        pred = torch.randn(2, 1, 4, 4)
        val = loss(pred, target_is_real=True, for_discriminator=False)
        assert val.dim() == 0
