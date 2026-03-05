"""Tests for generator and discriminator architectures."""

import torch

from src.models.discriminators import PatchGANDiscriminator
from src.models.generators import ResNetGenerator, UNetGenerator


class TestUNetGenerator:
    def test_output_shape(self):
        gen = UNetGenerator(in_channels=3, out_channels=3, num_downs=5, base_filters=16)
        x = torch.randn(1, 3, 32, 32)
        y = gen(x)
        assert y.shape == (1, 3, 32, 32)

    def test_different_channels(self):
        gen = UNetGenerator(in_channels=1, out_channels=4, num_downs=5, base_filters=16)
        x = torch.randn(1, 1, 32, 32)
        y = gen(x)
        assert y.shape == (1, 4, 32, 32)

    def test_output_range(self):
        gen = UNetGenerator(in_channels=3, out_channels=3, num_downs=5, base_filters=16)
        x = torch.randn(1, 3, 32, 32)
        y = gen(x)
        assert y.min() >= -1.0
        assert y.max() <= 1.0


class TestResNetGenerator:
    def test_output_shape(self):
        gen = ResNetGenerator(in_channels=3, out_channels=3, base_filters=16, n_residual_blocks=2)
        x = torch.randn(1, 3, 64, 64)
        y = gen(x)
        assert y.shape == (1, 3, 64, 64)

    def test_output_range(self):
        gen = ResNetGenerator(in_channels=3, out_channels=3, base_filters=16, n_residual_blocks=2)
        x = torch.randn(1, 3, 64, 64)
        y = gen(x)
        assert y.min() >= -1.0
        assert y.max() <= 1.0


class TestPatchGANDiscriminator:
    def test_output_is_spatial(self):
        disc = PatchGANDiscriminator(in_channels=6, base_filters=16, n_layers=2)
        x = torch.randn(1, 6, 64, 64)
        y = disc(x)
        # Output should be a spatial map, not a scalar
        assert y.dim() == 4
        assert y.shape[0] == 1
        assert y.shape[1] == 1

    def test_batch_processing(self):
        disc = PatchGANDiscriminator(in_channels=6, base_filters=16, n_layers=2)
        x = torch.randn(4, 6, 64, 64)
        y = disc(x)
        assert y.shape[0] == 4
