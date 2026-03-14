# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Tests for data loading utilities."""

import os
import tempfile

import torch
from PIL import Image

from src.data.datasets import PairedImageDataset, UnpairedImageDataset
from src.data.transforms import default_transforms, get_transforms


class TestTransforms:
    def test_get_transforms_output_shape(self):
        t = get_transforms(load_size=64, crop_size=32)
        img = Image.new("RGB", (100, 100))
        tensor = t(img)
        assert tensor.shape == (3, 32, 32)

    def test_default_transforms_output_shape(self):
        t = default_transforms(image_size=64)
        img = Image.new("RGB", (100, 100))
        tensor = t(img)
        assert tensor.shape == (3, 64, 64)

    def test_default_transforms_normalized_range(self):
        t = default_transforms(image_size=32, normalize=True)
        img = Image.new("RGB", (32, 32), color=(128, 128, 128))
        tensor = t(img)
        assert tensor.min() >= -1.0
        assert tensor.max() <= 1.0


class TestPairedImageDataset:
    def test_loads_pairs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "source"))
            os.makedirs(os.path.join(tmpdir, "target"))
            for i in range(3):
                Image.new("RGB", (32, 32)).save(
                    os.path.join(tmpdir, "source", f"img_{i}.png")
                )
                Image.new("RGB", (32, 32)).save(
                    os.path.join(tmpdir, "target", f"img_{i}.png")
                )

            ds = PairedImageDataset(
                tmpdir,
                source_transform=default_transforms(32),
                target_transform=default_transforms(32),
            )
            assert len(ds) == 3
            sample = ds[0]
            assert "source" in sample
            assert "target" in sample
            assert isinstance(sample["source"], torch.Tensor)


class TestUnpairedImageDataset:
    def test_loads_domains(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dir_a = os.path.join(tmpdir, "A")
            dir_b = os.path.join(tmpdir, "B")
            os.makedirs(dir_a)
            os.makedirs(dir_b)
            for i in range(3):
                Image.new("RGB", (32, 32)).save(os.path.join(dir_a, f"a_{i}.png"))
                Image.new("RGB", (32, 32)).save(os.path.join(dir_b, f"b_{i}.png"))

            ds = UnpairedImageDataset(
                dir_a,
                dir_b,
                transform_a=default_transforms(32),
                transform_b=default_transforms(32),
            )
            assert len(ds) == 3
            sample = ds[0]
            assert "A" in sample
            assert "B" in sample
