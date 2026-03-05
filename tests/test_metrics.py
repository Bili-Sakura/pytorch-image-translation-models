"""Tests for image quality metrics."""

import torch

from src.metrics.image_quality import compute_psnr, compute_ssim


class TestMetrics:
    def test_psnr_identical(self):
        x = torch.rand(2, 3, 32, 32)
        psnr = compute_psnr(x, x)
        # Identical images should have very high (or inf) PSNR
        assert psnr > 30.0

    def test_ssim_identical(self):
        x = torch.rand(2, 3, 32, 32)
        ssim = compute_ssim(x, x)
        # SSIM of identical images should be close to 1
        assert ssim > 0.99

    def test_psnr_different(self):
        x = torch.rand(2, 3, 32, 32)
        y = torch.rand(2, 3, 32, 32)
        psnr = compute_psnr(x, y)
        assert isinstance(psnr, float)

    def test_ssim_different(self):
        x = torch.rand(2, 3, 32, 32)
        y = torch.rand(2, 3, 32, 32)
        ssim = compute_ssim(x, y)
        assert isinstance(ssim, float)
