"""Tests for image quality metrics."""

import torch

from src.metrics.paired import PairedImageMetricEvaluator
from src.metrics.paired.psnr import compute_psnr
from src.metrics.paired.ssim import compute_ssim


class TestPairedMetrics:
    def test_psnr_identical(self):
        x = torch.rand(2, 3, 32, 32)
        psnr = compute_psnr(x, x)
        # Identical images should have very high (or inf) PSNR
        assert psnr > 30.0 or psnr == float("inf")

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
        assert psnr >= 0

    def test_ssim_different(self):
        x = torch.rand(2, 3, 32, 32)
        y = torch.rand(2, 3, 32, 32)
        ssim = compute_ssim(x, y)
        assert isinstance(ssim, float)
        assert 0 <= ssim <= 1

    def test_evaluator_psnr_ssim(self):
        evaluator = PairedImageMetricEvaluator(metrics=["psnr", "ssim"])
        x = torch.rand(2, 3, 32, 32)
        y = torch.rand(2, 3, 32, 32)
        scores = evaluator(x, y)
        assert "psnr" in scores
        assert "ssim" in scores
        assert isinstance(scores["psnr"], float)
        assert isinstance(scores["ssim"], float)

    def test_evaluator_subset(self):
        evaluator = PairedImageMetricEvaluator(metrics=["psnr", "ssim"])
        x = torch.rand(2, 3, 32, 32)
        scores = evaluator(x, x, metrics=["psnr"])
        assert scores.keys() == {"psnr"}
