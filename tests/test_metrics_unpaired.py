"""Tests for unpaired image quality metrics."""

import torch

from src.metrics.unpaired import UnpairedImageMetricEvaluator
from src.metrics.unpaired.fid import compute_fid


class TestUnpairedMetrics:
    def test_fid_basic(self):
        if not hasattr(compute_fid, "__wrapped__"):
            try:
                from torchmetrics.image.fid import FrechetInceptionDistance
            except ImportError:
                return  # skip if torchmetrics not available
        r = torch.rand(20, 3, 64, 64)
        f = torch.rand(20, 3, 64, 64)
        fid = compute_fid(r, f)
        assert isinstance(fid, float)
        assert fid >= 0

    def test_evaluator_fid(self):
        try:
            from torchmetrics.image.fid import FrechetInceptionDistance
        except ImportError:
            return
        e = UnpairedImageMetricEvaluator(metrics=["fid"])
        r = torch.rand(20, 3, 64, 64)
        f = torch.rand(20, 3, 64, 64)
        s = e(r, f)
        assert "fid" in s
        assert isinstance(s["fid"], float)
