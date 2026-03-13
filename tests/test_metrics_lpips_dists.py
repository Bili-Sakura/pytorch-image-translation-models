# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Tests for LPIPS and DISTS (require additional deps)."""

import pytest
import torch


@pytest.fixture
def image_pair():
    return torch.rand(2, 3, 64, 64), torch.rand(2, 3, 64, 64)


def test_lpips():
    from src.metrics.paired.lpips_impl import compute_lpips

    x, y = torch.rand(2, 3, 64, 64), torch.rand(2, 3, 64, 64)
    score = compute_lpips(x, y)
    assert isinstance(score, float)
    assert score >= 0


def test_dists(image_pair):
    from src.metrics.paired.dists_impl import compute_dists

    x, y = image_pair
    score = compute_dists(x, y)
    assert isinstance(score, float)
    assert 0 <= score <= 1
