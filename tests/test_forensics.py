"""Tests for advanced forensic modules: SRM, ELA, noise inconsistency, and PyTorch forensics net."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
from PIL import Image

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from palette_auth import ai_detector


def test_srm_filter_bank():
    filters = ai_detector.get_srm_filters()
    assert filters.shape == (30, 3, 3, 3)
    # Check that high-pass filters are zero-sum across channels
    for i in range(len(filters)):
        assert abs(float(filters[i, 0].sum())) < 1e-4


def test_ela_computation():
    arr = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
    ela = ai_detector.compute_ela(arr, quality=90)
    assert ela.shape == (64, 64, 3)
    assert ela.dtype == np.float32
