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
    assert 0.0 <= float(ela.min()) <= float(ela.max()) <= 1.0


def test_noise_inconsistency():
    arr = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
    noise_map = ai_detector.compute_noise_inconsistency(arr, block_size=16)
    assert isinstance(noise_map, np.ndarray)
    assert noise_map.dtype == np.float32
    assert 0.0 <= float(noise_map.min()) <= float(noise_map.max()) <= 1.0


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_dual_stream_forensic_net_forward():
    model = ai_detector.DualStreamForensicNet(pretrained_srm=True)
    model.eval()
    x = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, 1)
    assert (out >= 0.0).all() and (out <= 1.0).all()


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_synthetic_training_and_serialization(tmp_path: Path):
    model_path = tmp_path / "forensic_test.pt"
    # Fast 1-epoch train on small synthetic set
    model, history = ai_detector.train(n_per_class=10, epochs=1, batch_size=4)
    assert model is not None
    assert "train_loss" in history
    
    ai_detector.save_model(model, model_path)
    assert model_path.exists()
    
    loaded_model = ai_detector.load_model(model_path)
    assert isinstance(loaded_model, ai_detector.DualStreamForensicNet)


def test_predict_heatmap_on_sample(tmp_path: Path, sample_palette_image: Path):
    blocks, probs = ai_detector.predict_heatmap(sample_palette_image, block_size=32)
    assert len(blocks) == 4
    assert len(probs) == 4
    assert all(0.0 <= p <= 1.0 for p in probs)

    heatmap_path = tmp_path / "test_heatmap.png"
    ai_detector.render_heatmap(sample_palette_image, blocks, probs, heatmap_path)
    assert heatmap_path.exists()


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_bundled_forensic_weights_loading():
    engine = ai_detector.get_default_forensic_engine()
    assert isinstance(engine, ai_detector.DualStreamForensicNet)
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        out = engine(x)
    assert out.shape == (2, 1)
    assert (out >= 0.0).all() and (out <= 1.0).all()

