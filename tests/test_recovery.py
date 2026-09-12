"""Tests for image recovery engines: classical bilinear and AI guided neural inpainting."""
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

from palette_auth import core, neural_recovery


def _create_tampered_copy(signed_path: Path, tampered_path: Path) -> None:
    img = Image.open(signed_path)
    arr = np.array(img, dtype=np.uint8).copy()
    pal = np.array(img.getpalette(), dtype=np.uint8).reshape(-1, 3)
    # Alter bottom-right region
    arr[36:58, 36:58] = 128
    tampered_img = Image.fromarray(arr, mode="P")
    tampered_img.putpalette(pal.flatten().tolist())
    tampered_img.save(tampered_path)


def test_bilinear_recovery(tmp_path: Path, signed_image: tuple[Path, Path]):
    signed_path, pub_path = signed_image
    tampered_path = tmp_path / "tampered.png"
    _create_tampered_copy(signed_path, tampered_path)

    result = core.verify_image(tampered_path, pub_path)
    assert not result.authentic
    assert len(result.tampered_blocks) >= 1

    out_bilinear = tmp_path / "rec_bilinear.png"
    core.render_recovery(tampered_path, result, out_bilinear, method="bilinear")
    assert out_bilinear.exists()

    rec_img = Image.open(out_bilinear)
    assert rec_img.size == (64, 64)


def test_neural_recovery(tmp_path: Path, signed_image: tuple[Path, Path]):
    signed_path, pub_path = signed_image
    tampered_path = tmp_path / "tampered.png"
    _create_tampered_copy(signed_path, tampered_path)

    result = core.verify_image(tampered_path, pub_path)
    out_neural = tmp_path / "rec_neural.png"
    neural_recovery.neural_recover_image(tampered_path, result, out_neural)
    assert out_neural.exists()

    rec_img = Image.open(out_neural)
    assert rec_img.size == (64, 64)


def test_untampered_recovery_returns_original(tmp_path: Path, signed_image: tuple[Path, Path]):
    signed_path, pub_path = signed_image
    result = core.verify_image(signed_path, pub_path)
    assert result.authentic

    out_path = tmp_path / "rec_clean.png"
    res_img = neural_recovery.neural_recover_image(signed_path, result, out_path)
    assert out_path.exists()
    assert res_img.size == (64, 64)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_guided_inpainting_net_forward():
    model = neural_recovery.GuidedInpaintingNet()
    model.eval()
    x = torch.randn(2, 7, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 3, 32, 32)
    assert (out >= 0.0).all() and (out <= 1.0).all()


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_train_inpainter():
    clean_imgs = [
        np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8),
        np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8),
    ]
    model = neural_recovery.train_inpainter(clean_imgs, epochs=1, patch_size=64)
    assert isinstance(model, neural_recovery.GuidedInpaintingNet)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_bundled_inpainter_weights_loading():
    engine = neural_recovery.get_default_recovery_engine()
    assert isinstance(engine, neural_recovery.NeuralRecoveryEngine)
    x = torch.randn(2, 7, 32, 32)
    with torch.no_grad():
        out = engine(x)
    assert out.shape == (2, 3, 32, 32)
    assert (out >= 0.0).all() and (out <= 1.0).all()

