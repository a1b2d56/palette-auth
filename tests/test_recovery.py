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
