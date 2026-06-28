"""Shared pytest fixtures for palette_auth test suite."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from palette_auth import core, crypto


@pytest.fixture
def keypair_paths(tmp_path: Path) -> tuple[Path, Path]:
    """Generate and save an Ed25519 keypair into tmp_path."""
    priv, pub = crypto.generate_keypair()
    priv_file = tmp_path / "test_signer.priv.pem"
    pub_file = tmp_path / "test_signer.pub.pem"
    crypto.save_private_key(priv, priv_file)
    crypto.save_public_key(pub, pub_file)
    return priv_file, pub_file


@pytest.fixture
def sample_palette_image(tmp_path: Path) -> Path:
    """Create a 64x64 adaptive palette image."""
    img_path = tmp_path / "sample_palette.png"
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    arr[:32, :32] = [210, 45, 45]
    arr[:32, 32:] = [45, 210, 45]
    arr[32:, :32] = [45, 45, 210]
    arr[32:, 32:] = [210, 210, 45]
    img = Image.fromarray(arr, "RGB").convert("P", palette=Image.ADAPTIVE, colors=256)
    img.save(img_path)
    return img_path


@pytest.fixture
def signed_image(tmp_path: Path, sample_palette_image: Path, keypair_paths: tuple[Path, Path]) -> tuple[Path, Path]:
    """Return (signed_image_path, public_key_path)."""
    priv_path, pub_path = keypair_paths
    signed_path = tmp_path / "sample_signed.png"
    core.sign_image(sample_palette_image, signed_path, priv_path, block_size=32)
    return signed_path, pub_path
