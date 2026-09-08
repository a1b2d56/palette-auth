"""Tests for edge cases, non-palette auto-conversion, ragged sizes, and key handling."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from palette_auth import core, crypto


def test_rgb_auto_quantization_during_signing(tmp_path: Path, keypair_paths: tuple[Path, Path]):
    priv_path, pub_path = keypair_paths
    rgb_path = tmp_path / "truecolor_rgb.png"
    signed_path = tmp_path / "rgb_signed.png"

    # Create raw 24-bit RGB image (mode "RGB", not "P")
    arr = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(rgb_path)

    # sign_image should automatically convert RGB to P mode
    info = core.sign_image(rgb_path, signed_path, priv_path, block_size=32)
    assert info["n_blocks"] == 4
    assert signed_path.exists()

    # Verify signature
    res = core.verify_image(signed_path, pub_path)
    assert res.authentic is True
    assert len(res.tampered_blocks) == 0


def test_ragged_dimensions(tmp_path: Path, keypair_paths: tuple[Path, Path]):
    priv_path, pub_path = keypair_paths
    ragged_path = tmp_path / "ragged.png"
    signed_path = tmp_path / "ragged_signed.png"

    # Non-multiple of 32 with ample capacity in ragged edges: 120x120 (min cell 24x24 = 576px)
    h, w = 120, 120
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    xx, yy = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    arr[..., 0] = xx * 255
    arr[..., 1] = yy * 255
    arr[..., 2] = (1 - xx) * 255
    Image.fromarray(arr, mode="RGB").convert("P", palette=Image.ADAPTIVE, colors=256).save(ragged_path)

    info = core.sign_image(ragged_path, signed_path, priv_path, block_size=32)
    assert info["n_blocks"] == 16

    res = core.verify_image(signed_path, pub_path)
    assert res.authentic is True
    assert len(res.tampered_blocks) == 0


def test_wrong_public_key_fails(tmp_path: Path, sample_palette_image: Path, keypair_paths: tuple[Path, Path]):
    priv_path, _ = keypair_paths
    signed_path = tmp_path / "signed_key_test.png"
    core.sign_image(sample_palette_image, signed_path, priv_path, block_size=32)

    # Generate a completely separate wrong keypair
    _, wrong_pub = crypto.generate_keypair()
    wrong_pub_path = tmp_path / "wrong.pub.pem"
    crypto.save_public_key(wrong_pub, wrong_pub_path)

    res = core.verify_image(signed_path, wrong_pub_path)
    assert res.authentic is False
    assert "signature mismatch" in res.reason.lower() or "invalid" in res.reason.lower()


def test_corrupted_header_raises_or_flags(tmp_path: Path, sample_palette_image: Path, keypair_paths: tuple[Path, Path]):
    priv_path, pub_path = keypair_paths
    signed_path = tmp_path / "corrupted_signed.png"
    core.sign_image(sample_palette_image, signed_path, priv_path, block_size=32)

    # Corrupt the header magic bytes
    img = Image.open(signed_path)
    arr = np.array(img, dtype=np.uint8).copy()
    pal = np.array(img.getpalette(), dtype=np.uint8).reshape(-1, 3)
    # Flip bytes in block 0
    arr[:4, :4] ^= 0xFF
    corrupt_img = Image.fromarray(arr, mode="P")
    corrupt_img.putpalette(pal.flatten().tolist())
    corrupt_path = tmp_path / "corrupt.png"
    corrupt_img.save(corrupt_path)

    res = core.verify_image(corrupt_path, pub_path)
    assert res.authentic is False


def test_passphrase_protected_keys(tmp_path: Path):
    priv, pub = crypto.generate_keypair()
    priv_path = tmp_path / "enc_priv.pem"
    pub_path = tmp_path / "enc_pub.pem"
    password = b"supersecret123"

    crypto.save_private_key(priv, priv_path, password=password)
    crypto.save_public_key(pub, pub_path)

    # Loading with correct password works
    loaded_priv = crypto.load_private_key(priv_path, password=password)
    assert loaded_priv is not None

    # Loading with wrong password raises ValueError or TypeError
    with pytest.raises((ValueError, TypeError)):
        crypto.load_private_key(priv_path, password=b"wrongpassword")
