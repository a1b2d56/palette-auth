import pytest
import numpy as np
from PIL import Image
from palette_auth import core, crypto

def create_sample_palette_image(path, size=(64, 64)):
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    arr[:32, :32] = [200, 50, 50]
    arr[:32, 32:] = [50, 200, 50]
    arr[32:, :32] = [50, 50, 200]
    arr[32:, 32:] = [200, 200, 50]
    img = Image.fromarray(arr, "RGB").convert("P", palette=Image.ADAPTIVE, colors=256)
    img.save(path)

def test_full_signing_verification_cycle(tmp_path):
    orig = tmp_path / "orig.png"
    signed = tmp_path / "signed.png"
    tampered = tmp_path / "tampered.png"
    create_sample_palette_image(orig, (64, 64))
    
    priv, pub = crypto.generate_keypair()
    priv_file = tmp_path / "k.priv.pem"
    pub_file = tmp_path / "k.pub.pem"
    crypto.save_private_key(priv, priv_file)
    crypto.save_public_key(pub, pub_file)
    
    info = core.sign_image(orig, signed, priv_file, block_size=32)
    assert info["n_blocks"] == 4
    
    # Verify clean image
    clean_res = core.verify_image(signed, pub_file)
    assert clean_res.authentic is True
    assert len(clean_res.tampered_blocks) == 0
    
    # Tamper block 3 (bottom-right) by changing its palette index
    img = Image.open(signed)
    arr = np.array(img, dtype=np.uint8).copy()
    pal = np.array(img.getpalette(), dtype=np.uint8).reshape(-1, 3)
    arr[40:55, 40:55] = 255
    tampered_img = Image.fromarray(arr, mode="P")
    tampered_img.putpalette(pal.flatten().tolist())
    tampered_img.save(tampered)

    tamp_res = core.verify_image(tampered, pub_file)
    assert tamp_res.authentic is False
    assert len(tamp_res.tampered_blocks) >= 1