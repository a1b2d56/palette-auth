import pytest
from pathlib import Path
from PIL import Image
import numpy as np
from palette_auth import cli

def test_cli_genkey(tmp_path, monkeypatch):
    out_key = str(tmp_path / "test_cli_key")
    ret = cli.main(["genkey", "--out", out_key])
    assert ret == 0
    assert Path(f"{out_key}.private.pem").exists()
    assert Path(f"{out_key}.public.pem").exists()

def test_cli_sign_and_verify(tmp_path):
    out_key = str(tmp_path / "signer")
    cli.main(["genkey", "--out", out_key])
    
    # Create 64x64 palette image with varied colors
    orig = tmp_path / "img.png"
    signed = tmp_path / "signed.png"
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    arr[:32, :32] = [200, 50, 50]
    arr[:32, 32:] = [50, 200, 50]
    arr[32:, :32] = [50, 50, 200]
    arr[32:, 32:] = [200, 200, 50]
    Image.fromarray(arr, "RGB").convert("P", palette=Image.ADAPTIVE, colors=256).save(orig)
    
    # Sign
    ret_sign = cli.main(["sign", str(orig), str(signed), "--key", f"{out_key}.private.pem", "--block-size", "32"])
    assert ret_sign == 0
    assert signed.exists()
    
    # Verify clean
    ret_verify = cli.main(["verify", str(signed), "--key", f"{out_key}.public.pem"])
    assert ret_verify == 0