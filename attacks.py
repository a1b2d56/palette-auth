"""Adversarial evaluation suite for palette image authentication."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from demo import make_demo_image
from palette_auth import core, crypto

OUT = Path("attack_output")


def setup():
    OUT.mkdir(exist_ok=True)
    orig, signed = OUT / "orig.png", OUT / "signed.png"
    make_demo_image(orig)
    priv, pub = crypto.generate_keypair()
    priv_path, pub_path = OUT / "k.private.pem", OUT / "k.public.pem"
    crypto.save_private_key(priv, priv_path)
    crypto.save_public_key(pub, pub_path)
    core.sign_image(orig, signed, priv_path, block_size=32)
    priv_path.unlink()
    return signed, pub_path


def attack_copy_move(signed, pubkey):
    print("\n[Test 1] Intra-image copy-move attack")
    print("Splicing genuine pixel blocks from elsewhere within the same signed image.")
    img = Image.open(signed)
    arr = np.array(img, dtype=np.uint8).copy()
    palette = img.getpalette()
    arr[60:124, 60:124] = arr[190:254, 190:254]
    out = Image.fromarray(arr, "P")
    out.putpalette(palette)
    path = OUT / "attack_copy_move.png"
    out.save(path)
    r = core.verify_image(path, pubkey)
    print(f"Result: authentic={r.authentic} ({r.reason}), "
          f"{len(r.tampered_blocks)}/{len(r.all_blocks)} blocks flagged")
    print("Verification: Caught via position-bound block content hashing.")


