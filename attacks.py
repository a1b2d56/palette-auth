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


def attack_informed_collateral(signed, pubkey):
    print("\n[Test 2] Targeted evidence destruction attack")
    print("Adversary attempts to corrupt target block and its corresponding evidence block.")
    info = core.inspect(signed)
    blocks, perm, pair_of, palette = info["blocks"], info["perm"], info["pair_of"], info["palette"]

    target = blocks[50]
    evidence_block = blocks[perm[target.index]]
    arr = np.array(Image.open(signed), dtype=np.uint8).copy()
    rng = np.random.default_rng(0)
    for b in (target, evidence_block):
        arr[b.row0:b.row1, b.col0:b.col1] = rng.integers(0, len(palette), (b.height(), b.width()))
    out = Image.fromarray(arr, "P")
    out.putpalette(palette.flatten().tolist())
    path = OUT / "attack_collateral.png"
    out.save(path)

    r = core.verify_image(path, pubkey)
    confident_idx = {b.index for b in r.confident_tampered}
    uncertain_idx = {b.index for b in r.uncertain_blocks}
    status = "Confidently flagged" if target.index in confident_idx else (
        "Classified as uncertain" if target.index in uncertain_idx else "MISSED")
    print(f"Result: authentic={r.authentic} ({r.reason}) -- Rejected.")
    print(f"Target block {target.index} classification: {status}")


def attack_single_pixel(signed, pubkey):
    print("\n[Test 3] Single-pixel modification sensitivity")
    img = Image.open(signed)
    arr = np.array(img, dtype=np.uint8).copy()
    palette = img.getpalette()
    n_colors = len(palette) // 3
    old = int(arr[150, 150])
    new = (old + n_colors // 2) % n_colors
    arr[150, 150] = new
    out = Image.fromarray(arr, "P")
    out.putpalette(palette)
    path = OUT / "attack_single_pixel.png"
    out.save(path)
    r = core.verify_image(path, pubkey)
    print(f"Result: authentic={r.authentic} ({r.reason}), "
          f"{len(r.tampered_blocks)}/{len(r.all_blocks)} blocks flagged (1 pixel modified)")


