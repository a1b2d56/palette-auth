"""Command-line interface: generate keys, sign a palette image, verify one.

Examples:
    python -m palette_auth.cli genkey --out signer
    python -m palette_auth.cli sign photo.png photo_signed.png --key signer.private.pem
    python -m palette_auth.cli verify photo_signed.png --key signer.public.pem --tamper-map map.png
"""
from __future__ import annotations

import argparse
import sys

from . import core, crypto


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="palette-auth", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser("genkey", help="generate an Ed25519 keypair")
    p_gen.add_argument("--out", default="key", help="writes <out>.private.pem and <out>.public.pem")

    p_sign = sub.add_parser("sign", help="sign a palette image")
    p_sign.add_argument("input", help="path to source image (PNG/GIF)")
    p_sign.add_argument("output", help="path to write authenticated image")
    p_sign.add_argument("--key", required=True, help="private key .pem")
    p_sign.add_argument("--block-size", type=int, default=core.MIN_BLOCK_SIZE, help="block partition size (pixels)")

    p_verify = sub.add_parser("verify", help="verify a signed palette image")
    p_verify.add_argument("input", help="path to image to verify")
    p_verify.add_argument("--key", required=True, help="public key .pem")
    p_verify.add_argument("--tamper-map", help="optional path to save a visual tamper-map overlay")
    p_verify.add_argument("--recover", help="optional path to save a version with tampered blocks restored")
    p_verify.add_argument("--recover-method", choices=["neural", "bilinear"], default="neural", help="recovery reconstruction engine (default: neural)")

