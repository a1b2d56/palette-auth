"""Ed25519 signing utilities.

This replaces the elliptic-curve-plus-hand-tuned-finite-field-arithmetic
approach from the 2011 paper with a modern, constant-time, well-audited
signature scheme. Ed25519 keys and signatures are also much smaller
(32-byte public key, 64-byte signature) than a typical 2010-era ECC
setup, which matters here because the signature has to be small enough
to embed invisibly inside the image itself.
"""
from __future__ import annotations

from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

PUBLIC_KEY_SIZE: int = 32
SIGNATURE_SIZE: int = 64


def generate_keypair() -> tuple[Ed25519PrivateKey, Ed25519PublicKey]:
    """Generate a new Ed25519 private/public keypair."""
    priv = Ed25519PrivateKey.generate()
    return priv, priv.public_key()


