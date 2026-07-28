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


def save_private_key(
    key: Ed25519PrivateKey,
    path: str | Path,
    password: str | bytes | None = None,
) -> None:
    """Save an Ed25519 private key to a PEM-encoded file.
    
    If password is provided, the key is encrypted using PKCS#8 BestAvailableEncryption;
    otherwise, it is written unencrypted.
    """
    if password is not None:
        pw_bytes = password.encode("utf-8") if isinstance(password, str) else password
        enc: serialization.KeySerializationEncryption = serialization.BestAvailableEncryption(pw_bytes)
    else:
        enc = serialization.NoEncryption()

    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=enc,
    )
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(pem)


def save_public_key(key: Ed25519PublicKey, path: str | Path) -> None:
    """Save an Ed25519 public key to a SubjectPublicKeyInfo PEM-encoded file."""
    pem = key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(pem)


def load_private_key(
    path: str | Path,
    password: str | bytes | None = None,
) -> Ed25519PrivateKey:
    """Load an Ed25519 private key from a PEM-encoded file.
    
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If key format or password is invalid.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Private key file not found: {path}")
    pw_bytes = password.encode("utf-8") if isinstance(password, str) else password
    key = serialization.load_pem_private_key(p.read_bytes(), password=pw_bytes)
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError(f"Expected Ed25519 private key, got {type(key).__name__}")
    return key


def load_public_key(path: str | Path) -> Ed25519PublicKey:
    """Load an Ed25519 public key from a PEM-encoded file.
    
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If key format is invalid.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Public key file not found: {path}")
    key = serialization.load_pem_public_key(p.read_bytes())
    if not isinstance(key, Ed25519PublicKey):
        raise ValueError(f"Expected Ed25519 public key, got {type(key).__name__}")
    return key


def sign(private_key: Ed25519PrivateKey, message: bytes) -> bytes:
    """Sign a byte message using an Ed25519 private key, returning a 64-byte signature."""
    return private_key.sign(message)


def verify(public_key: Ed25519PublicKey, signature: bytes, message: bytes) -> bool:
    """Verify an Ed25519 signature against message bytes. Returns True if authentic."""
    try:
        public_key.verify(signature, message)
        return True
    except InvalidSignature:
        return False
    except Exception:
        return False
