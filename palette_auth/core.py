"""High-level sign / verify pipeline for palette image authentication.

sign_image(): hash every block, sign the overall root with Ed25519, then
embed each block's own tag -- its content hash, plus a coarse recovery
thumbnail -- into a *different*, mapped block, plus a small header
(holding the signature itself) in the top-left corner.

verify_image(): re-derives everything from the image alone. Checks two
independent things:
  1. Global: does the recomputed root hash still match the embedded
     Ed25519 signature? This alone detects that *something* changed
     (this is essentially what the 2011 scheme provides).
  2. Local: for every block, does its recomputed content hash match the
     tag that was stored *elsewhere* for it at signing time? Blocks that
     disagree are flagged -- this is the added tamper-localization, and
     for flagged blocks the recovery thumbnail lets render_recovery()
     paint back an approximation of the original.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
import struct
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image, ImageDraw

from . import crypto
from .blocks import (
    BLOCK_HASH_SIZE,
    RECOVERY_GRID,
    RECOVERY_SIZE,
    TAG_SIZE,
    BlockCoords,
    block_content_hash,
    block_recovery_digest,
    build_block_mapping,
    destination_groups,
    partition_blocks,
)
from .embed import (
    bits_to_bytes,
    block_capacity,
    build_pair_map,
    bytes_to_bits,
    canonical_indices,
    embed_bits_in_block,
    extract_bits_from_block,
)

logger = logging.getLogger(__name__)

MAGIC = b"PIAH"
VERSION = 1
MIN_BLOCK_SIZE = 32  # block 0 must be big enough to hold the header
HEADER_LEN = 4 + 1 + 2 + 4 + 8 + crypto.SIGNATURE_SIZE  # magic+ver+bsize+n+seed+sig = 83 bytes


def _load_palette_image(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load an image and return its palette index array and RGB palette."""
    img = Image.open(path)
    if img.mode != "P":
        logger.info("Converting %s from %s to palette mode 'P' (256 colors adaptive)", path, img.mode)
        img = img.convert("P", palette=Image.ADAPTIVE, colors=256)
    index_array = np.array(img, dtype=np.uint8).copy()
    palette_flat = img.getpalette() or []
    n = max(1, len(palette_flat) // 3)
    palette = np.array(palette_flat[: n * 3], dtype=np.uint8).reshape(n, 3)
    return index_array, palette


def _save_palette_image(index_array: np.ndarray, palette: np.ndarray, path: str | Path) -> None:
    """Save palette image ensuring 768-byte palette padding."""
    out = Image.fromarray(index_array, mode="P")
    pal = palette.flatten().tolist()
    pal += [0] * (768 - len(pal))
    out.putpalette(pal)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    out.save(p)


