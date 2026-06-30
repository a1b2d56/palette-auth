"""Palette pair-swap steganography.

Hides bits in a palette image by choosing between two perceptually
similar palette colors per pixel -- the same family of technique the
2011 paper used. The one addition here is `canonical_indices`: it maps
both colors in a pair to the same representative value, so the content
hash used for authentication is *invariant* to which twin is currently
displayed. Without that, hashing the raw (post-embedding) pixel data
would create a chicken-and-egg problem where embedding a signature
changes the very content the signature is supposed to describe.
"""
from __future__ import annotations

import numpy as np


def build_pair_map(palette_rgb: np.ndarray) -> dict[int, int]:
    """Greedily pair each palette color with its nearest not-yet-paired
    neighbor (by squared RGB distance). Returns a symmetric index -> index
    dict; an odd color out (if any) is left unpaired and never carries
    embedded bits."""
    n = len(palette_rgb)
    if n < 2:
        return {}
    diffs = palette_rgb[:, None, :].astype(np.int32) - palette_rgb[None, :, :].astype(np.int32)
    dist2 = (diffs ** 2).sum(axis=2)
    np.fill_diagonal(dist2, np.iinfo(np.int64).max)

    order = np.dstack(np.unravel_index(np.argsort(dist2, axis=None), dist2.shape))[0]
    unpaired = set(range(n))
    pair_of: dict[int, int] = {}
    for a, b in order:
        a, b = int(a), int(b)
        if a in unpaired and b in unpaired:
            pair_of[a] = b
            pair_of[b] = a
            unpaired.discard(a)
            unpaired.discard(b)
        if not unpaired:
            break
    return pair_of


def canonical_indices(index_array: np.ndarray, pair_of: dict[int, int]) -> np.ndarray:
    """Map every pixel's palette index to a value shared by both twins of
    its pair. This is what gets hashed for authentication, so embedding a
    bit (i.e. picking a twin) never looks like tampering."""
    lut = np.arange(256, dtype=np.uint8)
    for a, b in pair_of.items():
        lut[a] = min(a, b)
    return lut[index_array]


def _slot_positions(index_block: np.ndarray, pair_of: dict[int, int]) -> list[tuple[int, int]]:
    if not pair_of:
        return []
    mask = np.isin(index_block, list(pair_of.keys()))
    rows, cols = np.where(mask)
    order = np.lexsort((cols, rows))  # deterministic raster order
    return list(zip(rows[order].tolist(), cols[order].tolist()))


def block_capacity(index_block: np.ndarray, pair_of: dict[int, int]) -> int:
    return len(_slot_positions(index_block, pair_of))


def bytes_to_bits(data: bytes) -> list[int]:
    return [(byte >> shift) & 1 for byte in data for shift in range(7, -1, -1)]


def bits_to_bytes(bits: list[int]) -> bytes:
    if len(bits) % 8 != 0:
        raise ValueError("bit length must be a multiple of 8")
    out = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for b in bits[i : i + 8]:
            byte = (byte << 1) | b
        out.append(byte)
    return bytes(out)


