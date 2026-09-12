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

    # Upper triangular pairs (i < j)
    i_idx, j_idx = np.triu_indices(n, k=1)
    dists = dist2[i_idx, j_idx]
    # Deterministic sort with tie-breaking: primary=distance, secondary=i, tertiary=j
    order = np.lexsort((j_idx, i_idx, dists))

    unpaired = set(range(n))
    pair_of: dict[int, int] = {}
    for idx in order:
        a = int(i_idx[idx])
        b = int(j_idx[idx])
        if a in unpaired and b in unpaired:
            pair_of[a] = b
            pair_of[b] = a
            unpaired.discard(a)
            unpaired.discard(b)
        if len(unpaired) < 2:
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


def embed_bits_in_block(
    index_block: np.ndarray, pair_of: dict[int, int], bits: list[int], start_slot: int = 0
) -> None:
    """Modifies index_block IN PLACE. It must be a numpy *view* into the
    full image array (e.g. image_array[r0:r1, c0:c1]), not a copy, or the
    change won't propagate back to the image being signed."""
    positions = _slot_positions(index_block, pair_of)
    end = start_slot + len(bits)
    if end > len(positions):
        raise ValueError(f"block capacity exceeded: need slots {start_slot}:{end}, have {len(positions)}")
    for bit, (r, c) in zip(bits, positions[start_slot:end]):
        idx = int(index_block[r, c])
        partner = pair_of[idx]
        lo, hi = (idx, partner) if idx < partner else (partner, idx)
        index_block[r, c] = hi if bit else lo


def extract_bits_from_block(
    index_block: np.ndarray, pair_of: dict[int, int], n_bits: int, start_slot: int = 0
) -> list[int]:
    positions = _slot_positions(index_block, pair_of)
    end = start_slot + n_bits
    if end > len(positions):
        raise ValueError(f"block capacity exceeded: need slots {start_slot}:{end}, have {len(positions)}")
    bits = []
    for r, c in positions[start_slot:end]:
        idx = int(index_block[r, c])
        partner = pair_of[idx]
        _, hi = (idx, partner) if idx < partner else (partner, idx)
        bits.append(1 if idx == hi else 0)
    return bits
