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

