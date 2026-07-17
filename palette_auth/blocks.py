"""Block partitioning, content hashing, and the block-mapping scheme.

The core idea that makes *localization* actually trustworthy: block i's
authentication tag is never stored inside block i itself. It's stored in
a different block, chosen by a keyed pseudo-random mapping derived from
a seed embedded in the header. If tags were stored locally, anyone who
knows the (public) algorithm could edit a block's content and simply
recompute + re-embed a matching tag right next to it, defeating
localization even though they don't hold the private signing key. By
mapping block i's tag onto an unpredictable different block, an attacker
who only touches block i has no easy way to also patch the one block
that holds evidence of the change.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

import numpy as np

BLOCK_HASH_SIZE = 8  # bytes (64-bit truncated SHA-256) stored per block
RECOVERY_GRID = 2  # a 2x2 grid of average colors per block, for coarse recovery
RECOVERY_SIZE = RECOVERY_GRID * RECOVERY_GRID * 3  # bytes (12 for a 2x2 RGB grid)
TAG_SIZE = BLOCK_HASH_SIZE + RECOVERY_SIZE  # what gets embedded per block, elsewhere


@dataclass(frozen=True)
class BlockCoords:
    index: int
    row0: int
    col0: int
    row1: int  # exclusive
    col1: int  # exclusive

    def height(self) -> int:
        return self.row1 - self.row0

    def width(self) -> int:
        return self.col1 - self.col0


def partition_blocks(height: int, width: int, block_size: int) -> list[BlockCoords]:
    """Partition an H x W grid into block_size x block_size cells in raster
    order. Edge cells are allowed to be smaller (ragged) so any image size
    is supported."""
    blocks = []
    idx = 0
    for row0 in range(0, height, block_size):
        row1 = min(row0 + block_size, height)
        for col0 in range(0, width, block_size):
            col1 = min(col0 + block_size, width)
            blocks.append(BlockCoords(idx, row0, col0, row1, col1))
            idx += 1
    return blocks


def block_content_hash(content_array: np.ndarray, block: BlockCoords) -> bytes:
    """Digest of one block's canonical (embedding-invariant) content, bound
    to its position so two blocks can't be silently swapped with each other."""
    patch = content_array[block.row0 : block.row1, block.col0 : block.col1]
    payload = patch.tobytes() + block.index.to_bytes(4, "big")
    return hashlib.sha256(payload).digest()[:BLOCK_HASH_SIZE]


def block_recovery_digest(rgb_array: np.ndarray, block: BlockCoords, grid: int = RECOVERY_GRID) -> bytes:
    """A coarse grid x grid average-color thumbnail of one block in the
    *original* RGB image, captured at signing time. If the block later
    fails its hash check, this is enough to paint a blurry but genuine
    approximation of what used to be there instead of leaving the
    tampered pixels on screen."""
    patch = rgb_array[block.row0 : block.row1, block.col0 : block.col1]
    h, w = patch.shape[:2]
    out = bytearray()
    for gy in range(grid):
        y0, y1 = h * gy // grid, max(h * gy // grid + 1, h * (gy + 1) // grid)
        for gx in range(grid):
            x0, x1 = w * gx // grid, max(w * gx // grid + 1, w * (gx + 1) // grid)
            mean = patch[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0)
            out.extend(int(round(c)) for c in mean)
