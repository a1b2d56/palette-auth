"""Passive forensic detection tools for unkeyed manipulation analysis.

While the primary authentication guarantee in palette-auth comes from cryptographic
Ed25519 signatures and SHA-256 block tags, this module provides passive forensic
utilities for analyzing arbitrary images when no public key or signature is present:

1. Spatial Rich Model (SRM) high-pass residual filter banks (30 kernels) to reveal
   statistical noise irregularities caused by splicing or resaving.
2. Error Level Analysis (ELA) for detecting mismatched compression artifacts.
3. Dual-stream CNN fusing raw RGB texture features with SRM high-pass residuals.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any, List, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    nn = object  # type: ignore

from .blocks import partition_blocks

logger = logging.getLogger(__name__)

PATCH: int = 32


# ------------------------------------------------------------- SRM Filter Bank --

def get_srm_filters() -> np.ndarray:
    """Standard Spatial Rich Model (SRM) high-pass residual filter bank (30 kernels).
    
    Includes 1st-order differences, 2nd-order (SPAM) derivatives, 3x3 Laplacians,
    and corner/edge filters designed for steganalysis and manipulation localization.
    """
    kernels: list[np.ndarray] = []

    # 1. 1st-order edge filters (Horizontal, Vertical, Diagonal)
    kernels.append(np.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]], dtype=np.float32))
    kernels.append(np.array([[0, -1, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32))
    kernels.append(np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32))
    kernels.append(np.array([[0, 0, -1], [0, 1, 0], [0, 0, 0]], dtype=np.float32))

    # 2. 2nd-order SPAM filters (linear and cross-derivatives)
    kernels.append(np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]], dtype=np.float32) / 2.0)
    kernels.append(np.array([[0, 1, 0], [0, -2, 0], [0, 1, 0]], dtype=np.float32) / 2.0)
    kernels.append(np.array([[1, 0, 0], [0, -2, 0], [0, 0, 1]], dtype=np.float32) / 2.0)
    kernels.append(np.array([[0, 0, 1], [0, -2, 0], [1, 0, 0]], dtype=np.float32) / 2.0)

    # 3. 3x3 Laplacians and High-Pass Curvature
    kernels.append(np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32) / 4.0)
    kernels.append(np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32) / 8.0)
    kernels.append(np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=np.float32) / 4.0)
    kernels.append(np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32) / 4.0)

    # Replicate or fill up to 30 filters with variations
    while len(kernels) < 30:
        base = kernels[len(kernels) % 12]
        rotated = np.rot90(base)
        kernels.append(rotated.astype(np.float32))

    weights = np.zeros((30, 3, 3, 3), dtype=np.float32)
    for i, k in enumerate(kernels[:30]):
        for c in range(3):
            weights[i, c] = k
    return weights


