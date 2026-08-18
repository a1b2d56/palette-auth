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


