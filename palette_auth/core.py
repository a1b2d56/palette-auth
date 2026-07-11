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


def inspect(path: str | Path) -> dict[str, Any]:
    """Parse a signed image's header and derive its block layout and
    block-mapping, without doing full verification. Meant for tooling (e.g.
    adversarial test scripts) that needs to know where a given block's
    evidence lives -- an informed attacker could derive exactly this from
    the public algorithm plus the image, so there's no extra secrecy lost
    by exposing it here too."""
    index_array, palette = _load_palette_image(path)
    height, width = index_array.shape
    pair_of = build_pair_map(palette)
    header_region = index_array[0:MIN_BLOCK_SIZE, 0:MIN_BLOCK_SIZE]
    header_bits = extract_bits_from_block(header_region, pair_of, HEADER_LEN * 8, start_slot=0)
    header = bits_to_bytes(header_bits)
    if header[:4] != MAGIC:
        raise ValueError("no valid signature header found")
    version = header[4]
    if version != VERSION:
        raise ValueError(f"unsupported header version: {version} (expected {VERSION})")
    block_size = struct.unpack(">H", header[5:7])[0]
    seed = header[11:19]
    blocks = partition_blocks(height, width, block_size)
    perm = build_block_mapping(seed, len(blocks), blocks=blocks)
    return {
        "blocks": blocks,
        "perm": perm,
        "pair_of": pair_of,
        "palette": palette,
        "index_array": index_array,
        "block_size": block_size,
        "version": version,
    }


@dataclass
class VerificationResult:
    authentic: bool
    tampered_blocks: list[BlockCoords]
    all_blocks: list[BlockCoords]
    reason: str = ""
    recovered_colors: dict[int, list[tuple[int, int, int]]] = field(default_factory=dict)
    confident_tampered: list[BlockCoords] = field(default_factory=list)
    uncertain_blocks: list[BlockCoords] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.confident_tampered and self.tampered_blocks:
            self.confident_tampered = list(self.tampered_blocks)

    @property
    def total_blocks(self) -> int:
        return len(self.all_blocks)

    @property
    def tampered_count(self) -> int:
        return len(self.tampered_blocks)

    @property
    def confident_count(self) -> int:
        return len(self.confident_tampered)

    @property
    def uncertain_count(self) -> int:
        return len(self.uncertain_blocks)

    @property
    def tampered_ratio(self) -> float:
        return len(self.tampered_blocks) / max(1, len(self.all_blocks))


def sign_image(
    input_path: str | Path,
    output_path: str | Path,
    private_key_path: str | Path,
    block_size: int = MIN_BLOCK_SIZE,
) -> dict[str, Any]:
    """Sign a palette image, embedding authentication tags and recovery priors in-pixel."""
    if block_size < MIN_BLOCK_SIZE:
        raise ValueError(f"block_size must be >= {MIN_BLOCK_SIZE} (needs room for the header)")

    index_array, palette = _load_palette_image(input_path)
    height, width = index_array.shape
    blocks = partition_blocks(height, width, block_size)
    n_blocks = len(blocks)
    if n_blocks < 3:
        raise ValueError("image too small for this block size")

    pair_of = build_pair_map(palette)
    content = canonical_indices(index_array, pair_of)
    rgb_array = palette[index_array]  # pristine RGB, captured before any embedding

    hashes = [block_content_hash(content, b) for b in blocks]
    recoveries = [block_recovery_digest(rgb_array, b) for b in blocks]
    tags = [hashes[i] + recoveries[i] for i in range(n_blocks)]
    root = hashlib.sha256(b"".join(hashes)).digest()

    private_key = crypto.load_private_key(private_key_path)
    signature = crypto.sign(private_key, root)

    seed = np.random.default_rng().integers(0, 2**63 - 1, dtype=np.int64).tobytes()
    perm = build_block_mapping(seed, n_blocks, blocks=blocks)
    groups = destination_groups(perm, n_blocks)

    header = (
        MAGIC
        + bytes([VERSION])
        + struct.pack(">H", block_size)
        + struct.pack(">I", n_blocks)
        + seed[:8]
        + signature
    )
    if len(header) != HEADER_LEN:
        raise ValueError(f"Header construction error: expected {HEADER_LEN} bytes, got {len(header)}")

    # Capacity checks up front so we fail loudly before mutating anything.
    b0 = blocks[0]
    b0_slice = index_array[b0.row0 : b0.row1, b0.col0 : b0.col1]
    if block_capacity(b0_slice, pair_of) < HEADER_LEN * 8:
        raise ValueError("not enough embeddable pixels in the top-left block for the header")
    for dest_idx, sources in groups.items():
        if not sources:
            continue
        needed = len(sources) * TAG_SIZE * 8
        d = blocks[dest_idx]
        d_slice = index_array[d.row0 : d.row1, d.col0 : d.col1]
        if block_capacity(d_slice, pair_of) < needed:
            raise ValueError(
                f"block {dest_idx} needs {needed} embeddable bits for {len(sources)} tags "
                f"(hash+recovery) but doesn't have them; try a larger block_size"
            )

    embed_bits_in_block(b0_slice, pair_of, bytes_to_bits(header), start_slot=0)
    for dest_idx, sources in groups.items():
        if not sources:
            continue
        d = blocks[dest_idx]
        d_slice = index_array[d.row0 : d.row1, d.col0 : d.col1]
        payload: list[int] = []
        for src in sources:
            payload.extend(bytes_to_bits(tags[src]))
        embed_bits_in_block(d_slice, pair_of, payload, start_slot=0)

    _save_palette_image(index_array, palette, output_path)
    return {"n_blocks": n_blocks, "block_size": block_size, "output": str(output_path)}


def verify_image(input_path: str | Path, public_key_path: str | Path) -> VerificationResult:
    """Verify a signed palette image and localize any tampered blocks."""
    index_array, palette = _load_palette_image(input_path)
    height, width = index_array.shape
    pair_of = build_pair_map(palette)

    # The header always lives in the top-left MIN_BLOCK_SIZE region --
    # that's what lets us read it before we know the image's own block_size.
    header_region = index_array[0:MIN_BLOCK_SIZE, 0:MIN_BLOCK_SIZE]
    try:
        header_bits = extract_bits_from_block(header_region, pair_of, HEADER_LEN * 8, start_slot=0)
        header = bits_to_bytes(header_bits)
    except ValueError:
        return VerificationResult(False, [], [], reason="could not read a header (too small, or not signed)")

    if header[:4] != MAGIC:
        return VerificationResult(False, [], [], reason="no valid signature header found")
    header_version = header[4]
    if header_version != VERSION:
        return VerificationResult(
            False,
            [],
            [],
            reason=f"unsupported header version: {header_version} (expected {VERSION})",
        )
    block_size = struct.unpack(">H", header[5:7])[0]
    n_blocks_claimed = struct.unpack(">I", header[7:11])[0]
    seed = header[11:19]
    signature = header[19 : 19 + crypto.SIGNATURE_SIZE]

    blocks = partition_blocks(height, width, block_size)
    if len(blocks) != n_blocks_claimed:
        return VerificationResult(False, [], blocks, reason="image dimensions don't match the signed original")

    content = canonical_indices(index_array, pair_of)
    hashes_now = [block_content_hash(content, b) for b in blocks]
    root_now = hashlib.sha256(b"".join(hashes_now)).digest()

    public_key = crypto.load_public_key(public_key_path)
    globally_valid = crypto.verify(public_key, signature, root_now)

    perm = build_block_mapping(seed, len(blocks), blocks=blocks)
    groups = destination_groups(perm, len(blocks))

    stored_hash_of: dict[int, bytes] = {}
    stored_recovery_of: dict[int, bytes] = {}
    for dest_idx, sources in groups.items():
        if not sources:
            continue
        d = blocks[dest_idx]
        d_slice = index_array[d.row0 : d.row1, d.col0 : d.col1]
        needed_bits = len(sources) * TAG_SIZE * 8
        try:
            bits = extract_bits_from_block(d_slice, pair_of, needed_bits, start_slot=0)
            payload = bits_to_bytes(bits)
        except ValueError:
            payload = b""
        for i, src in enumerate(sources):
            tag = payload[i * TAG_SIZE : (i + 1) * TAG_SIZE] if payload else b""
            stored_hash_of[src] = tag[:BLOCK_HASH_SIZE] if tag else b""
            stored_recovery_of[src] = tag[BLOCK_HASH_SIZE:] if tag else b""

    tampered = [b for b in blocks if stored_hash_of.get(b.index) != hashes_now[b.index]]
    tampered_idx = {b.index for b in tampered}
    # A flagged block is only "confidently" tampered if the block holding
    # *its* evidence wasn't itself flagged -- otherwise the mismatch could
    # just as easily mean the evidence was corrupted, not the block itself.
    confident = [b for b in tampered if perm[b.index] not in tampered_idx]
    uncertain = [b for b in tampered if perm[b.index] in tampered_idx]

    recovered_colors: dict[int, list[tuple[int, int, int]]] = {}
    for b in confident:
        digest = stored_recovery_of.get(b.index, b"")
        if len(digest) == RECOVERY_SIZE:
            cells = [tuple(digest[i : i + 3]) for i in range(0, RECOVERY_SIZE, 3)]
            recovered_colors[b.index] = cells

    authentic = globally_valid and not tampered
    if authentic:
        reason = ""
    elif not globally_valid:
        reason = "global signature mismatch"
    else:
        reason = "localized tampering detected"
    return VerificationResult(
        authentic,
        tampered,
        blocks,
        reason=reason,
        recovered_colors=recovered_colors,
        confident_tampered=confident,
        uncertain_blocks=uncertain,
    )


