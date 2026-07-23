# palette-auth

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/a1b2d56/palette-auth/actions/workflows/ci.yml/badge.svg)](https://github.com/a1b2d56/palette-auth/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/Tests-35%20Passing-brightgreen.svg)]()
[![Type Checked: mypy](https://img.shields.io/badge/Type%20Checked-mypy-blue.svg)]()

**palette-auth** is a lightweight, self-contained image authentication and tamper-localization library for indexed color images (PNG-8, GIF). It embeds an Ed25519 digital signature, block-level verification tags, and low-frequency recovery color priors directly into pixel data using embedding-invariant pair-swap steganography.

No sidecar files, external manifests, or out-of-band databases are required: verifying an image verifies whether it has been altered since signing, pinpoints the tampered spatial blocks, and reconstructs the modified regions.

---

## Architecture Overview

```
                          SIGNING PIPELINE
  [Input Image] ──> [Pair Map (min(Ca, Cb))] ──> [Block Hashes + 2x2 RGB Priors]
                           │                                  │
                           ▼                                  ▼
                  [Root SHA-256 Digest]          [Distance-Max PRNG Mapping]
                           │                                  │
                           ▼ (Ed25519)                        ▼
                  [Digital Signature]            [Embed into Partner Blocks]
                           │                                  │
                           ▼                                  │
                 [Header in Block 0]                          │
                           └─────────────────┬────────────────┘
                                             ▼
                                  [Authenticated Image]
```

```
                        VERIFICATION PIPELINE
  [Suspect Image] ──> [Extract Header from Block 0] ──> [Verify Ed25519 Root]
                             │
                             ├─ If Root Signature Invalid:
                             │    ├── [Extract Block Tags from Partner Blocks]
                             │    ├── [Recompute Canonical Block Hashes]
                             │    ├── [Localize Tampered Coordinates]
                             │    └── [Self-Recovery: 2x2 Priors + Poisson Relaxation]
                             │
                             └─ If Root Signature Valid:
                                  └── Authentic (Zero Modification)
```

---

## Core Technical Concepts

### 1. The Canonical Pair-Swap Invariant
In indexed color images (e.g. 256-color PNGs), pixels store palette indices rather than direct RGB values. Modifying the least significant bit of an index directly alters the color drastically. 

To solve this, `palette-auth` computes an optimal Euclidean color pairing:
- Each palette color $c_a$ is greedily paired with its nearest neighbor $c_b$ in RGB space.
- A bit $1$ is represented by $\max(c_a, c_b)$, while bit $0$ is represented by $\min(c_a, c_b)$.
- Before computing SHA-256 content hashes, indices are mapped to their canonical representative $\min(c_a, c_b)$.
- **Result**: Embedding signatures via pair swaps is mathematically invariant to the hash computation, eliminating the cyclic dependency of signing steganographic content.

### 2. Distance-Maximizing Permutation
If a block's authentication tag were stored within itself, an attacker editing that block could easily destroy the verification evidence. 

`palette-auth` maps each block's tag into a distant partner block using a keyed pseudo-random permutation seeded in the image header:
- Partner blocks are chosen to maximize spatial Euclidean distance.
- Load-balancing ensures uniform payload distribution across blocks.
- If an attacker tampers with a localized region, the evidence remains safely preserved in untouched blocks elsewhere on the canvas.

### 3. Two-Tier Self-Recovery
When localized tampering is detected, the framework can reconstruct the missing content:
