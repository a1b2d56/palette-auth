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
