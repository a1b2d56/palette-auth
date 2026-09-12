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
- **Baseline Priors**: Each block tag carries a 12-byte digest representing a $2 \times 2$ downsampled RGB average of the pristine content.
- **Harmonic Poisson Relaxation**: For contiguous tampered clusters, the recovery engine upsamples color priors, anchors structural features, and solves a discrete Dirichlet Poisson relaxation ($\nabla^2 E = 0$) along cluster boundaries to guarantee $C^1$ smoothness with zero boundary seam artifacts.
- **Guided Neural Inpainting**: When PyTorch is available, an optional gated convolutional network fuses the steganographic color priors with contextual surrounding textures.

### 4. Passive Forensic Saliency (No Keys Required)
When no cryptographic key is available, `palette-auth` includes passive forensic detection tools:
- **Spatial Rich Model (SRM)**: A 30-filter high-pass residual bank capturing 1st/2nd-order derivatives and Laplacian curvature to detect statistical noise anomalies.
- **Error Level Analysis (ELA)**: Evaluates compression history discrepancies across image regions.
- **Dual-Stream CNN**: Combines RGB texture features with SRM high-pass residual maps to highlight spliced boundaries and generative infill.

---

## Installation

### Core Package (Lightweight, Zero PyTorch Dependency)
```bash
pip install palette-auth
```
*Dependencies: `pillow`, `numpy`, `cryptography`.*

### With Optional Deep Learning & Forensics
```bash
pip install "palette-auth[ai]"
```
*Adds: `torch`, `scipy`.*

### Development Setup
```bash
git clone https://github.com/a1b2d56/palette-auth.git
cd palette-auth
pip install -e ".[dev,ai]"
```

---

## Quickstart & Demo

Run the end-to-end interactive demo and view the web verification dashboard:

```bash
# Run demonstration pipeline
python demo.py --open

# Or on Windows using the batch launcher:
run_demo.bat
```

The demo script automatically:
1. Loads a test scene and generates an Ed25519 keypair.
2. Invisibly signs the image in-pixel.
3. Applies a realistic localized forgery.
4. Cryptographically verifies both copies and localizes the tampered blocks.
5. Performs hybrid discrete Poisson self-recovery.
6. Runs the passive forensic detector.
7. Generates an interactive verification dashboard (`demo_output/index.html`) with an image comparison slider and 6-stage pipeline composite (`demo_output/comparison.png`).

---

## CLI Usage

### Generate Keys
```bash
python -m palette_auth.cli genkey --out signer
# Output: signer.private.pem and signer.public.pem
```

### Sign an Image
```bash
python -m palette_auth.cli sign photo.png photo_signed.png \
    --key signer.private.pem \
    --block-size 32
```

### Verify and Recover
```bash
python -m palette_auth.cli verify photo_signed.png \
    --key signer.public.pem \
    --tamper-map tamper_overlay.png \
    --recover recovered.png \
    --recover-method neural
```

### Keyless Forensic Analysis
```bash
python -m palette_auth.cli detect suspect.png --heatmap forensic_heatmap.png
```

---

## Python API

```python
from palette_auth import generate_keypair, sign_image, verify_image, render_recovery
from palette_auth.ai_detector import predict_heatmap, render_heatmap

# 1. Key generation
private_key, public_key = generate_keypair()

# 2. In-pixel signing
sign_info = sign_image("original.png", "signed.png", private_key, block_size=32)
print(f"Signed {sign_info['n_blocks']} blocks.")

# 3. Verification & localization
result = verify_image("signed.png", public_key)
if result.authentic:
    print("Image is intact and verified authentic.")
else:
    print(f"Tampering detected: {result.tampered_count}/{result.total_blocks} blocks modified.")
    
    # Render restored approximation
    render_recovery("signed.png", result, "recovered.png", method="neural")

# 4. Passive forensic analysis (unkeyed)
blocks, probabilities = predict_heatmap("suspect.png")
render_heatmap("suspect.png", blocks, probabilities, "heatmap.png")
```

---

## Binary Data Specification

### Header Structure (Block 0)
The primary header occupies 83 bytes (664 bits) within Block 0:

| Offset (Bytes) | Field | Size | Description |
|---|---|---|---|
| `0..3` | Magic | 4 | ASCII `'PIAH'` (`Palette Image Auth Header`) |
| `4` | Version | 1 | Header format version (`0x01`) |
| `5..6` | Block Size | 2 | Block dimension $B$ in pixels (big-endian `uint16`) |
| `7..10` | Block Count $N$ | 4 | Total number of partitioned blocks (big-endian `uint32`) |
| `11..18` | Seed | 8 | PRNG seed for deterministic block mapping (big-endian `int64`) |
| `19..82` | Signature | 64 | Ed25519 signature over concatenated block hashes |

### Local Block Tag
Each block has a 20-byte tag embedded into its distant partner block:
- **Block Content Hash** (8 bytes): Truncated SHA-256 digest bound to block position.
- **Recovery Prior** (12 bytes): Downsampled $2 \times 2$ RGB grid average ($4 \text{ cells} \times 3 \text{ bytes}$).

---

## Benchmark Results

Evaluated across synthetic benchmark scenes ($256 \times 256$ pixels, block size $= 32 \times 32$):

| Metric | Cryptographic Verification | Passive Dual-Stream SRM Net |
|---|---|---|
| **False Positive Rate (Intact)** | **0.00%** | 3.1% |
| **Localization Precision** | **1.00** | 0.84 |
| **Localization Recall** | **0.92** | 0.88 |
| **Localization F1 Score** | **0.96** | 0.86 |
| **Mean PSNR (Embedding Impact)** | **31.6 dB** | N/A (Passive) |

Run the test and benchmark suites locally:
```bash
# Run unit and integration tests (35 tests)
pytest tests/ -v

# Run adversarial scenarios (copy-move, collateral evidence destruction, single-pixel)
python attacks.py

# Run quantitative evaluation harness
python evaluate.py
```

---

## Security Model & Limitations

- **Fragile Watermark**: Any geometric transform, re-compression, or resizing deliberately breaks the signature. This is intended behavior for digital forensics and integrity verification.
- **Key Security**: Signing requires the private key; verification requires only the public key.
- **Adversarial Collateral Destruction**: If an attacker modifies both block $B$ and its partner block $B'$, the system flags the blocks as `uncertain` rather than silently failing.
- For vulnerability disclosure instructions, see [SECURITY.md](SECURITY.md).

---

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.
