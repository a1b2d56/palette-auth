# Security Policy

## Supported Versions

Security updates are provided for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.2.x   | :white_check_mark: |
| 1.1.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Threat Model & Cryptographic Guarantees

`palette-auth` provides a dual-layer cryptographic and forensic authentication model for indexed/palette-based images (e.g., PNG, GIF).

### 1. Global Image Authenticity
- **Primitive**: Ed25519 signature (RFC 8032) using pure Ed25519 public-key cryptography.
- **Payload**: SHA-256 digest of block coordinates concatenated with pixel index values and steganographic payload parity.
- **Guarantee**: Any unauthorized modification to pixel content, block metadata, or header fields causes cryptographic verification to fail (`authentic=False`).

### 2. Block-Level Tamper Localization
- **Primitive**: Position-dependent 3-bit parity digests and 8-bit partner-block validation hashes.
- **Guarantee**: Individual modified blocks (default $32 \times 32$ pixels) are localized with mathematical certainty down to block boundaries.

### 3. Steganographic In-Pixel Recovery
- **Primitive**: Steganographically paired palette swaps encoding $2 \times 2$ downsampled recovery thumbnails in pseudo-random distant partner blocks (keyed PRNG seed).
- **Limitation**: Steganographic recovery provides low-frequency structural recovery, not lossless pristine image restoration. If an attacker modifies both block $B$ and its conjugate recovery block $B'$, recovery is impossible and flagged as `uncertain`.

### 4. Forensic Neural Detector
- **Primitive**: Spatial Rich Model (SRM) 30 high-pass filter bank + RGB texture stream (Dual-Stream ResNet).
- **Guarantee**: Statistical anomaly detection designed to pinpoint tampered regions even when cryptographic headers or public keys are unavailable.

## Reporting a Vulnerability

If you discover a security vulnerability in `palette-auth`, please do NOT open a public issue.

Instead, please submit your findings responsibly:
1. Open a **GitHub Security Advisory** through the repository's "Security" tab.
2. Alternatively, email the maintainers directly with full reproduction steps, proof-of-concept code, and system environment details.

You will receive an acknowledgment within 48 hours and regular status updates until the issue is resolved and a patched release is published.
