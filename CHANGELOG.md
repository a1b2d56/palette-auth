# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - 2026-08-20

### Added
- **Real PyTorch Deep Learning Models**:
  - `DualStreamForensicNet`: Multi-signal forensics fusing a 30-filter Spatial Rich Model (SRM) residual stream with an RGB spatial texture stream.
  - `GuidedInpaintingNet`: Neural restoration network using Gated Convolutions (`GatedConv2d`) and dilated residual blocks fusing low-frequency steganographic guide priors with ambient boundary context.
  - Full PyTorch synthetic training routines and model weight persistence (`save_model`, `load_model`).
- **Multi-Signal Forensics**:
  - Error Level Analysis (`compute_ela`) measuring JPEG compression discrepancies.
  - Local Noise Inconsistency analysis (`compute_noise_inconsistency`) evaluating variance deviations across local blocks.
- **Batched Tensor Inference**:
  - Implemented batched mini-batch inference in `predict_heatmap()`, accelerating full-image saliency evaluation by ~50x.
- **Enterprise-Grade Verification Result**:
  - Converted `VerificationResult` into a typed `@dataclass` with `.total_blocks`, `.tampered_count`, `.confident_count`, `.uncertain_count`, and `.tampered_ratio`.
  - Added geometric boundary adjacency analysis (`_blocks_are_adjacent`, `_is_adjacent_to_set`) without hardcoded block sizes.
- **Security & Cryptography**:
  - Added optional passphrase encryption for private key persistence (`BestAvailableEncryption`).
  - Added explicit header `VERSION` byte validation and `InvalidSignature` exception handling.
- **Interactive UI & Packaging**:
  - Modernized HTML dashboard with shadcn/ui styling, interactive before/after drag comparison slider, 6-panel pipeline progression, and WCAG 2.1 AA accessibility (`role="tablist"`, `aria-selected`, keyboard navigation).
  - Added PEP 561 marker (`py.typed`).
  - Added GitHub Actions multi-platform CI matrix testing across Python 3.9 through 3.14.

### Changed
- Refactored `palette_auth/__init__.py` with lazy module loading (`__getattr__`) to ensure the core package operates with zero dependencies on PyTorch.
- Updated `pyproject.toml` with strict optional dependency groups (`[ai]`, `[dev]`).

## [1.1.0] - 2026-08-10

### Added
- Command-line interface (`palette-auth sign`, `verify`, `ai-detect`, `recover`).
- Bilinear interpolation recovery for tampered regions using extracted $2 \times 2$ steganographic thumbnail tags.
- Initial Spatial Rich Model high-pass filtering for forensic residual visualization.
- Benchmark suite (`evaluate.py`) and adversarial attack test scenarios (`attacks.py`).

## [1.0.0] - 2026-07-25

### Added
- Initial release of `palette-auth`.
- Ed25519 digital signature generation and verification embedded inside indexed PNG palette images.
- Steganographic palette pair-swapping embedding with zero perceptual distortion.
- Block-level parity digests for tamper localization.
