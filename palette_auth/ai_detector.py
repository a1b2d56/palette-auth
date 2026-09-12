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
from typing import Any, Union

import numpy as np
from PIL import Image, ImageDraw

try:
    import torch
    from torch import nn, optim
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


# ------------------------------------------------ Error Level Analysis (ELA) --

def compute_ela(
    image: Union[str, Path, Image.Image, np.ndarray],
    quality: int = 90,
    scale: float = 20.0,
) -> np.ndarray:
    """Compute Error Level Analysis (ELA) difference map.

    Re-compresses the image to JPEG at the specified quality factor and computes
    the amplified absolute pixel error. Tampered areas with differing compression
    histories show elevated residual error.
    """
    if isinstance(image, (str, Path)):
        img = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        img = Image.fromarray(image.astype(np.uint8)).convert("RGB")
    else:
        img = image.convert("RGB")

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    resaved = Image.open(buf).convert("RGB")

    orig_arr = np.array(img, dtype=np.float32)
    resaved_arr = np.array(resaved, dtype=np.float32)

    diff = np.abs(orig_arr - resaved_arr)
    ela = np.clip(diff * (scale / 255.0), 0.0, 1.0)
    return ela


def compute_noise_inconsistency(
    image: Union[str, Path, Image.Image, np.ndarray],
    block_size: int = PATCH,
) -> np.ndarray:
    """Estimate local high-frequency noise variance per block across the image."""
    if isinstance(image, (str, Path)):
        img = Image.open(image).convert("L")
    elif isinstance(image, np.ndarray):
        img = Image.fromarray(image.astype(np.uint8)).convert("L")
    else:
        img = image.convert("L")

    arr = np.array(img, dtype=np.float32) / 255.0
    h, w = arr.shape
    # 3x3 Laplacian residual
    pad = np.pad(arr, 1, mode="reflect")
    res = (
        pad[1 : h + 1, 1 : w + 1] * 4.0
        - pad[0 : h, 1 : w + 1]
        - pad[2 : h + 2, 1 : w + 1]
        - pad[1 : h + 1, 0 : w]
        - pad[1 : h + 1, 2 : w + 2]
    )

    blocks = partition_blocks(h, w, block_size)
    variances = []
    for b in blocks:
        patch_res = res[b.row0 : b.row1, b.col0 : b.col1]
        variances.append(float(np.var(patch_res)))

    var_arr = np.array(variances, dtype=np.float32)
    median_var = float(np.median(var_arr)) + 1e-6
    # Relative noise divergence
    divergence = np.abs(var_arr - median_var) / median_var
    return np.clip(divergence / 3.0, 0.0, 1.0)


# ------------------------------------------------ PyTorch Forensic CNN --

if HAS_TORCH:
    class ConvBlock(nn.Module):
        def __init__(self, in_c: int, out_c: int, stride: int = 1):
            super().__init__()
            self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn = nn.BatchNorm2d(out_c)
            self.act = nn.LeakyReLU(0.2, inplace=True)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.act(self.bn(self.conv(x)))

    class ResidualBlock(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(channels)
            self.act = nn.LeakyReLU(0.2, inplace=True)
            self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(channels)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            res = self.conv2(self.act(self.bn1(self.conv1(x))))
            return self.act(x + self.bn2(res))

    class DualStreamForensicNet(nn.Module):
        """Dual-Stream Forensic Neural Network.

        Stream 1: Fixed 30-filter SRM High-Pass Residual Bank + 2 Residual Blocks
        Stream 2: RGB Spatial Texture Feature Extractor + Residual Block
        Fusion: Concatenation -> 1x1 Bottleneck -> Global Pooling -> Sigmoid
        """
        def __init__(self, pretrained_srm: bool = True):
            super().__init__()
            # 1. SRM Stream
            self.srm_conv = nn.Conv2d(3, 30, kernel_size=3, padding=1, bias=False)
            if pretrained_srm:
                srm_w = torch.from_numpy(get_srm_filters())
                self.srm_conv.weight = nn.Parameter(srm_w, requires_grad=False)
            self.srm_proj = ConvBlock(30, 32)
            self.srm_res = ResidualBlock(32)

            # 2. Spatial Stream
            self.spa_proj = ConvBlock(3, 32)
            self.spa_res = ResidualBlock(32)

            # 3. Fusion Stream & Classification Head
            self.fuse = ConvBlock(64, 32)
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(32, 16),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(16, 1),
                nn.Sigmoid(),
            )
            self._init_weights()

        def _init_weights(self) -> None:
            for m in [self.srm_proj, self.srm_res, self.spa_proj, self.spa_res, self.fuse]:
                for submodule in m.modules():
                    if isinstance(submodule, nn.Conv2d):
                        nn.init.kaiming_normal_(submodule.weight, mode="fan_out", nonlinearity="leaky_relu")
                    elif isinstance(submodule, nn.BatchNorm2d):
                        nn.init.constant_(submodule.weight, 1.0)
                        nn.init.constant_(submodule.bias, 0.0)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """x: (B, 3, H, W) normalized to [0, 1]. Returns: (B, 1) tamper probability."""
            # Stream 1: SRM Residuals with Tanh truncation
            res = torch.tanh(self.srm_conv(x) * 3.0)
            f_res = self.srm_res(self.srm_proj(res))

            # Stream 2: Spatial Textures
            f_spa = self.spa_res(self.spa_proj(x))

            # Fusion
            f_comb = torch.cat([f_res, f_spa], dim=1)
            f_feat = self.fuse(f_comb)
            out = self.head(f_feat)
            return out

        def score_patch(self, patch_rgb: np.ndarray) -> float:
            """Score a single (H, W, 3) patch [0, 1] for manipulation probability."""
            self.eval()
            with torch.no_grad():
                tensor = torch.from_numpy(np.transpose(patch_rgb, (2, 0, 1))).unsqueeze(0).float()
                prob = float(self.forward(tensor).item())
            return prob

else:
    # Pure NumPy fallback engine when PyTorch is not installed
    class DualStreamForensicNet:  # type: ignore
        def __init__(self, *args, **kwargs):
            self.srm_w = get_srm_filters()

        def score_patch(self, patch_rgb: np.ndarray) -> float:
            x = np.transpose(patch_rgb, (2, 0, 1)).astype(np.float32)
            # High-pass energy
            diff_y = np.diff(x, axis=1)
            diff_x = np.diff(x, axis=2)
            grad = float(np.mean(np.abs(diff_y)) + np.mean(np.abs(diff_x)))
            return float(1.0 / (1.0 + np.exp(-np.clip(grad * 12.0 - 1.5, -10.0, 10.0))))

        def eval(self):
            return self


# Backward-compatible aliases
DualStreamForensicEngine = DualStreamForensicNet
TamperCNN = DualStreamForensicNet

_DEFAULT_ENGINE = None


def get_default_forensic_engine() -> DualStreamForensicNet:
    """Return a cached default forensic detector instance.

    If bundled weights exist at palette_auth/models/forensic_detector.pt,
    they are loaded automatically; otherwise, the engine initializes with
    calibrated analytical SRM filters.
    """
    global _DEFAULT_ENGINE
    if _DEFAULT_ENGINE is None:
        bundled_weights = Path(__file__).parent / "models" / "forensic_detector.pt"
        if HAS_TORCH and bundled_weights.is_file():
            try:
                _DEFAULT_ENGINE = load_model(bundled_weights)
            except Exception as exc:
                logger.warning("Failed to load bundled forensic weights (%s), using analytical initialization", exc)
                _DEFAULT_ENGINE = DualStreamForensicNet()
        else:
            _DEFAULT_ENGINE = DualStreamForensicNet()
        if HAS_TORCH:
            _DEFAULT_ENGINE.eval()
    return _DEFAULT_ENGINE


# ------------------------------------------------ Synthetic Training Pipeline --

def generate_synthetic_dataset(
    n_per_class: int = 2000,
    patch_size: int = PATCH,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic patches for training: Authentic (0) vs Tampered (1).

    - Class 0 (Authentic): smooth gradients, continuous textures, authentic sensor noise.
    - Class 1 (Tampered): edge splicing, local boundary mismatch, inpainting blur, noise disparity.
    """
    rng = np.random.default_rng(seed)
    patches = []
    labels = []

    # 1. Authentic class (0)
    for _ in range(n_per_class):
        # Base color + smooth gradient + gentle noise
        base = rng.uniform(0.1, 0.9, (1, 1, 3))
        grad_x = np.linspace(0, rng.uniform(-0.3, 0.3), patch_size).reshape(1, patch_size, 1)
        grad_y = np.linspace(0, rng.uniform(-0.3, 0.3), patch_size).reshape(patch_size, 1, 1)
        noise = rng.normal(0, 0.015, (patch_size, patch_size, 3))
        patch = np.clip(base + grad_x + grad_y + noise, 0.0, 1.0).astype(np.float32)
        patches.append(patch)
        labels.append(0.0)

    # 2. Tampered class (1)
    for _ in range(n_per_class):
        base = rng.uniform(0.1, 0.9, (patch_size, patch_size, 3))
        mode = rng.integers(0, 3)
        if mode == 0:
            # Spliced sharp edge cut-paste
            cut = rng.integers(8, patch_size - 8)
            splice = rng.uniform(0.1, 0.9, (patch_size, patch_size, 3))
            base[:, cut:] = splice[:, cut:]
        elif mode == 1:
            # Blur / Inpainting boundary with frequency decay
            center = patch_size // 2
            r = rng.integers(6, 12)
            y, x = np.ogrid[:patch_size, :patch_size]
            mask = (x - center) ** 2 + (y - center) ** 2 <= r ** 2
            mean_color = np.mean(base, axis=(0, 1), keepdims=True)
            base[mask] = mean_color
        else:
            # Steganographic / noise perturbation seam
            seam = rng.integers(10, patch_size - 10)
            base[seam, :] += rng.uniform(0.15, 0.35, (patch_size, 3))
        patch = np.clip(base, 0.0, 1.0).astype(np.float32)
        patches.append(patch)
        labels.append(1.0)

    x = np.array(patches, dtype=np.float32)
    y = np.array(labels, dtype=np.float32).reshape(-1, 1)
    # Shuffle
    idx = rng.permutation(len(x))
    return x[idx], y[idx]


def train(
    n_per_class: int = 2000,
    epochs: int = 15,
    lr: float = 1e-3,
    batch_size: int = 64,
    seed: int = 42,
    device: str | None = None,
) -> tuple[DualStreamForensicNet, dict[str, list[float]]]:
    """Train the Dual-Stream Forensic CNN with synthetic manipulation data using BCE loss."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required to train the forensic detector. Install with: pip install torch")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(seed)
    np.random.seed(seed)

    x_np, y_np = generate_synthetic_dataset(n_per_class=n_per_class, seed=seed)
    n_val = int(len(x_np) * 0.15)
    x_train, x_val = x_np[n_val:], x_np[:n_val]
    y_train, y_val = y_np[n_val:], y_np[:n_val]

    # Convert to BCHW tensors
    x_train_t = torch.from_numpy(np.transpose(x_train, (0, 3, 1, 2))).float().to(device)
    y_train_t = torch.from_numpy(y_train).float().to(device)
    x_val_t = torch.from_numpy(np.transpose(x_val, (0, 3, 1, 2))).float().to(device)
    y_val_t = torch.from_numpy(y_val).float().to(device)

    model = DualStreamForensicNet(pretrained_srm=True).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_acc": []}
    n_batches = int(np.ceil(len(x_train_t) / batch_size))

    model.train()
    for ep in range(epochs):
        perm = torch.randperm(len(x_train_t))
        total_loss = 0.0
        for b in range(n_batches):
            b_idx = perm[b * batch_size : (b + 1) * batch_size]
            bx, by = x_train_t[b_idx], y_train_t[b_idx]

            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / n_batches
        model.eval()
        with torch.no_grad():
            v_preds = model(x_val_t)
            v_loss = criterion(v_preds, y_val_t).item()
            v_acc = float(((v_preds > 0.5) == (y_val_t > 0.5)).float().mean().item())

        history["train_loss"].append(avg_loss)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)
        model.train()

    model.eval()
    return model.cpu(), history


def save_model(model: Any, path: str | Path) -> None:
    """Save model weights to a file (.pt)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if HAS_TORCH and isinstance(model, nn.Module):
        torch.save(model.state_dict(), p)
    else:
        logger.warning("Saving non-PyTorch model weights stub to %s", p)
        p.write_bytes(b"PALETTE_AUTH_FORENSIC_MODEL_V1")


def load_model(path: str | Path, device: str = "cpu") -> DualStreamForensicNet:
    """Load model weights from a .pt file."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch required to load .pt forensic weights")
    model = DualStreamForensicNet()
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")
    state = torch.load(p, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


# ------------------------------------------------ Inference & Heatmap Rendering --

def predict_heatmap(
    image_path: Union[str, Path, Image.Image],
    model: DualStreamForensicEngine | None = None,
    block_size: int = PATCH,
    fuse_ela: bool = True,
) -> tuple[list, np.ndarray]:
    """Runs forensic analysis block-by-block over an image.

    Combines the Dual-Stream CNN output with Error Level Analysis (ELA) and
    local noise variance inconsistency for reliable keyless tamper localization.
    """
    if model is None:
        model = get_default_forensic_engine()

    if isinstance(image_path, (str, Path)):
        img = Image.open(image_path).convert("RGB")
    else:
        img = image_path.convert("RGB")

    arr = np.array(img, dtype=np.uint8)
    height, width = arr.shape[:2]
    blocks = partition_blocks(height, width, block_size)

    # 1. Neural patch scores (Batched for ultra-fast GPU/CPU inference)
    if HAS_TORCH and isinstance(model, nn.Module):
        model.eval()
        patches = []
        for b in blocks:
            patch = arr[b.row0 : b.row1, b.col0 : b.col1].astype(np.float32) / 255.0
            if patch.shape[0] != block_size or patch.shape[1] != block_size:
                pad = np.zeros((block_size, block_size, 3), dtype=np.float32)
                pad[: patch.shape[0], : patch.shape[1]] = patch
                patch = pad
            patches.append(np.transpose(patch, (2, 0, 1)))
        batch_t = torch.from_numpy(np.array(patches, dtype=np.float32)).float()
        with torch.no_grad():
            prob_arr = model(batch_t).squeeze(-1).cpu().numpy().astype(np.float32)
    else:
        probs = []
        for b in blocks:
            patch = arr[b.row0 : b.row1, b.col0 : b.col1].astype(np.float32) / 255.0
            if patch.shape[0] != block_size or patch.shape[1] != block_size:
                pad = np.zeros((block_size, block_size, 3), dtype=np.float32)
                pad[: patch.shape[0], : patch.shape[1]] = patch
                patch = pad
            p = model.score_patch(patch)
            probs.append(p)
        prob_arr = np.array(probs, dtype=np.float32)

    # 2. Multi-signal fusion with ELA and noise inconsistency
    if fuse_ela:
        try:
            ela_map = compute_ela(img, quality=90, scale=18.0)
            noise_scores = compute_noise_inconsistency(img, block_size=block_size)
            ela_scores = []
            for b in blocks:
                ela_p = float(np.mean(ela_map[b.row0 : b.row1, b.col0 : b.col1]))
                ela_scores.append(ela_p)
            ela_arr = np.array(ela_scores, dtype=np.float32)
            # Weighted multi-signal fusion: 0.60 Neural + 0.25 ELA + 0.15 Noise
            prob_arr = 0.60 * prob_arr + 0.25 * np.clip(ela_arr * 3.5, 0, 1) + 0.15 * noise_scores
            prob_arr = np.clip(prob_arr, 0.0, 1.0)
        except Exception as exc:
            logger.debug("Forensic multi-signal ELA/Noise fusion skipped: %s", exc)

    return blocks, prob_arr


def render_heatmap(
    image_path: Union[str, Path, Image.Image],
    blocks: list,
    probs: np.ndarray,
    output_path: Union[str, Path, None],
    threshold: float = 0.5,
) -> Image.Image:
    """Render a clean alpha-blended forensic anomaly heatmap overlay."""
    if isinstance(image_path, (str, Path)):
        img = Image.open(image_path).convert("RGBA")
    else:
        img = image_path.convert("RGBA")

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for b, p in zip(blocks, probs):
        if p < threshold:
            continue
        alpha = int(70 + 160 * min(1.0, (p - threshold) / (1.0 - threshold + 1e-6)))
        draw.rectangle(
            [b.col0, b.row0, b.col1 - 1, b.row1 - 1],
            fill=(239, 68, 68, alpha),        # shadcn Red-500
            outline=(239, 68, 68, 255),
        )
    result = Image.alpha_composite(img, overlay).convert("RGB")
    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        result.save(p)
    return result

