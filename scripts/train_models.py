"""Comprehensive heavy-duty GPU training pipeline for palette-auth.

Accelerated on NVIDIA RTX 4060 (or available CUDA GPU / CPU).
Trains:
1. DualStreamForensicNet on 30,000 multi-signal forensic patches (splicing, inpainting, stego seams, ELA)
2. GuidedInpaintingNet on batched canvas scenes with random occlusions & steganographic color priors
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
from PIL import Image

try:
    import torch
    from torch import nn, optim
    from torch.optim.lr_scheduler import CosineAnnealingLR
except ImportError:
    print("Error: PyTorch is required. Install with: pip install torch")
    sys.exit(1)

from palette_auth import ai_detector, neural_recovery

MODELS_DIR = ROOT_DIR / "palette_auth" / "models"


def get_target_device() -> torch.device:
    if torch.cuda.is_available():
        dev_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[Hardware] CUDA GPU Detected: {dev_name} ({vram:.2f} GB VRAM)")
        return torch.device("cuda:0")
    print("[Hardware] CUDA not available, falling back to CPU.")
    return torch.device("cpu")


# ------------------------------------------------ 1. Heavy Forensic Dataset & Training --

def generate_augmented_forensic_dataset(
    n_per_class: int = 15000,
    patch_size: int = 32,
    seed: int = 1337,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate 2 * n_per_class synthetic patches with comprehensive adversarial manipulations."""
    print(f"   Generating {n_per_class * 2:,} diverse forensic patches (Authentic vs Tampered)...")
    rng = np.random.default_rng(seed)
    patches = []
    labels = []

    # 1. Authentic patches (Class 0): smooth gradients, natural texture, authentic grain
    for _ in range(n_per_class):
        base = rng.uniform(0.1, 0.9, (1, 1, 3))
        # Linear or diagonal gradient
        gx = np.linspace(0, rng.uniform(-0.35, 0.35), patch_size).reshape(1, patch_size, 1)
        gy = np.linspace(0, rng.uniform(-0.35, 0.35), patch_size).reshape(patch_size, 1, 1)
        # Authentic sensor noise (Gaussian + subtle Poisson)
        noise = rng.normal(0, rng.uniform(0.005, 0.025), (patch_size, patch_size, 3))
        patch = np.clip(base + gx + gy + noise, 0.0, 1.0)
        # Random 90-degree rotations & flips
        k = rng.integers(0, 4)
        patch = np.rot90(patch, k)
        if rng.random() > 0.5:
            patch = np.fliplr(patch)
        patches.append(patch.astype(np.float32))
        labels.append(0.0)

    # 2. Tampered patches (Class 1): 5 distinct manipulation modes
    for _ in range(n_per_class):
        base = rng.uniform(0.1, 0.9, (patch_size, patch_size, 3))
        mode = rng.integers(0, 5)

        if mode == 0:
            # Mode 0: Sharp edge cut-paste splicing (horizontal, vertical, or diagonal)
            cut = rng.integers(6, patch_size - 6)
            donor = rng.uniform(0.1, 0.9, (patch_size, patch_size, 3))
            if rng.random() > 0.5:
                base[:, cut:] = donor[:, cut:]
            else:
                base[cut:, :] = donor[cut:, :]

        elif mode == 1:
            # Mode 1: Generative / Inpainting blur & smooth hole replacement
            cy, cx = rng.integers(10, patch_size - 10, 2)
            r = rng.integers(5, 14)
            y, x = np.ogrid[:patch_size, :patch_size]
            mask = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
            mean_col = np.mean(base, axis=(0, 1), keepdims=True)
            base[mask] = mean_col + rng.normal(0, 0.005, base[mask].shape)

        elif mode == 2:
            # Mode 2: Steganographic pair-swapping seam & high-frequency edge perturbation
            seam = rng.integers(6, patch_size - 6)
            shift = rng.uniform(0.15, 0.4, (patch_size, 3))
            base[seam, :] = np.clip(base[seam, :] + shift, 0.0, 1.0)

        elif mode == 3:
            # Mode 3: Localized noise discrepancy (splice with different sensor ISO)
            quadrant = rng.integers(0, 4)
            half = patch_size // 2
            slices = [
                (slice(0, half), slice(0, half)),
                (slice(0, half), slice(half, patch_size)),
                (slice(half, patch_size), slice(0, half)),
                (slice(half, patch_size), slice(half, patch_size)),
            ]
            sy, sx = slices[quadrant]
            base[sy, sx] = np.clip(base[sy, sx] + rng.normal(0, 0.08, base[sy, sx].shape), 0.0, 1.0)

        else:
            # Mode 4: Chromatic / luminance contrast boundary step
            cut = rng.integers(8, patch_size - 8)
            base[:cut, :] = np.clip(base[:cut, :] * rng.uniform(1.2, 1.6), 0.0, 1.0)

        # Random 90-degree rotations & flips
        k = rng.integers(0, 4)
        base = np.rot90(base, k)
        if rng.random() > 0.5:
            base = np.fliplr(base)
        patches.append(base.astype(np.float32))
        labels.append(1.0)

    x = np.array(patches, dtype=np.float32)
    y = np.array(labels, dtype=np.float32).reshape(-1, 1)
    perm = rng.permutation(len(x))
    return x[perm], y[perm]


def train_heavy_forensic_detector(device: torch.device) -> Path:
    print("\n" + "=" * 70)
    print("1. Training DualStreamForensicNet (30,000 Patches, 25 Epochs on GPU)")
    print("=" * 70)
    out_path = MODELS_DIR / "forensic_detector.pt"
    t0 = time.perf_counter()

    # Generate 30,000 samples
    x_np, y_np = generate_augmented_forensic_dataset(n_per_class=15000, patch_size=32, seed=42)
    n_val = int(len(x_np) * 0.15)
    x_train, x_val = x_np[n_val:], x_np[:n_val]
    y_train, y_val = y_np[n_val:], y_np[:n_val]

    # Convert to BCHW tensors and transfer to GPU
    x_train_t = torch.from_numpy(np.transpose(x_train, (0, 3, 1, 2))).float().to(device)
    y_train_t = torch.from_numpy(y_train).float().to(device)
    x_val_t = torch.from_numpy(np.transpose(x_val, (0, 3, 1, 2))).float().to(device)
    y_val_t = torch.from_numpy(y_val).float().to(device)

    model = ai_detector.DualStreamForensicNet(pretrained_srm=True).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    epochs = 25
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    batch_size = 128
    n_batches = int(np.ceil(len(x_train_t) / batch_size))

    print(f"   Training {len(x_train_t):,} samples across {n_batches} batches/epoch (Batch Size: {batch_size})...")

    best_val_acc = 0.0
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(x_train_t), device=device)
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

        scheduler.step()
        avg_loss = total_loss / n_batches

        # Validation pass
        model.eval()
        with torch.no_grad():
            val_preds = model(x_val_t)
            val_loss = float(criterion(val_preds, y_val_t).item())
            val_acc = float(((val_preds > 0.5) == y_val_t).float().mean().item()) * 100

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ai_detector.save_model(model.cpu(), out_path)
            model.to(device)

        if ep % 5 == 0 or ep == epochs:
            lr_curr = scheduler.get_last_lr()[0]
            print(f"   Epoch {ep:2d}/{epochs} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {lr_curr:.2e}")

    elapsed = time.perf_counter() - t0
    file_size_kb = out_path.stat().st_size / 1024
    print(f"   [+] Best Validation Accuracy : {best_val_acc:.2f}%")
    print(f"   [+] Training Time            : {elapsed:.1f}s")
    print(f"   [+] Saved Model Weights      : {out_path} ({file_size_kb:.1f} KB)")
    return out_path


# ------------------------------------------------ 2. Heavy Inpainting Training --

def generate_inpainting_scenes(n_scenes: int = 48, size: int = 256, seed: int = 777) -> list[np.ndarray]:
    scenes = []
    rng = np.random.default_rng(seed)
    for _ in range(n_scenes):
        arr = np.zeros((size, size, 3), dtype=np.float32)
        c1 = rng.uniform(0.1, 0.9, 3)
        c2 = rng.uniform(0.1, 0.9, 3)
        xx, yy = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
        for c in range(3):
            arr[..., c] = c1[c] * xx + c2[c] * yy
        for _ in range(rng.integers(4, 10)):
            rx0, ry0 = rng.integers(0, size - 50, 2)
            rw, rh = rng.integers(24, 72, 2)
            arr[ry0 : ry0 + rh, rx0 : rx0 + rw] = rng.uniform(0.1, 0.9, 3)
        arr = np.clip(arr + rng.normal(0, 0.015, arr.shape), 0.0, 1.0)
        scenes.append((arr * 255.0).astype(np.uint8))
    return scenes


def train_heavy_guided_inpainter(device: torch.device) -> Path:
    print("\n" + "=" * 70)
    print("2. Training GuidedInpaintingNet (Batched Multi-Scale Gated ResNet on GPU)")
    print("=" * 70)
    out_path = MODELS_DIR / "guided_inpainter.pt"
    t0 = time.perf_counter()

    scenes = generate_inpainting_scenes(n_scenes=48, size=256)
    patch_size = 64
    crops_per_scene = 25
    total_patches = len(scenes) * crops_per_scene
    print(f"   Synthesizing {total_patches:,} training patches with random block occlusions...")

    # Pre-generate inpainting training tensors
    inp_list, target_list = [], []
    rng = np.random.default_rng(42)
    for img in scenes:
        h, w = img.shape[:2]
        for _ in range(crops_per_scene):
            ry = rng.integers(0, h - patch_size + 1)
            rx = rng.integers(0, w - patch_size + 1)
            target = img[ry : ry + patch_size, rx : rx + patch_size].astype(np.float32) / 255.0

            mask = np.zeros((patch_size, patch_size), dtype=np.float32)
            mh, mw = 32, 32
            my = rng.integers(8, patch_size - mh - 8 + 1)
            mx = rng.integers(8, patch_size - mw - 8 + 1)
            mask[my : my + mh, mx : mx + mw] = 1.0

            guide = target.copy()
            thumb = Image.fromarray((target[my : my + mh, mx : mx + mw] * 255).astype(np.uint8)).resize(
                (2, 2), Image.Resampling.BOX
            )
            up = np.array(thumb.resize((mw, mh), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
            guide[my : my + mh, mx : mx + mw] = up

            masked_canvas = target * (1.0 - mask[..., None])
            inp_7ch = np.concatenate(
                [
                    np.transpose(masked_canvas, (2, 0, 1)),
                    mask[None, ...],
                    np.transpose(guide, (2, 0, 1)),
                ],
                axis=0,
            )
            inp_list.append(inp_7ch)
            target_list.append(np.transpose(target, (2, 0, 1)))

    inp_t = torch.from_numpy(np.array(inp_list, dtype=np.float32)).to(device)
    target_t = torch.from_numpy(np.array(target_list, dtype=np.float32)).to(device)

    model = neural_recovery.GuidedInpaintingNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    epochs = 20
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion_l1 = nn.L1Loss()
    batch_size = 32
    n_batches = int(np.ceil(len(inp_t) / batch_size))

    print(f"   Training {len(inp_t):,} samples across {epochs} epochs (Batch Size: {batch_size})...")

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(inp_t), device=device)
        total_loss = 0.0
        for b in range(n_batches):
            b_idx = perm[b * batch_size : (b + 1) * batch_size]
            bx, by = inp_t[b_idx], target_t[b_idx]

            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion_l1(pred, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        if ep % 5 == 0 or ep == epochs:
            avg_loss = total_loss / n_batches
            print(f"   Epoch {ep:2d}/{epochs} | L1 Reconstruction Loss: {avg_loss:.5f}")

    model.eval()
    torch.save(model.cpu().state_dict(), out_path)
    elapsed = time.perf_counter() - t0
    file_size_kb = out_path.stat().st_size / 1024

    print(f"   [+] Training Time       : {elapsed:.1f}s")
    print(f"   [+] Saved Inpainter     : {out_path} ({file_size_kb:.1f} KB)")
    return out_path


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    device = get_target_device()
    t_start = time.perf_counter()

    print("\nStarting Heavy Production Model Training...")
    m1 = train_heavy_forensic_detector(device)
    m2 = train_heavy_guided_inpainter(device)

    total_time = time.perf_counter() - t_start
    print("\n" + "=" * 70)
    print(f"Heavy training finished successfully in {total_time:.1f}s ({total_time/60:.2f} min)!")
    print(f"  1. {m1.name} ({m1.stat().st_size / 1024:.1f} KB)")
    print(f"  2. {m2.name} ({m2.stat().st_size / 1024:.1f} KB)")
    print("=" * 70)


if __name__ == "__main__":
    main()
