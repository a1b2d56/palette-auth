"""Quantitative evaluation benchmark for palette image authentication.

Measures PSNR (imperceptibility), false-positive rate on untouched signed images,
and block-level localization precision, recall, and F1 score.
"""
from __future__ import annotations

import random
import statistics
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from palette_auth import ai_detector as ad
from palette_auth import core, crypto

OUT = Path("eval_output")
N_IMAGES = 8
BLOCK_SIZE = 32


def make_scene(seed: int, size=(256, 256)) -> Image.Image:
    """Generate a synthetic test scene with varied geometric and gradient patterns."""
    rng = random.Random(seed)
    w, h = size
    xx, yy = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    kind = rng.choice(["diag", "radial", "bands"])
    t = {
        "diag": (xx + yy) / 2,
        "radial": np.clip(np.sqrt((xx - 0.5) ** 2 + (yy - 0.5) ** 2) * 1.4, 0, 1),
        "bands": (np.floor(xx * 6) % 2).astype(float),
    }[kind]
    c1 = np.array([rng.randint(0, 255) for _ in range(3)])
    c2 = np.array([rng.randint(0, 255) for _ in range(3)])
    arr = (c1[None, None, :] + (c2 - c1)[None, None, :] * t[..., None]).astype(np.uint8)
    img = Image.fromarray(arr, "RGB")
    draw = ImageDraw.Draw(img)
    for _ in range(rng.randint(2, 4)):
        color = tuple(rng.randint(0, 255) for _ in range(3))
        x0, y0 = rng.randint(0, w - 40), rng.randint(0, h - 40)
        x1 = min(x0 + rng.randint(30, 80), w)
        y1 = min(y0 + rng.randint(30, 80), h)
        (draw.ellipse if rng.random() < 0.5 else draw.rectangle)([x0, y0, x1, y1], fill=color)
    return img.convert("P", palette=Image.ADAPTIVE, colors=256)


def apply_random_tamper(img: Image.Image, rng: random.Random):
    """Repaints a random rectangle -- half the time a solid splice, half
    the time a copy-move from elsewhere in the same image -- preserving
    the palette either way. Returns (tampered_image, (r0, r1, c0, c1))."""
    arr = np.array(img, dtype=np.uint8).copy()
    palette = np.array(img.getpalette(), dtype=np.uint8).reshape(-1, 3)
    h, w = arr.shape
    tw, th = rng.randint(20, 70), rng.randint(20, 70)
    x0, y0 = rng.randint(0, w - tw), rng.randint(0, h - th)

    if rng.random() < 0.5:
        sx0, sy0 = rng.randint(0, w - tw), rng.randint(0, h - th)
        arr[y0 : y0 + th, x0 : x0 + tw] = arr[sy0 : sy0 + th, sx0 : sx0 + tw]
    else:
        target = np.array([rng.randint(0, 255) for _ in range(3)])
        diffs = palette.astype(np.int32) - target
        idx = int(np.argmin((diffs ** 2).sum(axis=1)))
        arr[y0 : y0 + th, x0 : x0 + tw] = idx

    out = Image.fromarray(arr, "P")
    out.putpalette(palette.flatten().tolist())
    return out, (y0, y0 + th, x0, x0 + tw)


def block_ground_truth(blocks, region) -> set[int]:
    r0, r1, c0, c1 = region
    return {b.index for b in blocks if b.row0 < r1 and b.row1 > r0 and b.col0 < c1 and b.col1 > c0}


def prf(flagged: set[int], truth: set[int]) -> tuple[float, float, float]:
    tp, fp, fn = len(flagged & truth), len(flagged - truth), len(truth - flagged)
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 20 * np.log10(255.0) - 10 * np.log10(mse)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    priv, pub = crypto.generate_keypair()
    priv_path, pub_path = OUT / "eval.private.pem", OUT / "eval.public.pem"
    crypto.save_private_key(priv, priv_path)
    crypto.save_public_key(pub, pub_path)

    print("Training the AI detector once for this evaluation run (~15s)...")
    ai_model, _ = ad.train(n_per_class=1000, epochs=8, seed=2)

    psnrs, crypto_fp, ai_fp = [], [], []
    crypto_scores, ai_scores = [], []

    for i in range(N_IMAGES):
        rng = random.Random(1000 + i)
        scene = make_scene(seed=i)
        orig_p, signed_p, tampered_p = (OUT / f"scene_{i}_{s}.png" for s in ("orig", "signed", "tampered"))
        scene.save(orig_p)
        info = core.sign_image(orig_p, signed_p, priv_path, block_size=BLOCK_SIZE)

        psnrs.append(psnr(np.array(Image.open(orig_p).convert("RGB")), np.array(Image.open(signed_p).convert("RGB"))))

        clean = core.verify_image(signed_p, pub_path)
        crypto_fp.append(len(clean.tampered_blocks) / info["n_blocks"])
        _, clean_probs = ad.predict_heatmap(signed_p, ai_model, block_size=BLOCK_SIZE)
        ai_fp.append(float((clean_probs > 0.5).mean()))

        tampered_img, region = apply_random_tamper(Image.open(signed_p), rng)
        tampered_img.save(tampered_p)

        result = core.verify_image(tampered_p, pub_path)
        truth = block_ground_truth(result.all_blocks, region)
        flagged = {b.index for b in result.tampered_blocks}
        crypto_scores.append(prf(flagged, truth))

        _, probs = ad.predict_heatmap(tampered_p, ai_model, block_size=BLOCK_SIZE)
        ai_flagged = {result.all_blocks[j].index for j in range(len(result.all_blocks)) if probs[j] > 0.5}
        ai_scores.append(prf(ai_flagged, truth))

        print(f"scene {i}: psnr={psnrs[-1]:.1f}dB  crypto_prf={tuple(round(v,2) for v in crypto_scores[-1])}  "
              f"ai_prf={tuple(round(v,2) for v in ai_scores[-1])}")

    def summarize(name, vals):
        p, r, f = (statistics.mean(v[i] for v in vals) for i in range(3))
        print(f"{name:>28}: precision={p:.2f}  recall={r:.2f}  F1={f:.2f}")

    print(f"\n{'Imperceptibility':>28}: mean PSNR (original vs signed) = {statistics.mean(psnrs):.1f} dB over {N_IMAGES} images")
    print(f"{'Crypto false-positive rate':>28}: {statistics.mean(crypto_fp)*100:.2f}% of blocks on untouched signed images")
    print(f"{'AI false-positive rate':>28}: {statistics.mean(ai_fp)*100:.2f}% of blocks on untouched signed images")
    print()
    summarize("Crypto localization", crypto_scores)
    summarize("AI detector", ai_scores)


if __name__ == "__main__":
    main()
