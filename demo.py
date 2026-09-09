"""End-to-end demo: generate a colorful palette image, sign it, tamper a
copy, and verify both -- producing a tamper map that shows *where* the
tampered copy was altered.

Run with:  python demo.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from palette_auth import ai_detector, core, crypto, neural_recovery

OUT = Path("demo_output")


def make_demo_image(path: Path, size=(512, 512)) -> None:
    sample_path = Path(__file__).parent / "assets" / "sample.png"
    if sample_path.exists():
        img = Image.open(sample_path)
        img.save(path)
        return

    w, h = size
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    xx, yy = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    arr[..., 0] = xx * 255
    arr[..., 1] = yy * 255
    arr[..., 2] = (1 - xx) * (1 - yy) * 255
    img = Image.fromarray(arr, "RGB")
    draw = ImageDraw.Draw(img)
    draw.ellipse([40, 40, 160, 160], fill=(255, 255, 255))
    draw.rectangle([180, 180, 300, 300], fill=(20, 20, 20))
    draw.polygon([(20, 300), (100, 200), (180, 300)], fill=(255, 200, 0))
    img = img.convert("P", palette=Image.ADAPTIVE, colors=256)
    img.save(path)


def tamper(path_in: Path, path_out: Path) -> None:
    """Simulate a forger repainting a region of the signed image, while
    preserving the existing palette -- the realistic case for a palette
    image. (Re-quantizing to a brand-new adaptive palette would shift
    every pixel's index globally, which isn't a "local edit" at all.)"""
    img = Image.open(path_in)
    assert img.mode == "P"
    index_array = np.array(img, dtype=np.uint8).copy()
    palette = np.array(img.getpalette(), dtype=np.uint8).reshape(-1, 3)

    def nearest_index(rgb):
        diffs = palette.astype(np.int32) - np.array(rgb, dtype=np.int32)
        return int(np.argmin((diffs ** 2).sum(axis=1)))

    h, w = index_array.shape
    r0, r1 = int(0.34 * h), int(0.42 * h)
    c0, c1 = int(0.26 * w), int(0.50 * w)

    fake_fill = nearest_index((210, 40, 40))   # vivid red
    fake_mark = nearest_index((255, 240, 190))  # cream yellow
    index_array[r0:r1, c0:c1] = fake_fill
    index_array[r0 + 4 : r1 - 4, c0 + 6 : c1 - 6] = fake_mark

    out = Image.fromarray(index_array, mode="P")
    out.putpalette(palette.flatten().tolist())
    out.save(path_out)


def generate_composite(out_dir: Path) -> Path:
    """Generate a clean 6-panel composite comparison image of the entire pipeline."""
    img_names = [
        ("1. Original (Source)", out_dir / "original.png"),
        ("2. Signed (In-Pixel)", out_dir / "signed.png"),
        ("3. Tampered Copy", out_dir / "tampered.png"),
        ("4. Tamper Map", out_dir / "tamper_map_tampered.png"),
        ("5. Classical Bilinear", out_dir / "recovered_bilinear.png"),
        ("6. AI Neural Restored", out_dir / "recovered_neural.png"),
    ]

    imgs = [Image.open(p).convert("RGB") for _, p in img_names if p.exists()]
    labels = [label for label, p in img_names if p.exists()]
    if len(imgs) != 6:
        imgs = [Image.open(p).convert("RGB") for _, p in img_names[:5] if p.exists()]
        labels = [label for label, p in img_names[:5] if p.exists()]

    header_h = 36
    thumb_w, thumb_h = 360, 360
    cols = 3
    rows = (len(imgs) + cols - 1) // cols

    grid_w = thumb_w * cols + 40
    grid_h = (thumb_h + header_h) * rows + 30
    composite = Image.new("RGB", (grid_w, grid_h), (18, 18, 22))
    draw = ImageDraw.Draw(composite)

    for idx, (label, img) in enumerate(zip(labels, imgs)):
        r = idx // cols
        c = idx % cols
        x = 15 + c * (thumb_w + 10)
        y = 15 + r * (thumb_h + header_h + 10)
        resized = img.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        draw.rectangle([x, y, x + thumb_w, y + header_h], fill=(28, 28, 35))
        draw.text((x + 10, y + 10), label, fill=(230, 230, 235))
        composite.paste(resized, (x, y + header_h))

    comp_path = out_dir / "comparison.png"
    composite.save(comp_path, quality=95)
    return comp_path


def generate_html_dashboard(
    out_dir: Path,
    result: core.VerificationResult | None = None,
    stats: dict | None = None,
) -> Path:
    """Generate a clean, high-precision shadcn/ui visual verification dashboard with dynamic data."""
    import palette_auth

    version = f"v{palette_auth.__version__}"
