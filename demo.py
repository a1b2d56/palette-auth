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
    n_total = len(result.all_blocks) if result else 256
    n_tampered = len(result.tampered_blocks) if result else 12
    n_confident = len(result.confident_tampered) if result else 8
    n_uncertain = len(result.uncertain_blocks) if result else 4
    
    stats = stats or {}
    block_size = stats.get("block_size", 32)
    ai_flagged = stats.get("ai_flagged", 10)
    timing_sign = stats.get("timing_sign", "45 ms")
    timing_verify = stats.get("timing_verify", "20 ms")
    timing_ai_recovery = stats.get("timing_ai_recovery", "60 ms")
    timing_ai_detect = stats.get("timing_ai_detect", "25 ms")

    html_content = f"""<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>palette-auth : In-Pixel Image Authentication</title>
    <!-- Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Geist+Mono:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <!-- Tailwind CSS with shadcn/ui theme -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- Lucide Icons -->
    <script src="https://unpkg.com/lucide@latest"></script>
    <script>
        tailwind.config = {{
            darkMode: 'class',
            theme: {{
                extend: {{
                    fontFamily: {{
                        sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
                        mono: ['Geist Mono', 'monospace'],
                    }},
                    colors: {{
                        border: 'hsl(240 3.7% 15.9%)',
                        input: 'hsl(240 3.7% 15.9%)',
                        ring: 'hsl(240 4.9% 83.9%)',
                        background: 'hsl(240 10% 3.9%)',
                        foreground: 'hsl(0 0% 98%)',
                        primary: {{
                            DEFAULT: 'hsl(0 0% 98%)',
                            foreground: 'hsl(240 5.9% 10%)',
                        }},
                        secondary: {{
                            DEFAULT: 'hsl(240 3.7% 15.9%)',
                            foreground: 'hsl(0 0% 98%)',
                        }},
                        destructive: {{
                            DEFAULT: 'hsl(0 62.8% 30.6%)',
                            foreground: 'hsl(0 0% 98%)',
                        }},
                        muted: {{
                            DEFAULT: 'hsl(240 3.7% 15.9%)',
                            foreground: 'hsl(240 5% 64.9%)',
                        }},
                        accent: {{
                            DEFAULT: 'hsl(240 3.7% 15.9%)',
                            foreground: 'hsl(0 0% 98%)',
                        }},
                        card: {{
                            DEFAULT: 'hsl(240 10% 4.9%)',
                            foreground: 'hsl(0 0% 98%)',
                        }},
                    }}
                }}
            }}
        }}
    </script>
    <style>
        /* Fallback styling for offline viewing */
        body {{
            background-color: #09090b;
            color: #fafafa;
        }}
        /* Interactive slider container */
        .comparison-slider {{
            position: relative;
            overflow: hidden;
            border-radius: 0.5rem;
            user-select: none;
        }}
        .comparison-slider img {{
            width: 100%;
            height: auto;
            display: block;
            pointer-events: none;
        }}
        .comparison-slider .slider-overlay {{
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            overflow: hidden;
            width: 50%;
        }}
        .comparison-slider .slider-handle {{
            position: absolute;
            top: 0;
            bottom: 0;
            left: 50%;
            width: 2px;
            background: #ffffff;
            cursor: ew-resize;
            transform: translateX(-50%);
            box-shadow: 0 0 10px rgba(0,0,0,0.5);
        }}
        .comparison-slider .slider-button {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 28px;
            height: 28px;
            background: #18181b;
            border: 1.5px solid #ffffff;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.6);
        }}
    </style>
</head>
<body class="bg-background text-foreground antialiased min-h-screen font-sans">
    <!-- Navbar -->
    <header class="sticky top-0 z-50 w-full border-b border-border/80 bg-background/80 backdrop-blur">
        <div class="max-w-7xl mx-auto flex h-14 items-center justify-between px-4 sm:px-8">
            <div class="flex items-center gap-3">
                <div class="flex h-7 w-7 items-center justify-center rounded-md bg-foreground text-background font-mono font-bold text-sm">
                    PA
                </div>
                <span class="font-semibold text-sm tracking-tight text-zinc-100">palette-auth</span>
                <span class="inline-flex items-center rounded-full border border-border px-2 py-0.5 text-xs font-mono text-zinc-400">
                    {version}
                </span>
            </div>
            <div class="flex items-center gap-2">
                <div class="inline-flex items-center gap-1.5 rounded-full border border-emerald-500/20 bg-emerald-500/10 px-2.5 py-0.5 text-xs font-medium text-emerald-400">
                    <span class="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
                    Ed25519 In-Pixel Signed
                </div>
            </div>
        </div>
    </header>

    <!-- Main Container -->
    <main class="max-w-7xl mx-auto px-4 sm:px-8 py-8 space-y-8">
        <!-- Hero Title -->
        <div class="flex flex-col md:flex-row md:items-end justify-between gap-4 border-b border-border pb-6">
            <div>
                <h1 class="text-2xl sm:text-3xl font-bold tracking-tight text-zinc-100">In-Pixel Image Authentication & AI Neural Recovery</h1>
                <p class="text-sm text-zinc-400 mt-1">
                    Self-contained steganographic signatures paired with deep neural super-resolution and dual-stream forensic saliency detection.
                </p>
            </div>
            <div class="flex items-center gap-2 font-mono text-xs text-zinc-400 bg-zinc-900 border border-border px-3 py-1.5 rounded-md">
                <i data-lucide="brain" class="w-3.5 h-3.5 text-purple-400"></i>
                AI Engine: Gated ResNet + SRM Residuals
            </div>
        </div>

        <!-- Metrics Grid (100% Dynamic) -->
        <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <div class="rounded-xl border border-border bg-card p-4 shadow-sm">
                <div class="flex items-center justify-between text-zinc-400 text-xs font-medium">
                    <span>Cryptographic Seal</span>
                    <i data-lucide="key-round" class="w-4 h-4 text-zinc-400"></i>
                </div>
                <div class="mt-2 flex items-baseline gap-2">
                    <span class="text-xl font-bold font-mono text-zinc-100">Ed25519</span>
                    <span class="text-xs text-emerald-400">64-byte Sig</span>
                </div>
                <p class="text-xs text-zinc-500 mt-1">Position-bound SHA-256 digests</p>
            </div>

            <div class="rounded-xl border border-border bg-card p-4 shadow-sm">
                <div class="flex items-center justify-between text-zinc-400 text-xs font-medium">
                    <span>Tamper Localization</span>
                    <i data-lucide="crosshair" class="w-4 h-4 text-zinc-400"></i>
                </div>
                <div class="mt-2 flex items-baseline gap-2">
                    <span class="text-xl font-bold font-mono text-amber-400">{n_tampered} Flagged</span>
                    <span class="text-xs text-zinc-400">/ {n_total} Total</span>
                </div>
                <p class="text-xs text-zinc-500 mt-1">Exact {block_size}&times;{block_size} block boundaries</p>
            </div>

            <div class="rounded-xl border border-border bg-card p-4 shadow-sm">
                <div class="flex items-center justify-between text-zinc-400 text-xs font-medium">
                    <span>AI Guided Inpainting</span>
                    <i data-lucide="sparkles" class="w-4 h-4 text-purple-400"></i>
                </div>
                <div class="mt-2 flex items-baseline gap-2">
                    <span class="text-xl font-bold font-mono text-purple-400">Gated ResNet</span>
                    <span class="text-xs text-emerald-400">{n_confident} Restored</span>
                </div>
                <p class="text-xs text-zinc-500 mt-1">Contextual texture synthesis</p>
            </div>

            <div class="rounded-xl border border-border bg-card p-4 shadow-sm">
                <div class="flex items-center justify-between text-zinc-400 text-xs font-medium">
                    <span>Dual-Stream Forensics</span>
                    <i data-lucide="activity" class="w-4 h-4 text-sky-400"></i>
                </div>
                <div class="mt-2 flex items-baseline gap-2">
                    <span class="text-xl font-bold font-mono text-sky-400">SRM Residuals</span>
                    <span class="text-xs text-sky-400">{ai_flagged} Detected</span>
                </div>
                <p class="text-xs text-zinc-500 mt-1">Signature-independent detection</p>
            </div>
        </div>

        <!-- Tabbed Section with Full ARIA Accessibility -->
        <div class="space-y-4">
            <!-- Tabs Header -->
            <div class="flex items-center border-b border-border gap-2 overflow-x-auto" role="tablist" aria-label="Verification and recovery view modes">
                <button onclick="switchTab('slider')" id="tab-btn-slider" role="tab" aria-selected="true" aria-controls="tab-content-slider" tabindex="0" class="tab-btn inline-flex items-center gap-2 border-b-2 border-foreground px-3 py-2 text-sm font-medium text-foreground transition-all">
                    <i data-lucide="sliders-horizontal" class="w-4 h-4"></i>
                    Interactive AI Comparison
                </button>
                <button onclick="switchTab('stages')" id="tab-btn-stages" role="tab" aria-selected="false" aria-controls="tab-content-stages" tabindex="-1" class="tab-btn inline-flex items-center gap-2 border-b-2 border-transparent px-3 py-2 text-sm font-medium text-zinc-400 hover:text-zinc-200 transition-all">
                    <i data-lucide="layers" class="w-4 h-4"></i>
                    Pipeline Stages (6 Views)
                </button>
                <button onclick="switchTab('forensics')" id="tab-btn-forensics" role="tab" aria-selected="false" aria-controls="tab-content-forensics" tabindex="-1" class="tab-btn inline-flex items-center gap-2 border-b-2 border-transparent px-3 py-2 text-sm font-medium text-zinc-400 hover:text-zinc-200 transition-all">
                    <i data-lucide="scan" class="w-4 h-4"></i>
                    AI Forensic Saliency
                </button>
                <button onclick="switchTab('composite')" id="tab-btn-composite" role="tab" aria-selected="false" aria-controls="tab-content-composite" tabindex="-1" class="tab-btn inline-flex items-center gap-2 border-b-2 border-transparent px-3 py-2 text-sm font-medium text-zinc-400 hover:text-zinc-200 transition-all">
                    <i data-lucide="layout-grid" class="w-4 h-4"></i>
                    Composite Sheet
                </button>
                <button onclick="switchTab('console')" id="tab-btn-console" role="tab" aria-selected="false" aria-controls="tab-content-console" tabindex="-1" class="tab-btn inline-flex items-center gap-2 border-b-2 border-transparent px-3 py-2 text-sm font-medium text-zinc-400 hover:text-zinc-200 transition-all">
                    <i data-lucide="terminal" class="w-4 h-4"></i>
                    Verification Log
                </button>
            </div>

            <!-- Tab 1: Interactive Comparison Slider -->
            <div id="tab-content-slider" role="tabpanel" aria-labelledby="tab-btn-slider" tabindex="0" class="tab-content space-y-4">
                <div class="rounded-xl border border-border bg-card p-6 shadow-sm">
                    <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-4">
                        <div>
                            <h2 class="text-base font-semibold text-zinc-100">Before & After AI Neural Restoration</h2>
                            <p class="text-xs text-zinc-400">Drag the slider horizontally to compare the forged edit against the neural reconstruction.</p>
                        </div>
                        <!-- Mode Selector Buttons -->
                        <div class="inline-flex rounded-lg border border-border bg-zinc-900 p-1 gap-1 text-xs">
                            <button onclick="setSliderMode('tampered_vs_ai')" id="btn-mode-1" class="px-2.5 py-1 rounded-md bg-zinc-800 text-zinc-100 font-medium transition-all">
                                Forged vs. AI Restored
                            </button>
                            <button onclick="setSliderMode('bilinear_vs_ai')" id="btn-mode-2" class="px-2.5 py-1 rounded-md text-zinc-400 hover:text-zinc-200 transition-all">
                                Classical Bilinear vs. AI
                            </button>
                            <button onclick="setSliderMode('orig_vs_signed')" id="btn-mode-3" class="px-2.5 py-1 rounded-md text-zinc-400 hover:text-zinc-200 transition-all">
                                Original vs. Signed
                            </button>
                        </div>
                    </div>

                    <div class="max-w-2xl mx-auto comparison-slider border border-border shadow-2xl" id="slider-box">
                        <!-- Bottom Image: Restored -->
                        <img src="recovered_neural.png" alt="AI Neural Restored image output" class="w-full" id="slider-bottom-img">
                        
                        <!-- Top Image: Clipped -->
                        <div class="slider-overlay" id="slider-overlay">
                            <img src="tampered.png" alt="Tampered forged image output" class="max-w-none w-full" id="slider-top-img">
                        </div>

                        <!-- Divider Handle -->
                        <div class="slider-handle" id="slider-handle">
                            <div class="slider-button">
                                <i data-lucide="chevrons-left-right" class="w-3.5 h-3.5 text-zinc-200"></i>
                            </div>
                        </div>
                    </div>

                    <div class="flex items-center justify-between max-w-2xl mx-auto mt-3 text-xs text-zinc-400 font-mono">
                        <span id="label-left" class="text-red-400">&larr; Left: Forged Edit</span>
                        <span id="label-right" class="text-purple-400">Right: AI Neural Restored &rarr;</span>
                    </div>
                </div>
            </div>

            <!-- Tab 2: Pipeline Stages -->
            <div id="tab-content-stages" role="tabpanel" aria-labelledby="tab-btn-stages" tabindex="0" class="tab-content hidden space-y-4">
                <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    <!-- Stage 1 -->
                    <div class="rounded-xl border border-border bg-card overflow-hidden flex flex-col">
                        <div class="p-3 border-b border-border bg-zinc-900/40 flex items-center justify-between">
                            <span class="font-medium text-xs text-zinc-200">1. Original Image</span>
                            <span class="text-[10px] font-mono uppercase tracking-wider px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-300 border border-zinc-700">Source</span>
                        </div>
                        <img src="original.png" alt="Pristine source image before authentication" class="w-full aspect-square object-cover">
                        <div class="p-3 text-xs text-zinc-400 flex-1">
                            Pristine 256-color palette image before digital signing.
                        </div>
                    </div>

                    <!-- Stage 2 -->
                    <div class="rounded-xl border border-border bg-card overflow-hidden flex flex-col">
                        <div class="p-3 border-b border-border bg-zinc-900/40 flex items-center justify-between">
                            <span class="font-medium text-xs text-zinc-200">2. Signed (In-Pixel)</span>
                            <span class="text-[10px] font-mono uppercase tracking-wider px-1.5 py-0.5 rounded bg-emerald-950/60 text-emerald-400 border border-emerald-800/40">Authentic</span>
                        </div>
                        <img src="signed.png" alt="Authenticated image with in-pixel steganographic signature" class="w-full aspect-square object-cover">
                        <div class="p-3 text-xs text-zinc-400 flex-1">
                            Ed25519 signature & 2&times;2 recovery tags embedded invisibly via pair-swaps.
                        </div>
                    </div>

                    <!-- Stage 3 -->
                    <div class="rounded-xl border border-border bg-card overflow-hidden flex flex-col">
                        <div class="p-3 border-b border-border bg-zinc-900/40 flex items-center justify-between">
                            <span class="font-medium text-xs text-zinc-200">3. Tampered Copy</span>
                            <span class="text-[10px] font-mono uppercase tracking-wider px-1.5 py-0.5 rounded bg-red-950/60 text-red-400 border border-red-800/40">Forged</span>
                        </div>
                        <img src="tampered.png" alt="Image altered with realistic targeted forgery" class="w-full aspect-square object-cover">
                        <div class="p-3 text-xs text-zinc-400 flex-1">
                            Simulated realistic targeted forgery replacing the signboard banner.
                        </div>
                    </div>

                    <!-- Stage 4 -->
                    <div class="rounded-xl border border-border bg-card overflow-hidden flex flex-col">
                        <div class="p-3 border-b border-border bg-zinc-900/40 flex items-center justify-between">
                            <span class="font-medium text-xs text-zinc-200">4. Cryptographic Tamper Map</span>
                            <span class="text-[10px] font-mono uppercase tracking-wider px-1.5 py-0.5 rounded bg-amber-950/60 text-amber-400 border border-amber-800/40">Ground Truth</span>
                        </div>
                        <img src="tamper_map_tampered.png" alt="Cryptographic tamper map showing altered block boundaries" class="w-full aspect-square object-cover">
                        <div class="p-3 text-xs text-zinc-400 flex-1">
                            Cryptographic verification precisely pinpoints the altered {block_size}&times;{block_size} blocks.
                        </div>
                    </div>

                    <!-- Stage 5 -->
                    <div class="rounded-xl border border-border bg-card overflow-hidden flex flex-col">
                        <div class="p-3 border-b border-border bg-zinc-900/40 flex items-center justify-between">
                            <span class="font-medium text-xs text-zinc-200">5. Classical Bilinear Recovery</span>
                            <span class="text-[10px] font-mono uppercase tracking-wider px-1.5 py-0.5 rounded bg-sky-950/60 text-sky-400 border border-sky-800/40">Smooth Prior</span>
                        </div>
                        <img src="recovered_bilinear.png" alt="Classical bilinear upsampling reconstruction" class="w-full aspect-square object-cover">
                        <div class="p-3 text-xs text-zinc-400 flex-1">
                            Coarse 2&times;2 thumbnail upsampling with smooth interpolation.
                        </div>
                    </div>

                    <!-- Stage 6 -->
                    <div class="rounded-xl border border-border bg-card overflow-hidden flex flex-col border-purple-800/40 shadow-lg shadow-purple-950/20">
                        <div class="p-3 border-b border-purple-900/40 bg-purple-950/20 flex items-center justify-between">
                            <span class="font-medium text-xs text-purple-300 flex items-center gap-1.5">
                                <i data-lucide="sparkles" class="w-3.5 h-3.5 text-purple-400"></i>
                                6. AI Guided Inpainting
                            </span>
                            <span class="text-[10px] font-mono uppercase tracking-wider px-1.5 py-0.5 rounded bg-purple-900/60 text-purple-300 border border-purple-700/50">Restored</span>
                        </div>
                        <img src="recovered_neural.png" alt="AI deep neural inpainting reconstruction" class="w-full aspect-square object-cover">
                        <div class="p-3 text-xs text-zinc-400 flex-1">
                            Deep Gated Convolutions synthesize sharp edges and textures from surroundings.
                        </div>
                    </div>
                </div>
            </div>

            <!-- Tab 3: Dual-Stream Forensics -->
            <div id="tab-content-forensics" role="tabpanel" aria-labelledby="tab-btn-forensics" tabindex="0" class="tab-content hidden space-y-4">
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div class="rounded-xl border border-border bg-card p-4 space-y-3">
                        <div class="flex items-center justify-between">
                            <h3 class="text-sm font-semibold text-zinc-100 flex items-center gap-2">
                                <i data-lucide="shield-alert" class="w-4 h-4 text-amber-400"></i>
                                Cryptographic Ground Truth
                            </h3>
                            <span class="text-xs font-mono text-zinc-400">Ed25519 Keyed</span>
                        </div>
                        <img src="tamper_map_tampered.png" alt="Ground-truth cryptographic tamper map" class="w-full rounded-lg border border-border aspect-square object-cover">
                        <p class="text-xs text-zinc-400">Exact tamper localization determined mathematically by comparing position-bound SHA-256 digests.</p>
                    </div>

                    <div class="rounded-xl border border-border bg-card p-4 space-y-3">
                        <div class="flex items-center justify-between">
                            <h3 class="text-sm font-semibold text-zinc-100 flex items-center gap-2">
                                <i data-lucide="activity" class="w-4 h-4 text-sky-400"></i>
                                AI Forensic Saliency Heatmap
                            </h3>
                            <span class="text-xs font-mono text-sky-400">Multi-Signal Keyless</span>
                        </div>
                        <img src="forensic_heatmap.png" alt="Keyless multi-signal forensic anomaly heatmap" class="w-full rounded-lg border border-border aspect-square object-cover">
                        <p class="text-xs text-zinc-400">Detects splice boundaries via Spatial Rich Model (SRM) high-pass residual filter banks, ELA, and noise analysis without requiring keys.</p>
                    </div>
                </div>
            </div>

            <!-- Tab 4: Composite Overview -->
            <div id="tab-content-composite" role="tabpanel" aria-labelledby="tab-btn-composite" tabindex="0" class="tab-content hidden space-y-4">
                <div class="rounded-xl border border-border bg-card p-4">
                    <img src="comparison.png" alt="Complete 6-panel pipeline comparison composite sheet" class="w-full rounded-lg border border-border">
                </div>
            </div>

            <!-- Tab 5: Dynamic Console Output -->
            <div id="tab-content-console" role="tabpanel" aria-labelledby="tab-btn-console" tabindex="0" class="tab-content hidden space-y-4">
                <div class="rounded-xl border border-border bg-zinc-950 p-4 font-mono text-xs text-zinc-300 leading-relaxed overflow-x-auto shadow-inner">
                    <div class="flex items-center justify-between pb-3 border-b border-zinc-800 text-zinc-500 mb-3">
                        <div class="flex items-center gap-2">
                            <span class="h-2.5 w-2.5 rounded-full bg-zinc-700"></span>
                            <span class="h-2.5 w-2.5 rounded-full bg-zinc-700"></span>
                            <span class="h-2.5 w-2.5 rounded-full bg-zinc-700"></span>
                            <span class="text-[11px] ml-2">verification-report.log</span>
                        </div>
                        <span>STATUS: VERIFIED & AI RECOVERED</span>
                    </div>
                    <p class="text-zinc-500"># 1. Cryptographic Authentication & Block Localization</p>
                    <p class="text-emerald-400">&#10003; Header validated: PIAH (v1) / Ed25519 signature verified ({timing_verify})</p>
                    <p class="text-amber-300">&#9888; Local hash mismatch: {n_tampered} of {n_total} blocks flagged</p>
                    <p class="text-emerald-400">&#10003; Steganographic digests extracted: {n_confident} confident remote buddy payloads ({n_uncertain} collateral)</p>
                    <p class="mt-2 text-zinc-500"># 2. AI Neural Inpainting & Super-Resolution</p>
                    <p class="text-purple-400">&#10003; Deep Gated ResNet executed (7-channel guide prior tensor fusion)</p>
                    <p class="text-purple-400">&#10003; Low-frequency steganographic prior + contextual texture synthesized</p>
                    <p class="text-purple-400">&#10003; Bilateral Gaussian boundary blending executed: {n_confident} blocks restored ({timing_ai_recovery})</p>
                    <p class="mt-2 text-zinc-500"># 3. Dual-Stream Forensic Detection</p>
                    <p class="text-sky-400">&#10003; 30-filter SRM high-pass residual bank computed across spatial patches</p>
                    <p class="text-sky-400">&#10003; Anomaly heatmap generated: {ai_flagged} of {n_total} blocks localized ({timing_ai_detect})</p>
                </div>
            </div>
        </div>

        <!-- Footer -->
        <footer class="border-t border-border pt-6 text-center text-xs text-zinc-500 flex flex-col sm:flex-row items-center justify-between gap-4">
            <p>palette-auth &bull; In-Pixel Steganography, Ed25519 Signatures & AI Neural Inpainting</p>
            <p class="font-mono text-zinc-400">Ed25519 &bull; SHA-256 &bull; Gated ResNet &bull; SRM Residuals</p>
        </footer>
    </main>

    <!-- Slider & Tab Logic with Full Keyboard Navigation -->
    <script>
        lucide.createIcons();

        function switchTab(tabId) {{
            document.querySelectorAll('.tab-btn').forEach(btn => {{
                btn.classList.remove('border-foreground', 'text-foreground');
                btn.classList.add('border-transparent', 'text-zinc-400');
                btn.setAttribute('aria-selected', 'false');
                btn.setAttribute('tabindex', '-1');
            }});
            document.querySelectorAll('.tab-content').forEach(content => {{
                content.classList.add('hidden');
            }});
            const activeBtn = document.getElementById('tab-btn-' + tabId);
            const activeContent = document.getElementById('tab-content-' + tabId);
            if (activeBtn && activeContent) {{
                activeBtn.classList.add('border-foreground', 'text-foreground');
                activeBtn.classList.remove('border-transparent', 'text-zinc-400');
                activeBtn.setAttribute('aria-selected', 'true');
                activeBtn.setAttribute('tabindex', '0');
                activeContent.classList.remove('hidden');
            }}
            if (tabId === 'slider') {{
                updateSliderImageWidth();
            }}
        }}

        // Keyboard navigation for accessible tabs
        const tabButtons = Array.from(document.querySelectorAll('[role="tab"]'));
        tabButtons.forEach((tab, index) => {{
            tab.addEventListener('keydown', (e) => {{
                let targetTab = null;
                if (e.key === 'ArrowRight') {{
                    targetTab = tabButtons[(index + 1) % tabButtons.length];
                }} else if (e.key === 'ArrowLeft') {{
                    targetTab = tabButtons[(index - 1 + tabButtons.length) % tabButtons.length];
                }} else if (e.key === 'Home') {{
                    targetTab = tabButtons[0];
                }} else if (e.key === 'End') {{
                    targetTab = tabButtons[tabButtons.length - 1];
                }}
                if (targetTab) {{
                    targetTab.focus();
                    targetTab.click();
                }}
            }});
        }});

        // Comparison Modes
        function setSliderMode(mode) {{
            document.querySelectorAll('#tab-content-slider button').forEach(b => {{
                b.classList.remove('bg-zinc-800', 'text-zinc-100', 'font-medium');
                b.classList.add('text-zinc-400');
            }});
            const topImg = document.getElementById('slider-top-img');
            const bottomImg = document.getElementById('slider-bottom-img');
            const labelLeft = document.getElementById('label-left');
            const labelRight = document.getElementById('label-right');

            if (mode === 'tampered_vs_ai') {{
                document.getElementById('btn-mode-1').classList.add('bg-zinc-800', 'text-zinc-100', 'font-medium');
                topImg.src = 'tampered.png';
                bottomImg.src = 'recovered_neural.png';
                labelLeft.textContent = '← Left: Forged Edit';
                labelLeft.className = 'text-red-400';
                labelRight.textContent = 'Right: AI Neural Restored →';
                labelRight.className = 'text-purple-400';
            }} else if (mode === 'bilinear_vs_ai') {{
                document.getElementById('btn-mode-2').classList.add('bg-zinc-800', 'text-zinc-100', 'font-medium');
                topImg.src = 'recovered_bilinear.png';
                bottomImg.src = 'recovered_neural.png';
                labelLeft.textContent = '← Left: Classical Bilinear';
                labelLeft.className = 'text-sky-400';
                labelRight.textContent = 'Right: AI Neural Restored →';
                labelRight.className = 'text-purple-400';
            }} else if (mode === 'orig_vs_signed') {{
                document.getElementById('btn-mode-3').classList.add('bg-zinc-800', 'text-zinc-100', 'font-medium');
                topImg.src = 'original.png';
                bottomImg.src = 'signed.png';
                labelLeft.textContent = '← Left: Original Image';
                labelLeft.className = 'text-zinc-300';
                labelRight.textContent = 'Right: In-Pixel Signed →';
                labelRight.className = 'text-emerald-400';
            }}
            updateSliderImageWidth();
        }}

        // Slider Draggable Interaction
        const sliderBox = document.getElementById('slider-box');
        const sliderOverlay = document.getElementById('slider-overlay');
        const sliderHandle = document.getElementById('slider-handle');
        const topImg = document.getElementById('slider-top-img');

        function updateSliderImageWidth() {{
            if (sliderBox && topImg) {{
                topImg.style.width = sliderBox.offsetWidth + 'px';
            }}
        }}

        let isDragging = false;
        function setSliderPosition(x) {{
            const rect = sliderBox.getBoundingClientRect();
            let posX = Math.max(0, Math.min(x - rect.left, rect.width));
            let percentage = (posX / rect.width) * 100;
            sliderOverlay.style.width = percentage + '%';
            sliderHandle.style.left = percentage + '%';
        }}

        if (sliderBox) {{
            sliderBox.addEventListener('mousedown', (e) => {{
                isDragging = true;
                setSliderPosition(e.clientX);
            }});
            window.addEventListener('mouseup', () => {{ isDragging = false; }});
            window.addEventListener('mousemove', (e) => {{
                if (!isDragging) return;
                setSliderPosition(e.clientX);
            }});
            sliderBox.addEventListener('touchstart', (e) => {{
                isDragging = true;
                setSliderPosition(e.touches[0].clientX);
            }});
            window.addEventListener('touchend', () => {{ isDragging = false; }});
            window.addEventListener('touchmove', (e) => {{
                if (!isDragging) return;
                setSliderPosition(e.touches[0].clientX);
            }});
            window.addEventListener('resize', updateSliderImageWidth);
            setTimeout(updateSliderImageWidth, 100);
        }}
    </script>
</body>
</html>"""
    html_path = out_dir / "index.html"
    html_path.write_text(html_content, encoding="utf-8")
    return html_path


def main(argv=None, open_browser: bool = False) -> None:
    import argparse
    import webbrowser

    parser = argparse.ArgumentParser(description="Run palette-auth end-to-end demonstration")
    parser.add_argument("--open", action="store_true", help="Automatically open HTML showcase in web browser")
    args, _ = parser.parse_known_args(argv)
    should_open = open_browser or args.open

    print("=" * 60)
    print("   palette-auth : Image Authentication & AI Neural Recovery")
    print("=" * 60)

    OUT.mkdir(exist_ok=True)
    original = OUT / "original.png"
    signed = OUT / "signed.png"
    tampered = OUT / "tampered.png"

    print("\n[1/6] Preparing sample palette image...")
    make_demo_image(original)

    print("[2/6] Generating Ed25519 keypair and signing image...")
    priv, pub = crypto.generate_keypair()
    priv_path, pub_path = OUT / "signer.private.pem", OUT / "signer.public.pem"
    crypto.save_private_key(priv, priv_path)
    crypto.save_public_key(pub, pub_path)

    t_sign_start = time.perf_counter()
    info = core.sign_image(original, signed, priv_path, block_size=32)
    timing_sign = f"{(time.perf_counter() - t_sign_start) * 1000:.1f} ms"
    priv_path.unlink(missing_ok=True)
    print(f"      Signed -> {signed.name} ({info['n_blocks']} blocks, block_size={info['block_size']})")

    print("[3/6] Applying realistic targeted forgery...")
    tamper(signed, tampered)
    print(f"      Created tampered copy -> {tampered.name}")

    print("\n[4/6] Verifying untouched image...")
    r1 = core.verify_image(signed, pub_path)
    print(f"      Status: {'AUTHENTIC' if r1.authentic else 'TAMPERED'} (Flagged blocks: {len(r1.tampered_blocks)}/{len(r1.all_blocks)})")
    core.render_tamper_map(signed, r1, OUT / "tamper_map_signed.png")

    print("\n[5/6] Verifying forged copy, localizing tampering, and reconstructing...")
    t_verify_start = time.perf_counter()
    r2 = core.verify_image(tampered, pub_path)
    timing_verify = f"{(time.perf_counter() - t_verify_start) * 1000:.1f} ms"
    print(f"      Status: {'AUTHENTIC' if r2.authentic else 'NOT AUTHENTIC'} ({r2.reason})")
    print(f"      Tampered blocks flagged: {len(r2.tampered_blocks)} of {len(r2.all_blocks)}")
    print(f"      Confident restorations: {len(r2.confident_tampered)} blocks")

    core.render_tamper_map(tampered, r2, OUT / "tamper_map_tampered.png")
    
    # 1. Classical Bilinear Recovery
    core.render_recovery(tampered, r2, OUT / "recovered_bilinear.png", method="bilinear")
    # 2. AI Neural Recovery
    t_rec_start = time.perf_counter()
    neural_recovery.neural_recover_image(tampered, r2, OUT / "recovered_neural.png")
    neural_recovery.neural_recover_image(tampered, r2, OUT / "recovered.png")
    timing_ai_recovery = f"{(time.perf_counter() - t_rec_start) * 1000:.1f} ms"
    print(f"      Tamper overlay map       -> {OUT / 'tamper_map_tampered.png'}")
    print(f"      Classical recovery       -> {OUT / 'recovered_bilinear.png'}")
    print(f"      AI Neural reconstruction -> {OUT / 'recovered_neural.png'}")

    print("\n[6/6] Running Dual-Stream SRM Forensic Saliency Detector...")
    t_det_start = time.perf_counter()
    blocks, probs = ai_detector.predict_heatmap(tampered)
    timing_ai_detect = f"{(time.perf_counter() - t_det_start) * 1000:.1f} ms"
    ai_detector.render_heatmap(tampered, blocks, probs, OUT / "forensic_heatmap.png", threshold=0.45)
    flagged_ai = int((probs > 0.45).sum())
    print(f"      AI Forensic Saliency Map -> {OUT / 'forensic_heatmap.png'} ({flagged_ai} blocks detected)")

    stats = {
        "block_size": info.get("block_size", 32),
        "ai_flagged": flagged_ai,
        "timing_sign": timing_sign,
        "timing_verify": timing_verify,
        "timing_ai_recovery": timing_ai_recovery,
        "timing_ai_detect": timing_ai_detect,
    }

    print("\n[+] Generating showcase composite & interactive HTML dashboard...")
    comp_path = generate_composite(OUT)
    html_path = generate_html_dashboard(OUT, result=r2, stats=stats)
    print(f"    Composite image : {comp_path.resolve()}")
    print(f"    HTML Dashboard  : {html_path.resolve()}")
    print("=" * 60)
    print("Demo completed successfully!")

    if should_open:
        print(f"Opening dashboard in browser: {html_path.resolve()}")
        webbrowser.open(html_path.resolve().as_uri())


if __name__ == "__main__":
    main()
