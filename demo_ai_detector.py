"""Demonstration for the convolutional neural network tamper detector."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from demo import make_demo_image, tamper
from palette_auth import ai_detector as ad
from palette_auth import core, crypto

OUT = Path("demo_output")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    original = OUT / "original.png"
    signed = OUT / "signed.png"
    tampered = OUT / "tampered.png"

    if not signed.exists():
        make_demo_image(original)
        priv, pub = crypto.generate_keypair()
        crypto.save_private_key(priv, OUT / "signer.private.pem")
        crypto.save_public_key(pub, OUT / "signer.public.pem")
        core.sign_image(original, signed, OUT / "signer.private.pem", block_size=32)
        tamper(signed, tampered)
        (OUT / "signer.private.pem").unlink()  # don't leave a private key lying around

    print("Training the AI detector on synthetic splice data (~15s on CPU)...")
    model, _ = ad.train(n_per_class=1000, epochs=8, seed=1)
    ad.save_model(model, OUT / "tamper_cnn.pt")

    print("\n--- running the detector (no signature or key involved) ---")
    results = {}
    for name, path in [("original", original), ("signed", signed), ("tampered", tampered)]:
        blocks, probs = ad.predict_heatmap(path, model, block_size=32)
        results[name] = probs
        flagged = int((probs > 0.5).sum())
        print(f"{name:>9}: {flagged}/{len(blocks)} blocks flagged, mean_prob={probs.mean():.3f}")
        ad.render_heatmap(path, blocks, probs, OUT / f"ai_heatmap_{name}.png")

    # The cleanest evidence: signed.png and tampered.png are pixel-identical
    # except inside the forged rectangle, so any *change* in the model's
    # per-block score has to come from that region.
    diff = results["tampered"] - results["signed"]
    grid_dim = int(round(np.sqrt(len(diff))))
    if grid_dim * grid_dim == len(diff):
        moved = np.argwhere(np.abs(diff.reshape(grid_dim, grid_dim)) > 0.05)
        print(f"\nblocks whose score changed once the forgery was added: {len(moved)}")
        print(f"(their (row, col) positions in a {grid_dim}x{grid_dim} grid):")
        print([tuple(int(v) for v in rc) for rc in moved])
    else:
        moved = np.argwhere(np.abs(diff) > 0.05)
        print(f"\nblocks whose score changed once the forgery was added: {len(moved)}")


if __name__ == "__main__":
    main()
