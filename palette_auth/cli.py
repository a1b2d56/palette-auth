"""Command-line interface: generate keys, sign a palette image, verify one.

Examples:
    python -m palette_auth.cli genkey --out signer
    python -m palette_auth.cli sign photo.png photo_signed.png --key signer.private.pem
    python -m palette_auth.cli verify photo_signed.png --key signer.public.pem --tamper-map map.png
"""
from __future__ import annotations

import argparse
import sys

from . import core, crypto


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="palette-auth", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser("genkey", help="generate an Ed25519 keypair")
    p_gen.add_argument("--out", default="key", help="writes <out>.private.pem and <out>.public.pem")

    p_sign = sub.add_parser("sign", help="sign a palette image")
    p_sign.add_argument("input", help="path to source image (PNG/GIF)")
    p_sign.add_argument("output", help="path to write authenticated image")
    p_sign.add_argument("--key", required=True, help="private key .pem")
    p_sign.add_argument("--block-size", type=int, default=core.MIN_BLOCK_SIZE, help="block partition size (pixels)")

    p_verify = sub.add_parser("verify", help="verify a signed palette image")
    p_verify.add_argument("input", help="path to image to verify")
    p_verify.add_argument("--key", required=True, help="public key .pem")
    p_verify.add_argument("--tamper-map", help="optional path to save a visual tamper-map overlay")
    p_verify.add_argument("--recover", help="optional path to save a version with tampered blocks restored")
    p_verify.add_argument("--recover-method", choices=["neural", "bilinear"], default="neural", help="recovery reconstruction engine (default: neural)")

    p_train = sub.add_parser("train-detector", help="train the AI tamper detector on synthetic data")
    p_train.add_argument("--out", default="tamper_cnn.pt", help="output file for trained weights (.pt)")
    p_train.add_argument("--n-per-class", type=int, default=2000, help="synthetic patches per class")
    p_train.add_argument("--epochs", type=int, default=12, help="training epochs")
    p_train.add_argument("--lr", type=float, default=1e-3, help="learning rate")

    p_detect = sub.add_parser("detect", help="AI tamper detector -- no signature or key needed")
    p_detect.add_argument("input", help="path to image to analyze")
    p_detect.add_argument("--model", default=None, help="optional trained detector weights (.pt)")
    p_detect.add_argument("--heatmap", help="optional path to save a visual heatmap overlay")
    p_detect.add_argument("--threshold", type=float, default=0.45, help="tamper probability threshold")
    p_detect.add_argument("--no-ela", action="store_true", help="disable Error Level Analysis fusion")

    p_demo = sub.add_parser("demo", help="run end-to-end demonstration and generate showcase dashboard")
    p_demo.add_argument("--open", action="store_true", help="open showcase dashboard in web browser")

    args = parser.parse_args(argv)

    if args.command == "demo":
        try:
            import demo
        except ImportError:
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
            import demo
        demo.main(open_browser=args.open)
        return 0

    if args.command == "genkey":
        priv, pub = crypto.generate_keypair()
        crypto.save_private_key(priv, f"{args.out}.private.pem")
        crypto.save_public_key(pub, f"{args.out}.public.pem")
        print(f"wrote {args.out}.private.pem and {args.out}.public.pem")
        return 0

    if args.command == "sign":
        try:
            info = core.sign_image(args.input, args.output, args.key, block_size=args.block_size)
            print(f"signed -> {info['output']} ({info['n_blocks']} blocks, block_size={info['block_size']})")
            return 0
        except Exception as exc:
            print(f"Error signing image: {exc}", file=sys.stderr)
            return 1

    if args.command == "verify":
        try:
            result = core.verify_image(args.input, args.key)
            if result.authentic:
                print("AUTHENTIC: no tampering detected")
            else:
                print(f"NOT AUTHENTIC ({result.reason})")
                if result.tampered_blocks:
                    print(f"{len(result.tampered_blocks)} of {len(result.all_blocks)} blocks flagged as tampered")
            if args.tamper_map:
                core.render_tamper_map(args.input, result, args.tamper_map)
                print(f"tamper map -> {args.tamper_map}")
            if args.recover:
                core.render_recovery(args.input, result, args.recover, method=args.recover_method)
                print(f"recovered approximation ({args.recover_method}) -> {args.recover}")
            return 0 if result.authentic else 1
        except Exception as exc:
            print(f"Error verifying image: {exc}", file=sys.stderr)
            return 1

    if args.command == "train-detector":
        from . import ai_detector

        print(f"Training forensic detector ({args.n_per_class} per class, {args.epochs} epochs)...")
        model, hist = ai_detector.train(n_per_class=args.n_per_class, epochs=args.epochs, lr=args.lr)
        ai_detector.save_model(model, args.out)
        print(f"detector weights saved -> {args.out}")
        return 0

    if args.command == "detect":
        from . import ai_detector

        model = ai_detector.load_model(args.model) if args.model else ai_detector.get_default_forensic_engine()
        blocks, probs = ai_detector.predict_heatmap(args.input, model, fuse_ela=not args.no_ela)
        flagged = int((probs > args.threshold).sum())
        print(f"{flagged} of {len(blocks)} blocks flagged as manipulated (threshold={args.threshold})")
        if args.heatmap:
            ai_detector.render_heatmap(args.input, blocks, probs, args.heatmap, threshold=args.threshold)
            print(f"heatmap -> {args.heatmap}")
        return 0

