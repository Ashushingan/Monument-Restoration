"""
train_all.py — Master training script
======================================
Run all three modules in order:
    python train_all.py --stage all        # run everything
    python train_all.py --stage classifier # only style classifier
    python train_all.py --stage segmenter  # only U-Net
    python train_all.py --stage gan        # only GAN

Memory-safe defaults for ~4 GB GPU:
    --batch_cls 64  --image_size 512  --grad_accum 1
    --batch_seg 16  --batch_gan 8
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

# Reduce CUDA memory fragmentation before importing torch
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from models.style_classifier     import StyleClassifierTrainer
from models.damage_segmentation  import DamageSegmentationTrainer
from models.partial_conv_gan     import PartialConvGANTrainer
from data.dataset                import STYLE_CLASSES


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",    default="data/raw",   help="Root dataset directory")
    p.add_argument("--stage",       default="all",
                   choices=["all", "classifier", "segmenter", "gan"])
    p.add_argument("--cls_epochs",  type=int, default=30)
    p.add_argument("--seg_epochs",  type=int, default=40)
    p.add_argument("--gan_epochs",  type=int, default=100)

    # ── A100 80 GB optimised defaults ──────────────────────────────────
    # Full 512×512 resolution, large batches, no gradient accumulation needed.
    p.add_argument("--batch_cls",   type=int, default=64,
                   help="Batch size for classifier (default 64 for A100)")
    p.add_argument("--image_size",  type=int, default=512,
                   help="Input resolution for classifier (512 for A100)")
    p.add_argument("--grad_accum",  type=int, default=1,
                   help="Gradient accumulation steps (1 = disabled, fine on A100)")
    p.add_argument("--batch_seg",   type=int, default=16,
                   help="Batch size for U-Net segmenter (default 16 for A100)")
    p.add_argument("--batch_gan",   type=int, default=8,
                   help="Batch size for GAN (default 8 for A100)")
    return p.parse_args()


def run_classifier(args):
    print("\n" + "▓"*55)
    print("  STAGE 1 — Architectural Style Classifier (Weeks 3-4)")
    print("▓"*55)
    trainer = StyleClassifierTrainer(
        data_dir=args.data_dir,
        num_classes=len(STYLE_CLASSES),
        batch_size=args.batch_cls,
        image_size=args.image_size,
        grad_accum_steps=args.grad_accum,
    )
    trainer.train(epochs=args.cls_epochs)
    return trainer


def run_segmenter(args):
    print("\n" + "▓"*55)
    print("  STAGE 2 — U-Net Damage Segmentation (Weeks 5-6)")
    print("▓"*55)
    trainer = DamageSegmentationTrainer(
        data_dir=args.data_dir,
        batch_size=args.batch_seg,
    )
    trainer.train(epochs=args.seg_epochs)
    return trainer


def run_gan(args):
    print("\n" + "▓"*55)
    print("  STAGE 3 — Partial Conv GAN Reconstruction (Weeks 7-9)")
    print("▓"*55)
    trainer = PartialConvGANTrainer(
        data_dir=args.data_dir,
        batch_size=args.batch_gan,
    )
    trainer.train(epochs=args.gan_epochs)
    return trainer


def main():
    args = parse_args()
    print(f"\nMonument Digital Restoration — Training Pipeline")
    print(f"Data dir     : {args.data_dir}")
    print(f"Stage        : {args.stage}")
    print(f"Classifier   : batch={args.batch_cls}, size={args.image_size}px, "
          f"grad_accum={args.grad_accum} (effective batch={args.batch_cls * args.grad_accum})")
    print(f"Segmenter    : batch={args.batch_seg}")
    print(f"GAN          : batch={args.batch_gan}\n")

    if args.stage in ("all", "classifier"):
        run_classifier(args)
    if args.stage in ("all", "segmenter"):
        run_segmenter(args)
    if args.stage in ("all", "gan"):
        run_gan(args)

    print("\n✅ Training complete! Checkpoints saved in ./checkpoints/")


if __name__ == "__main__":
    main()
