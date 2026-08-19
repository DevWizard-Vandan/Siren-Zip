"""CLI Entrypoint for training SIREN Static Image INR Compressor."""

from __future__ import annotations

import argparse
import os
import sys

# Ensure UTF-8 output on Windows
if sys.stdout.encoding != "utf-8" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure root workspace is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.data.coordinate_dataset import ImageCoordinateData
from src.model.siren import SirenImage
from src.training.trainer import ImageTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a pure SIREN Implicit Neural Representation (INR) on an image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="test_target.png",
        help="Path to input image (e.g. test_target.png)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2000,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=131072,
        help="Batch size of random (x, y) coordinates per gradient step",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Initial learning rate for AdamW optimizer",
    )
    parser.add_argument(
        "--hidden_features",
        type=int,
        default=256,
        help="Number of hidden units per SineLayer",
    )
    parser.add_argument(
        "--hidden_layers",
        type=int,
        default=5,
        help="Number of hidden SineLayers",
    )
    parser.add_argument(
        "--omega_0",
        type=float,
        default=30.0,
        help="First layer angular frequency multiplier (Sitzmann et al.)",
    )
    parser.add_argument(
        "--omega_0_hidden",
        type=float,
        default=30.0,
        help="Hidden layer angular frequency multiplier",
    )
    parser.add_argument(
        "--final_activation",
        type=str,
        default="sigmoid",
        choices=["sigmoid", "clamp", "none"],
        help="Activation function applied to the RGB output layer",
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=100,
        help="Evaluation and visual sampling interval (in epochs)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--sample_dir",
        type=str,
        default="runs/samples",
        help="Directory to save visual reconstruction comparisons",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device ('cuda' or 'cpu')",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=None,
        help="Steps per epoch. Defaults to full pass over all image pixels.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.image_path):
        print(f"[ERROR] Image '{args.image_path}' does not exist.")
        print(f"[TIP] Please run: python scripts/generate_test_target.py first!")
        sys.exit(1)

    # 1. Load image and prepare GPU coordinate container
    print(f"[INFO] Loading image '{args.image_path}' to {args.device}...")
    dataset = ImageCoordinateData(
        image_path=args.image_path,
        device=args.device,
    )

    # 2. Instantiate SIREN Model
    model = SirenImage(
        in_features=2,
        hidden_features=args.hidden_features,
        hidden_layers=args.hidden_layers,
        out_features=3,
        omega_0=args.omega_0,
        omega_0_hidden=args.omega_0_hidden,
        final_activation=args.final_activation,
    )

    # 3. Instantiate Trainer
    trainer = ImageTrainer(
        model=model,
        data=dataset,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        save_freq=args.save_freq,
        checkpoint_dir=args.checkpoint_dir,
        sample_dir=args.sample_dir,
        device=args.device,
    )

    # 4. Train
    results = trainer.train()
    print(f"[SUCCESS] Training completed! Best PSNR: {results['best_psnr']:.2f} dB")


if __name__ == "__main__":
    main()
