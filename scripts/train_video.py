"""CLI Entrypoint for training Spatio-Temporal Video SIREN (Video INR)."""

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
from src.data.video_coordinate_dataset import VideoCoordinateData
from src.model.siren_video import SirenVideo
from src.training.video_trainer import VideoTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a continuous Spatio-Temporal SIREN on a video clip.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default="Short_Clip_720p.mp4",
        help="Path to input video file",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=4000,
        help="Number of training epochs (gradient steps)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=262144,
        help="Batch size of random (x, y, t) points per gradient step",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Initial learning rate for AdamW optimizer",
    )
    parser.add_argument(
        "--hidden_features",
        type=int,
        default=384,
        help="Number of hidden units per SineLayer",
    )
    parser.add_argument(
        "--hidden_layers",
        type=int,
        default=6,
        help="Number of hidden SineLayers",
    )
    parser.add_argument(
        "--omega_xy",
        type=float,
        default=30.0,
        help="Spatial frequency multiplier for first layer",
    )
    parser.add_argument(
        "--omega_t",
        type=float,
        default=10.0,
        help="Temporal frequency multiplier for first layer",
    )
    parser.add_argument(
        "--omega_0_hidden",
        type=float,
        default=30.0,
        help="Frequency multiplier for hidden layers",
    )
    parser.add_argument(
        "--final_activation",
        type=str,
        default="clamp",
        choices=["clamp", "sigmoid", "none"],
        help="Output layer activation",
    )
    parser.add_argument(
        "--resize_height",
        type=int,
        default=None,
        help="Optional height to downscale video for faster training",
    )
    parser.add_argument(
        "--resize_width",
        type=int,
        default=None,
        help="Optional width to downscale video for faster training",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Optional limit on total frames to load",
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=200,
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
        default="runs/video_samples",
        help="Directory to save visual keyframe comparisons",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device ('cuda' or 'cpu')",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.video_path):
        print(f"[ERROR] Video '{args.video_path}' does not exist.")
        sys.exit(1)

    target_size = None
    if args.resize_height is not None and args.resize_width is not None:
        target_size = (args.resize_height, args.resize_width)

    # 1. Load Video into GPU coordinate container
    print(f"[INFO] Loading video '{args.video_path}' on {args.device}...")
    dataset = VideoCoordinateData(
        video_path=args.video_path,
        target_size=target_size,
        max_frames=args.max_frames,
        device=args.device,
    )

    # 2. Instantiate Spatio-Temporal SIREN
    model = SirenVideo(
        in_features=3,
        hidden_features=args.hidden_features,
        hidden_layers=args.hidden_layers,
        out_features=3,
        omega_xy=args.omega_xy,
        omega_t=args.omega_t,
        omega_0_hidden=args.omega_0_hidden,
        final_activation=args.final_activation,
    )

    # 3. Instantiate Video Trainer
    trainer = VideoTrainer(
        model=model,
        data=dataset,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        steps_per_epoch=1,
        save_freq=args.save_freq,
        checkpoint_dir=args.checkpoint_dir,
        sample_dir=args.sample_dir,
        device=args.device,
    )

    # 4. Train
    results = trainer.train()
    print(f"[SUCCESS] Video training finished! Best PSNR: {results['best_psnr']:.2f} dB")


if __name__ == "__main__":
    main()
