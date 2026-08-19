"""Master CLI: Compress long videos into unified .neura 2.0 cinema containers."""

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

from src.chunking.chunk_orchestrator import ChunkOrchestrator
from src.chunking.video_splitter import VideoSplitter


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SIREN-ZIP 2.0: Multi-Chunk Neural GOP Cinema Compressor.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=str, required=True, help="Path to input video (.mp4, .mkv, .avi, etc.)")
    parser.add_argument("--output", type=str, default="movie.neura", help="Path to output .neura 2.0 container")
    parser.add_argument("--chunk_duration", type=float, default=3.0, help="Temporal chunk window size (seconds)")
    parser.add_argument("--epochs_per_chunk", type=int, default=800, help="Training epochs per micro-SIREN chunk")
    parser.add_argument("--batch_size", type=int, default=65536, help="GPU coordinate batch size per gradient step")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--hidden_features", type=int, default=384, help="Hidden units per SineLayer")
    parser.add_argument("--hidden_layers", type=int, default=6, help="Number of hidden SineLayers")
    parser.add_argument("--omega_xy", type=float, default=30.0, help="Spatial angular frequency")
    parser.add_argument("--omega_t", type=float, default=10.0, help="Temporal angular frequency")
    parser.add_argument("--max_chunks", type=int, default=None, help="Optional cap on total chunks (for fast test)")
    parser.add_argument("--resize_height", type=int, default=None, help="Optional resize height (e.g. 720 or 540)")
    parser.add_argument("--resize_width", type=int, default=None, help="Optional resize width (e.g. 1280 or 960)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")

    args = parser.parse_args()

    target_size = None
    if args.resize_height and args.resize_width:
        target_size = (args.resize_height, args.resize_width)

    splitter = VideoSplitter(
        video_path=args.input,
        chunk_duration=args.chunk_duration,
        target_size=target_size,
    )

    if args.max_chunks and args.max_chunks < len(splitter.chunks):
        splitter.chunks = splitter.chunks[: args.max_chunks]

    orchestrator = ChunkOrchestrator(
        video_splitter=splitter,
        output_neura_path=args.output,
        epochs_per_chunk=args.epochs_per_chunk,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_features=args.hidden_features,
        hidden_layers=args.hidden_layers,
        omega_xy=args.omega_xy,
        omega_t=args.omega_t,
        omega_0_hidden=30.0,
        device=args.device,
    )

    summary = orchestrator.run_pipeline()


if __name__ == "__main__":
    main()
