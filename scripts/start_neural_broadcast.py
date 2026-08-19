"""Siren-Cast: Real-Time Live Neural Video Broadcaster CLI.

Ingests a live webcam feed or video file, fits micro-hash SIREN weight deltas (Δθ)
in real time on GPU, and broadcasts over WebSockets to connected viewers.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time

# Ensure UTF-8 output on Windows
if sys.stdout.encoding != "utf-8" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure root workspace is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.live.broadcast_server import NeuralBroadcastServer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="⚡ Siren-Cast: Real-Time Live Neural Video Broadcaster (WebSockets)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Broadcast source: Webcam index (e.g., '0') or video filepath (e.g., 'Movie_Trailer_1080p.mp4')",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host IP to bind WebSocket server")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind WebSocket server")
    parser.add_argument("--chunk_duration", type=float, default=1.5, help="Neural GOP chunk window in seconds")
    parser.add_argument("--epochs", type=int, default=80, help="Online training epochs per chunk")
    parser.add_argument("--batch_size", type=int, default=65536, help="GPU batch size for coordinate fitting")
    parser.add_argument("--fast_hash", action="store_true", default=True, help="Use Instant-NGP Multi-Resolution Hash Grid")
    parser.add_argument("--no_hash", action="store_false", dest="fast_hash", help="Use pure 2D/3D Sinusoidal SIREN")
    parser.add_argument("--width", type=int, default=640, help="Target stream width")
    parser.add_argument("--height", type=int, default=360, help="Target stream height")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")

    args = parser.parse_args()

    # Determine source type
    src_val = int(args.source) if args.source.isdigit() else args.source

    print("\n==========================================================================")
    print("📡 SIREN-CAST: REAL-TIME LIVE NEURAL VIDEO BROADCASTER")
    print("==========================================================================")
    print(f"   Broadcast Source   : {src_val}")
    print(f"   WebSocket URL      : ws://{args.host}:{args.port}")
    print(f"   Resolution         : {args.width}x{args.height}")
    print(f"   Chunk Duration     : {args.chunk_duration:.1f}s / GOP")
    print(f"   Epochs Per Chunk   : {args.epochs}")
    print(f"   Architecture       : {'Instant-NGP Hash-Grid SIREN' if args.fast_hash else 'Classic SIREN'}")
    print(f"   Device             : {args.device.upper()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("==========================================================================\n")

    def telemetry_printer(tel: dict) -> None:
        key_str = "[I-FRAME]" if tel["is_keyframe"] else "[DELTA Δθ]"
        print(
            f"⚡ GOP #{tel['chunk_idx']:03d} {key_str:10s} "
            f"| Time: {tel['timestamp']:5.1f}s "
            f"| Loss: {tel['loss']:.5f} "
            f"| PSNR: {tel['psnr_db']:4.1f} dB "
            f"| Payload: {tel['payload_kb']:5.1f} KB "
            f"| Bitrate: {tel['bitrate_kbps']:5.1f} kbps "
            f"| Fit: {tel['train_time_ms']:5.1f}ms "
            f"| Viewers: {tel['viewers']}",
            flush=True,
        )

    server = NeuralBroadcastServer(
        source=src_val,
        host=args.host,
        port=args.port,
        chunk_duration=args.chunk_duration,
        epochs_per_chunk=args.epochs,
        batch_size=args.batch_size,
        target_size=(args.height, args.width),
        use_hash_grid=args.fast_hash,
        device=args.device,
        status_callback=telemetry_printer,
    )

    def handle_exit(signum, frame):
        print("\n\n[BROADCAST] Stopping live stream server...", flush=True)
        server.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    server.start()
    print(f"🟢 Broadcast server active! Stream available at ws://localhost:{args.port}")
    print("Press Ctrl+C at any time to stop broadcasting.\n")

    try:
        while server.is_running:
            time.sleep(0.5)
    except KeyboardInterrupt:
        handle_exit(None, None)


if __name__ == "__main__":
    main()
