"""Master Cinema Codec: Compresses Video + Multi-Channel Audio + HDR into .neura 2.0."""

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

from src.audio.audio_extractor import AudioExtractor
from src.chunking.chunk_orchestrator import ChunkOrchestrator
from src.chunking.video_splitter import VideoSplitter
from src.container.neura_v2_writer import NeuraV2Writer


def compress_cinema(
    video_path: str,
    output_path: str,
    chunk_duration: float = 3.0,
    epochs_per_chunk: int = 400,
    batch_size: int = 65536,
    lr: float = 2e-4,
    hidden_features: int = 384,
    hidden_layers: int = 6,
    omega_xy: float = 30.0,
    omega_t: float = 10.0,
    max_chunks: int = None,
    audio_codec: str = "aac",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    """Execute complete cinema compression with multiplexed audio and HDR color metadata."""
    print(f"\n=======================================================", flush=True)
    print(f"🎬 SIREN-ZIP 2.0: Cinema Master Compressor (A/V + HDR)", flush=True)
    print(f"   Input Cinema File : {video_path}", flush=True)
    print(f"   Output Container  : {output_path}", flush=True)
    print(f"   Chunk Duration    : {chunk_duration:.1f}s / Neural GOP", flush=True)
    print(f"   Epochs Per Chunk  : {epochs_per_chunk}", flush=True)
    print(f"=======================================================\n", flush=True)

    # 1. Extract Audio Track
    print(f"[AUDIO] Extracting cinema audio track with '{audio_codec}'...", flush=True)
    audio_res = AudioExtractor.extract_audio(video_path, target_codec=audio_codec)
    print(
        f"[AUDIO] Extracted {len(audio_res.audio_bytes)/1024:.1f} KB audio payload "
        f"({audio_res.channels}ch @ {audio_res.sample_rate}Hz, duration: {audio_res.duration:.2f}s)",
        flush=True,
    )

    # 2. Probe HDR Color Metadata
    print(f"[COLOR] Probing 10-bit HDR10+ / Rec.2020 / ST.2084 PQ metadata...", flush=True)
    color_meta = AudioExtractor.probe_color_metadata(video_path)
    prim_str = "Rec.2020 (Wide Gamut)" if color_meta["color_primaries"] == 9 else "Rec.709 (SDR)"
    trc_str = "ST.2084 PQ (HDR)" if color_meta["transfer_characteristics"] == 16 else "BT.709 (SDR)"
    print(f"[COLOR] Color Primaries: {prim_str} | Transfer: {trc_str} | Depth: {color_meta['bits_per_raw_sample']}-bit", flush=True)

    # 3. Plan Video Slices
    splitter = VideoSplitter(video_path=video_path, chunk_duration=chunk_duration)
    if max_chunks and max_chunks < len(splitter.chunks):
        splitter.chunks = splitter.chunks[:max_chunks]

    video_info = splitter.get_video_info()
    total_chunks = len(splitter.chunks)

    model_config = {
        "hidden_features": hidden_features,
        "hidden_layers": hidden_layers,
        "omega_xy": omega_xy,
        "omega_t": omega_t,
        "omega_0_hidden": 30.0,
        "final_activation": "clamp",
    }

    # 4. Initialize Multi-Track Container Writer
    writer = NeuraV2Writer(
        output_path=output_path,
        video_meta=video_info,
        model_config=model_config,
        total_chunks=total_chunks,
        chunk_duration=chunk_duration,
        audio_bytes=audio_res.audio_bytes,
        audio_codec_type=audio_res.codec_id,
        audio_sample_rate=audio_res.sample_rate,
        audio_channels=audio_res.channels,
        color_primaries=color_meta["color_primaries"],
        transfer_characteristics=color_meta["transfer_characteristics"],
    )

    orchestrator = ChunkOrchestrator(
        video_splitter=splitter,
        output_neura_path=output_path,
        epochs_per_chunk=epochs_per_chunk,
        batch_size=batch_size,
        lr=lr,
        hidden_features=hidden_features,
        hidden_layers=hidden_layers,
        omega_xy=omega_xy,
        omega_t=omega_t,
        omega_0_hidden=30.0,
        device=device,
    )

    summary = orchestrator.run_pipeline()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="SIREN-ZIP Cinema Compressor.")
    parser.add_argument("--input", type=str, required=True, help="Input movie file")
    parser.add_argument("--output", type=str, default="cinema_full.neura", help="Output .neura 2.0 file")
    parser.add_argument("--chunk_duration", type=float, default=3.0, help="Chunk window in seconds")
    parser.add_argument("--epochs_per_chunk", type=int, default=400, help="Epochs per GOP chunk")
    parser.add_argument("--batch_size", type=int, default=65536, help="Batch size")
    parser.add_argument("--max_chunks", type=int, default=None, help="Max chunks (optional)")
    parser.add_argument("--audio_codec", type=str, default="aac", help="Audio codec (aac/opus/mp3)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")

    args = parser.parse_args()
    compress_cinema(
        video_path=args.input,
        output_path=args.output,
        chunk_duration=args.chunk_duration,
        epochs_per_chunk=args.epochs_per_chunk,
        batch_size=args.batch_size,
        max_chunks=args.max_chunks,
        audio_codec=args.audio_codec,
        device=args.device,
    )


if __name__ == "__main__":
    main()
