"""End-to-End Integration Verification for Siren-Cast Live Neural Streaming."""

import os
import sys
import time
import numpy as np
import torch

if sys.stdout.encoding != "utf-8" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.live.delta_compressor import DeltaCompressor
from src.live.broadcast_server import NeuralBroadcastServer
from src.live.stream_client import NeuralStreamClient
from src.model.hash_siren_video import HashSirenVideo


def test_delta_compressor() -> None:
    print("\n--- 1. Testing DeltaCompressor ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m1 = HashSirenVideo().to(device)
    m2 = HashSirenVideo().to(device)

    compressor = DeltaCompressor(preferred_codec="auto")

    # Keyframe
    payload_key, stats_key = compressor.compress_state_dict(m1.state_dict(), is_keyframe=True)
    print(f"Keyframe payload: {stats_key['payload_kb']:.2f} KB | Codec: {stats_key['codec']} | Compression Time: {stats_key['compress_time_ms']:.2f}ms")
    assert len(payload_key) > 0

    state_key, dec_ms = compressor.decompress_state_dict(payload_key, is_keyframe=True, device=device)
    print(f"Keyframe decompress: {dec_ms:.2f}ms")
    assert len(state_key) == len(m1.state_dict())

    # Delta
    payload_delta, stats_delta = compressor.compress_state_dict(
        current_state_dict=m2.state_dict(),
        prev_state_dict=m1.state_dict(),
        is_keyframe=False,
    )
    print(f"Delta payload: {stats_delta['payload_kb']:.2f} KB | Ratio: {stats_delta['compression_ratio']:.1f}x | Compress Time: {stats_delta['compress_time_ms']:.2f}ms")
    assert len(payload_delta) > 0

    state_delta, dec_delta_ms = compressor.decompress_state_dict(
        payload_bytes=payload_delta,
        prev_state_dict=m1.state_dict(),
        is_keyframe=False,
        device=device,
    )
    print(f"Delta decompress: {dec_delta_ms:.2f}ms")
    assert len(state_delta) == len(m2.state_dict())
    assert dec_delta_ms < 15.0, f"Decompression took too long: {dec_delta_ms:.2f}ms"
    print("[PASS] DeltaCompressor PASSED!")


def test_live_streaming_pipeline() -> None:
    print("\n--- 2. Testing Live Broadcast Server & Stream Client ---")
    video_path = "Movie_Trailer_1080p.mp4"
    if not os.path.exists(video_path):
        print(f"Skipping video file test: {video_path} not found")
        return

    port = 8799
    server = NeuralBroadcastServer(
        source=video_path,
        host="127.0.0.1",
        port=port,
        chunk_duration=1.0,
        epochs_per_chunk=30,
        target_size=(180, 320),
        use_hash_grid=True,
    )

    client = NeuralStreamClient(
        url=f"ws://127.0.0.1:{port}",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    try:
        print("[TEST] Starting Broadcast Server on port 8799...")
        server.start()
        time.sleep(1.5)

        print("[TEST] Connecting Stream Client...")
        client.start()

        # Wait for client to receive first chunk
        t0 = time.perf_counter()
        while not client.is_connected or client.active_state_dict is None:
            if time.perf_counter() - t0 > 15.0:
                raise TimeoutError("Client timed out waiting for stream")
            time.sleep(0.2)

        print(f"[TEST] Client connected! Rendering frames at 60 FPS...")

        for i in range(30):
            frame, tel = client.render_frame(render_width=320, render_height=180)
            if frame is not None:
                assert frame.shape == (180, 320, 3)
            time.sleep(0.016)

        print(f"Rendered 30 live frames! FPS: {client.fps_counter:.1f} | Bitrate: {client.current_bitrate_kbps:.1f} kbps | Paging: {client.last_paging_time_ms:.2f}ms")
        assert client.total_packets_received > 0
        print("[PASS] Live Streaming Pipeline PASSED!")

    finally:
        print("[TEST] Shutting down client & server...")
        client.stop()
        server.stop()
        time.sleep(0.5)


if __name__ == "__main__":
    test_delta_compressor()
    test_live_streaming_pipeline()
    print("\n[SUCCESS] ALL SIREN-CAST OPTION C TESTS PASSED SUCCESSFULLY!")
