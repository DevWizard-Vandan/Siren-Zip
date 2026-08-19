"""Low-Latency Live Neural Streaming Client (Siren-Cast).

Connects to a live WebSocket broadcast, unpacks differential weight deltas (Δθ) in < 1ms,
and evaluates continuous spatio-temporal coordinates on GPU in real time at 60 FPS.
"""

from __future__ import annotations

import asyncio
import json
import struct
import sys
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import websockets

from src.color.hdr_transfer import pq_to_linear, rec2020_to_rec709
from src.color.tone_mapper import apply_tone_mapping
from src.live.broadcast_server import (
    FLAG_KEYFRAME,
    PACKET_TYPE_CHUNK_DATA,
    PACKET_TYPE_HANDSHAKE,
    PACKET_TYPE_HEARTBEAT,
)
from src.live.delta_compressor import DeltaCompressor
from src.model.hash_siren_video import HashSirenVideo
from src.model.siren_video import SirenVideo
from src.types.viewport import ViewportBounds


class NeuralStreamClient:
    """Async WebSocket client receiving live neural weight updates and rendering continuous video."""

    def __init__(
        self,
        url: str = "ws://localhost:8765",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        auto_reconnect: bool = True,
        on_connected_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_frame_cb: Optional[Callable[[np.ndarray, Dict[str, Any]], None]] = None,
        on_error_cb: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.url = url
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.auto_reconnect = auto_reconnect
        self.on_connected_cb = on_connected_cb
        self.on_frame_cb = on_frame_cb
        self.on_error_cb = on_error_cb

        self.delta_compressor = DeltaCompressor(preferred_codec="auto")

        # Stream State
        self.is_connected = False
        self.is_running = False
        self._stop_event = threading.Event()
        self._async_loop: Optional[asyncio.AbstractEventLoop] = None
        self._client_thread: Optional[threading.Thread] = None

        # Model & Weights State
        self.model: Optional[nn.Module] = None
        self.active_state_dict: Optional[Dict[str, torch.Tensor]] = None
        self.prev_state_dict: Optional[Dict[str, torch.Tensor]] = None
        self.model_lock = threading.Lock()

        # Metadata
        self.stream_width = 1280
        self.stream_height = 720
        self.stream_fps = 30.0
        self.chunk_duration = 1.5
        self.use_hash_grid = True
        self.model_config: Dict[str, Any] = {}

        # Timing & Synchronization
        self.current_chunk_idx: int = -1
        self.chunk_start_wall_time: float = 0.0
        self.chunk_timestamp: float = 0.0
        self.last_packet_wall_time: float = 0.0

        # Telemetry
        self.total_packets_received: int = 0
        self.total_bytes_received: int = 0
        self.last_paging_time_ms: float = 0.0
        self.last_eval_time_ms: float = 0.0
        self.current_bitrate_kbps: float = 0.0
        self.extrapolation_count: int = 0
        self.fps_counter: float = 0.0
        self._frame_times: list[float] = []

    def _init_model(self, model_type: int, config: Dict[str, Any]) -> None:
        """Initialize GPU model shell based on broadcast server handshake."""
        with self.model_lock:
            self.use_hash_grid = (model_type == 1)
            self.model_config = config

            if self.use_hash_grid:
                self.model = HashSirenVideo(
                    n_levels=12,
                    n_features_per_level=2,
                    log2_hashmap_size=16,
                    hidden_features=64,
                    hidden_layers=2,
                    out_features=3,
                ).to(self.device)
            else:
                self.model = SirenVideo(
                    in_features=3,
                    hidden_features=256,
                    hidden_layers=5,
                    out_features=3,
                    omega_xy=30.0,
                    omega_t=10.0,
                ).to(self.device)

            self.model.eval()
            self.active_state_dict = None
            self.prev_state_dict = None

    def _handle_handshake_packet(self, data: bytes) -> None:
        """Process 0x01 Handshake metadata packet."""
        # Format: [PacketType(1B)] [Magic(10B)] [Version(1B)] [Width(4B)] [Height(4B)] [FPS(4B float)] [Duration(4B float)] [ModelType(1B)] [ConfigLen(4B)] [ConfigJSON]
        if len(data) < 33:
            return

        packet_type, magic, version, width, height, fps, duration, model_type, config_len = struct.unpack(
            "!B10sBIIffBI", data[:33]
        )

        if magic != b"NEURALCAST":
            print(f"[CLIENT] Warning: Unknown stream magic header: {magic}", flush=True)
            return

        config_json = data[33 : 33 + config_len].decode("utf-8")
        config = json.loads(config_json) if config_json else {}

        self.stream_width = width
        self.stream_height = height
        self.stream_fps = fps
        self.chunk_duration = duration

        self._init_model(model_type, config)
        self.is_connected = True

        print(
            f"[CLIENT] Connected to Siren-Cast stream: {width}x{height} @ {fps:.1f} FPS "
            f"({'Hash-Siren' if model_type == 1 else 'Siren-Classic'})",
            flush=True,
        )

        if self.on_connected_cb:
            try:
                self.on_connected_cb({
                    "width": width,
                    "height": height,
                    "fps": fps,
                    "duration": duration,
                    "model_type": "HashSiren" if model_type == 1 else "Siren",
                })
            except Exception:
                pass

    def _handle_chunk_packet(self, data: bytes) -> None:
        """Process 0x02 Chunk data packet and update GPU weights in < 1ms."""
        # Format: [PacketType(1B)] [Flags(1B)] [ChunkID(4B)] [Timestamp(8B double)] [Duration(4B float)] [PayloadLen(4B)] [Payload]
        if len(data) < 22:
            return

        packet_type, flags, chunk_id, timestamp, duration, payload_len = struct.unpack(
            "!BBIdfI", data[:22]
        )
        payload = data[22 : 22 + payload_len]

        is_keyframe = bool(flags & FLAG_KEYFRAME)
        t_page_start = time.perf_counter()

        with self.model_lock:
            if self.model is None:
                return

            # Decompress differential weight delta
            new_state, decompress_ms = self.delta_compressor.decompress_state_dict(
                payload_bytes=payload,
                prev_state_dict=self.active_state_dict,
                is_keyframe=is_keyframe,
                device=self.device,
            )

            # Load into active GPU model
            self.model.load_state_dict(new_state, strict=False)

            self.prev_state_dict = self.active_state_dict
            self.active_state_dict = new_state

            self.current_chunk_idx = chunk_id
            self.chunk_timestamp = timestamp
            self.chunk_duration = duration
            self.chunk_start_wall_time = time.perf_counter()
            self.last_packet_wall_time = time.perf_counter()

            self.last_paging_time_ms = (time.perf_counter() - t_page_start) * 1000.0

        self.total_packets_received += 1
        self.total_bytes_received += len(data)
        self.current_bitrate_kbps = (len(data) * 8.0 / 1000.0) / max(0.01, duration)

    async def _async_receive_loop(self) -> None:
        """Asynchronous client connection & packet receiver."""
        while not self._stop_event.is_set():
            try:
                print(f"[CLIENT] Connecting to {self.url}...", flush=True)
                async with websockets.connect(self.url, max_size=32 * 1024 * 1024) as ws:
                    self.is_connected = True
                    async for message in ws:
                        if self._stop_event.is_set():
                            break

                        if isinstance(message, bytes) and len(message) > 0:
                            p_type = message[0]
                            if p_type == PACKET_TYPE_HANDSHAKE:
                                self._handle_handshake_packet(message)
                            elif p_type == PACKET_TYPE_CHUNK_DATA:
                                self._handle_chunk_packet(message)

            except (websockets.exceptions.ConnectionClosed, ConnectionRefusedError, OSError) as e:
                self.is_connected = False
                if self.on_error_cb and not self._stop_event.is_set():
                    try:
                        self.on_error_cb(f"Connection lost: {e}")
                    except Exception:
                        pass

                if not self.auto_reconnect or self._stop_event.is_set():
                    break
                await asyncio.sleep(1.0)
            except Exception as e:
                self.is_connected = False
                if not self.auto_reconnect or self._stop_event.is_set():
                    break
                await asyncio.sleep(1.0)

    @torch.no_grad()
    def render_frame(
        self,
        render_width: int = 640,
        render_height: int = 360,
        viewport: ViewportBounds = ViewportBounds(),
        tone_map_mode: str = "aces",
        exposure: float = 1.0,
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """Evaluate continuous mathematical field at current wall-clock live time on GPU."""
        t_eval_start = time.perf_counter()

        with self.model_lock:
            if self.model is None or self.active_state_dict is None:
                return None, {"is_ready": False, "status": "Waiting for live stream..."}

            # 1. Compute continuous normalized local time t_local in [-1.0, 1.0]
            elapsed_in_chunk = time.perf_counter() - self.chunk_start_wall_time
            t_ratio = elapsed_in_chunk / max(0.01, self.chunk_duration)
            t_local = -1.0 + 2.0 * t_ratio

            # Smooth Motion Extrapolation if network packet is late
            if t_local > 1.0:
                self.extrapolation_count += 1
                t_local = min(1.2, t_local)  # Smoothly extrapolate temporal trajectory

            # 2. Build 3D continuous coordinate grid within visible viewport
            eff_w = max(64, render_width)
            eff_h = max(36, render_height)

            y_coords = torch.linspace(viewport.y_min, viewport.y_max, steps=eff_h, device=self.device, dtype=torch.float32)
            x_coords = torch.linspace(viewport.x_min, viewport.x_max, steps=eff_w, device=self.device, dtype=torch.float32)
            y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing="ij")
            t_grid = torch.full_like(x_grid, fill_value=t_local)

            coords = torch.stack([x_grid, y_grid, t_grid], dim=-1).reshape(-1, 3)

            # 3. Model forward pass
            pred_rgb = self.model(coords)
            rgb_tensor = pred_rgb.reshape(eff_h, eff_w, 3)

            # Exposure adjustment
            if abs(exposure - 1.0) > 1e-4:
                rgb_tensor = rgb_tensor * exposure

            # Convert to uint8 numpy RGB array
            rgb_np = (rgb_tensor.clamp(0.0, 1.0) * 255.0).to(torch.uint8).cpu().numpy()

        eval_ms = (time.perf_counter() - t_eval_start) * 1000.0
        self.last_eval_time_ms = eval_ms

        # Calculate live FPS
        now = time.perf_counter()
        self._frame_times.append(now)
        self._frame_times = [t for t in self._frame_times if now - t <= 1.0]
        self.fps_counter = float(len(self._frame_times))

        telemetry = {
            "is_ready": True,
            "fps": self.fps_counter,
            "eval_time_ms": eval_ms,
            "paging_time_ms": self.last_paging_time_ms,
            "chunk_idx": self.current_chunk_idx,
            "t_local": t_local,
            "bitrate_kbps": self.current_bitrate_kbps,
            "packets_received": self.total_packets_received,
            "extrapolated": self.extrapolation_count,
            "is_connected": self.is_connected,
            "width": eff_w,
            "height": eff_h,
        }

        if self.on_frame_cb:
            try:
                self.on_frame_cb(rgb_np, telemetry)
            except Exception:
                pass

        return rgb_np, telemetry

    def start(self) -> None:
        """Start async client connection thread."""
        if self.is_running:
            return

        self.is_running = True
        self._stop_event.clear()

        def _run_thread() -> None:
            self._async_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._async_loop)
            try:
                self._async_loop.run_until_complete(self._async_receive_loop())
            except Exception:
                pass

        self._client_thread = threading.Thread(target=_run_thread, daemon=True, name="NeuralStreamClient")
        self._client_thread.start()

    def stop(self) -> None:
        """Disconnect and stop client threads."""
        if not self.is_running:
            return

        self.is_running = False
        self._stop_event.set()

        if self._async_loop and self._async_loop.is_running():
            self._async_loop.call_soon_threadsafe(self._async_loop.stop)

        if self._client_thread and self._client_thread.is_alive():
            self._client_thread.join(timeout=1.0)

        self.is_connected = False
        print("[CLIENT] Siren-Cast Stream Client stopped.", flush=True)
