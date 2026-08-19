"""High-Throughput Live Neural Broadcast Server (Siren-Cast).

Ingests live webcam or video streams, performs online micro-hash SIREN fitting,
computes differential weight packets (Δθ), and broadcasts to WebSocket viewers in real time.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import struct
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import websockets
from websockets.server import WebSocketServerProtocol

from src.live.delta_compressor import DeltaCompressor
from src.model.hash_siren_video import HashSirenVideo
from src.model.siren_video import SirenVideo

# Stream Packet Types
PACKET_TYPE_HANDSHAKE = 0x01
PACKET_TYPE_CHUNK_DATA = 0x02
PACKET_TYPE_HEARTBEAT = 0x03

# Packet Flags
FLAG_KEYFRAME = 0x01
FLAG_DELTA = 0x00


class NeuralBroadcastServer:
    """Real-time live neural broadcaster that trains micro-weight deltas on-the-fly

    and streams continuous mathematical field updates over WebSockets.
    """

    def __init__(
        self,
        source: Union[int, str] = 0,
        host: str = "0.0.0.0",
        port: int = 8765,
        chunk_duration: float = 1.5,
        epochs_per_chunk: int = 100,
        batch_size: int = 65536,
        target_size: Optional[Tuple[int, int]] = (360, 640),  # (H, W)
        use_hash_grid: bool = True,
        keyframe_interval: int = 8,
        lr: float = 3e-4,
        loop_video: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        status_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self.source = source
        self.host = host
        self.port = port
        self.chunk_duration = float(chunk_duration)
        self.epochs_per_chunk = int(epochs_per_chunk)
        self.batch_size = int(batch_size)
        self.target_size = target_size
        self.use_hash_grid = use_hash_grid
        self.keyframe_interval = int(keyframe_interval)
        self.lr = float(lr)
        self.loop_video = loop_video
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.status_callback = status_callback

        self.delta_compressor = DeltaCompressor(preferred_codec="auto")

        # Runtime State
        self.is_running = False
        self._stop_event = threading.Event()
        self.connected_clients: Set[WebSocketServerProtocol] = set()
        self._clients_lock = threading.Lock()

        # Threading & Event Loops
        self._server_thread: Optional[threading.Thread] = None
        self._trainer_thread: Optional[threading.Thread] = None
        self._async_loop: Optional[asyncio.AbstractEventLoop] = None
        self._ws_server = None

        # Cached Latest Packets for Immediate Handshake on New Client Connect
        self._handshake_packet: Optional[bytes] = None
        self._latest_keyframe_packet: Optional[bytes] = None
        self._latest_chunk_packet: Optional[bytes] = None
        self._packet_lock = threading.Lock()

        # Stream Metadata
        self.stream_width = target_size[1] if target_size else 1280
        self.stream_height = target_size[0] if target_size else 720
        self.stream_fps = 30.0

        # Telemetry
        self.chunk_counter = 0
        self.total_bytes_sent = 0
        self.current_loss = 0.0
        self.current_psnr = 0.0
        self.current_bitrate_kbps = 0.0
        self.start_wall_time = 0.0

    def _build_model(self) -> nn.Module:
        """Instantiate appropriate model architecture based on hash-grid setting."""
        if self.use_hash_grid:
            model = HashSirenVideo(
                n_levels=12,
                n_features_per_level=2,
                log2_hashmap_size=16,
                hidden_features=64,
                hidden_layers=2,
                out_features=3,
            )
        else:
            model = SirenVideo(
                in_features=3,
                hidden_features=256,
                hidden_layers=5,
                out_features=3,
                omega_xy=30.0,
                omega_t=10.0,
            )
        return model.to(self.device)

    def _build_handshake_packet(self) -> bytes:
        """Construct handshake metadata binary packet."""
        magic = b"NEURALCAST"  # 10 bytes
        version = 2
        model_type = 1 if self.use_hash_grid else 0

        config = {
            "use_hash_grid": self.use_hash_grid,
            "width": self.stream_width,
            "height": self.stream_height,
            "fps": self.stream_fps,
            "chunk_duration": self.chunk_duration,
            "keyframe_interval": self.keyframe_interval,
        }
        config_json = json.dumps(config).encode("utf-8")

        # Format: [PacketType(1B)] [Magic(10B)] [Version(1B)] [Width(4B)] [Height(4B)] [FPS(4B float)] [Duration(4B float)] [ModelType(1B)] [ConfigLen(4B)] [ConfigJSON]
        header = struct.pack(
            "!B10sBIIffBI",
            PACKET_TYPE_HANDSHAKE,
            magic,
            version,
            self.stream_width,
            self.stream_height,
            float(self.stream_fps),
            float(self.chunk_duration),
            model_type,
            len(config_json),
        )
        return header + config_json

    def _build_chunk_packet(
        self,
        chunk_idx: int,
        timestamp: float,
        duration: float,
        is_keyframe: bool,
        compressed_payload: bytes,
    ) -> bytes:
        """Construct chunk data binary packet."""
        flags = FLAG_KEYFRAME if is_keyframe else FLAG_DELTA
        # Format: [PacketType(1B)] [Flags(1B)] [ChunkID(4B)] [Timestamp(8B double)] [Duration(4B float)] [PayloadLen(4B)] [Payload]
        header = struct.pack(
            "!BBIdfI",
            PACKET_TYPE_CHUNK_DATA,
            flags,
            chunk_idx,
            float(timestamp),
            float(duration),
            len(compressed_payload),
        )
        return header + compressed_payload

    async def _handle_client(self, websocket: WebSocketServerProtocol) -> None:
        """Manage individual viewer WebSocket connection."""
        client_addr = websocket.remote_address
        with self._clients_lock:
            self.connected_clients.add(websocket)

        try:
            # 1. Send Handshake metadata packet
            with self._packet_lock:
                handshake_data = self._handshake_packet
                initial_keyframe = self._latest_keyframe_packet or self._latest_chunk_packet

            if handshake_data:
                await websocket.send(handshake_data)

            # 2. Immediately send latest keyframe/chunk so viewer starts rendering instantly
            if initial_keyframe:
                await websocket.send(initial_keyframe)

            # 3. Keep connection alive and process incoming messages/pings
            async for message in websocket:
                if isinstance(message, bytes) and len(message) > 0 and message[0] == PACKET_TYPE_HEARTBEAT:
                    await websocket.send(struct.pack("!B", PACKET_TYPE_HEARTBEAT))

        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            with self._clients_lock:
                self.connected_clients.discard(websocket)

    def _broadcast_packet_sync(self, packet_bytes: bytes) -> None:
        """Broadcast packet to all connected clients from worker thread."""
        if not self._async_loop or not self.is_running:
            return

        with self._clients_lock:
            active_clients = list(self.connected_clients)

        if not active_clients:
            return

        async def _do_broadcast() -> None:
            tasks = [client.send(packet_bytes) for client in active_clients]
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for client, res in zip(active_clients, results):
                    if isinstance(res, Exception):
                        with self._clients_lock:
                            self.connected_clients.discard(client)

        asyncio.run_coroutine_threadsafe(_do_broadcast(), self._async_loop)

    def _training_and_broadcast_loop(self) -> None:
        """Continuous camera ingest, online neural fitting, and delta broadcast thread."""
        # Open Video Capture
        if isinstance(self.source, int) or (isinstance(self.source, str) and self.source.isdigit()):
            cap_idx = int(self.source)
            cap = cv2.VideoCapture(cap_idx, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
        else:
            cap = cv2.VideoCapture(str(self.source))

        if not cap.isOpened():
            print(f"[BROADCAST] Error: Failed to open video source: {self.source}", flush=True)
            self.is_running = False
            return

        # Detect source resolution and fps
        native_fps = cap.get(cv2.CAP_PROP_FPS)
        if native_fps <= 0 or math.isnan(native_fps):
            native_fps = 30.0
        self.stream_fps = native_fps

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.target_size:
            self.stream_height, self.stream_width = self.target_size
        else:
            self.stream_width = w if w > 0 else 1280
            self.stream_height = h if h > 0 else 720

        # Build Handshake packet
        with self._packet_lock:
            self._handshake_packet = self._build_handshake_packet()

        # Initialize Base Model Shell
        model = self._build_model()
        prev_state_dict: Optional[Dict[str, torch.Tensor]] = None

        frames_per_chunk = max(1, int(round(self.chunk_duration * self.stream_fps)))
        optimizer = optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-6)
        criterion = nn.MSELoss()

        self.start_wall_time = time.perf_counter()
        stream_timestamp = 0.0

        print(
            f"[BROADCAST] Live Neural Broadcaster Started: {self.stream_width}x{self.stream_height} @ {self.stream_fps:.1f} FPS "
            f"({frames_per_chunk} frames/chunk @ {self.chunk_duration:.1f}s)",
            flush=True,
        )

        while not self._stop_event.is_set():
            chunk_start_wall = time.perf_counter()
            captured_frames: List[np.ndarray] = []

            # 1. Ingest temporal window frames
            for _ in range(frames_per_chunk):
                ret, frame = cap.read()
                if not ret:
                    if self.loop_video and not isinstance(self.source, int):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = cap.read()
                    if not ret:
                        break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.target_size:
                    frame_rgb = cv2.resize(frame_rgb, (self.stream_width, self.stream_height), interpolation=cv2.INTER_AREA)
                captured_frames.append(frame_rgb)

            if len(captured_frames) == 0:
                time.sleep(0.05)
                continue

            T_frames = len(captured_frames)
            frames_np = np.stack(captured_frames, axis=0).astype(np.float32) / 255.0
            target_tensor = torch.from_numpy(frames_np).to(self.device)  # (T, H, W, 3)

            # 2. Fast Online Neural Fitting
            model.train()
            t_coords = torch.linspace(-1.0, 1.0, steps=T_frames, device=self.device)
            y_coords = torch.linspace(-1.0, 1.0, steps=self.stream_height, device=self.device)
            x_coords = torch.linspace(-1.0, 1.0, steps=self.stream_width, device=self.device)

            train_t0 = time.perf_counter()
            final_loss = 0.0

            for _ in range(self.epochs_per_chunk):
                optimizer.zero_grad(set_to_none=True)

                t_idx = torch.randint(0, T_frames, (self.batch_size,), device=self.device)
                y_idx = torch.randint(0, self.stream_height, (self.batch_size,), device=self.device)
                x_idx = torch.randint(0, self.stream_width, (self.batch_size,), device=self.device)

                batch_coords = torch.stack([x_coords[x_idx], y_coords[y_idx], t_coords[t_idx]], dim=-1)
                batch_target = target_tensor[t_idx, y_idx, x_idx]

                pred = model(batch_coords)
                loss = criterion(pred, batch_target)
                loss.backward()
                optimizer.step()
                final_loss = float(loss.item())

            train_elapsed = (time.perf_counter() - train_t0) * 1000.0
            self.current_loss = final_loss
            self.current_psnr = -10.0 * math.log10(max(1e-7, final_loss))

            # 3. Differential / Keyframe Weight Compression
            is_keyframe = (self.chunk_counter % self.keyframe_interval == 0) or (prev_state_dict is None)
            curr_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

            payload_bytes, comp_stats = self.delta_compressor.compress_state_dict(
                current_state_dict=curr_state,
                prev_state_dict=prev_state_dict,
                is_keyframe=is_keyframe,
            )
            prev_state_dict = curr_state

            # 4. Construct Binary Chunk Packet & Broadcast
            chunk_packet = self._build_chunk_packet(
                chunk_idx=self.chunk_counter,
                timestamp=stream_timestamp,
                duration=self.chunk_duration,
                is_keyframe=is_keyframe,
                compressed_payload=payload_bytes,
            )

            with self._packet_lock:
                self._latest_chunk_packet = chunk_packet
                if is_keyframe:
                    self._latest_keyframe_packet = chunk_packet

            self._broadcast_packet_sync(chunk_packet)

            self.total_bytes_sent += len(chunk_packet)
            self.current_bitrate_kbps = (len(chunk_packet) * 8.0 / 1000.0) / max(0.01, self.chunk_duration)

            # Telemetry Callback
            with self._clients_lock:
                num_viewers = len(self.connected_clients)

            telemetry = {
                "chunk_idx": self.chunk_counter,
                "timestamp": stream_timestamp,
                "is_keyframe": is_keyframe,
                "loss": self.current_loss,
                "psnr_db": self.current_psnr,
                "payload_kb": comp_stats["payload_kb"],
                "bitrate_kbps": self.current_bitrate_kbps,
                "train_time_ms": train_elapsed,
                "compress_time_ms": comp_stats["compress_time_ms"],
                "viewers": num_viewers,
                "codec": comp_stats["codec"],
            }

            if self.status_callback:
                try:
                    self.status_callback(telemetry)
                except Exception:
                    pass

            self.chunk_counter += 1
            stream_timestamp += self.chunk_duration

            # Pacing to maintain real-time live clock
            chunk_total_elapsed = time.perf_counter() - chunk_start_wall
            sleep_needed = self.chunk_duration - chunk_total_elapsed
            if sleep_needed > 0:
                time.sleep(sleep_needed)

        cap.release()

    def start(self) -> None:
        """Start live neural broadcast server and training worker threads."""
        if self.is_running:
            return

        self.is_running = True
        self._stop_event.clear()

        # Start Asyncio WebSocket Server in dedicated background thread
        def _run_async_server() -> None:
            self._async_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._async_loop)

            async def _main() -> None:
                self._ws_server = await websockets.serve(
                    self._handle_client,
                    self.host,
                    self.port,
                    max_size=32 * 1024 * 1024,  # 32 MB max message
                )
                print(f"[BROADCAST] WebSocket Neural Stream listening at ws://{self.host}:{self.port}", flush=True)
                await asyncio.Future()  # run forever

            try:
                self._async_loop.run_until_complete(_main())
            except Exception:
                pass

        self._server_thread = threading.Thread(target=_run_async_server, daemon=True, name="NeuralBroadcastServer")
        self._server_thread.start()

        # Start Camera Ingest & Training Worker Thread
        self._trainer_thread = threading.Thread(
            target=self._training_and_broadcast_loop, daemon=True, name="NeuralOnlineTrainer"
        )
        self._trainer_thread.start()

    def stop(self) -> None:
        """Stop broadcast server, training worker, and disconnect clients."""
        if not self.is_running:
            return

        self.is_running = False
        self._stop_event.set()

        if self._async_loop and self._async_loop.is_running():
            async def _cleanup():
                if self._ws_server:
                    self._ws_server.close()
                    await self._ws_server.wait_closed()
                with self._clients_lock:
                    for client in list(self.connected_clients):
                        try:
                            await client.close()
                        except Exception:
                            pass
            try:
                future = asyncio.run_coroutine_threadsafe(_cleanup(), self._async_loop)
                future.result(timeout=1.0)
            except Exception:
                pass
            self._async_loop.call_soon_threadsafe(self._async_loop.stop)

        if self._trainer_thread and self._trainer_thread.is_alive():
            self._trainer_thread.join(timeout=2.0)

        print("[BROADCAST] Live Neural Broadcaster stopped.", flush=True)

    def get_status(self) -> Dict[str, Any]:
        """Return current real-time telemetry dictionary."""
        with self._clients_lock:
            num_viewers = len(self.connected_clients)

        return {
            "is_running": self.is_running,
            "host": self.host,
            "port": self.port,
            "viewers": num_viewers,
            "chunk_idx": self.chunk_counter,
            "loss": self.current_loss,
            "psnr_db": self.current_psnr,
            "bitrate_kbps": self.current_bitrate_kbps,
            "total_bytes_mb": self.total_bytes_sent / (1024 * 1024),
        }
