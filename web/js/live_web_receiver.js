/**
 * In-Browser WebSocket Receiver for Live Siren-Cast Neural Video Streams.
 * Connects to ws://<host>:<port>, unpacks differential weight deltas, and updates WebGPU buffers live.
 */

class LiveWebReceiver {
    constructor() {
        this.ws = null;
        this.url = "ws://localhost:8765";
        this.isConnected = false;
        this.streamMetadata = null;

        this.currentChunkIdx = -1;
        this.chunkStartWallTime = 0;
        this.chunkDuration = 1.5;

        this.activeWeights = null;
        this.prevWeights = null;

        this.onConnected = null;
        this.onChunkReceived = null;
        this.onError = null;

        this.totalPackets = 0;
        this.currentBitrateKbps = 0;
    }

    /**
     * Connect to live Siren-Cast WebSocket broadcast
     * @param {string} url
     */
    connect(url = "ws://localhost:8765") {
        this.disconnect();
        this.url = url;

        try {
            this.ws = new WebSocket(url);
            this.ws.binaryType = "arraybuffer";

            this.ws.onopen = () => {
                console.log(`[Siren-Cast Web] Connected to ${url}`);
                this.isConnected = true;
            };

            this.ws.onmessage = (event) => {
                if (typeof event.data === "string") return;
                this._handleBinaryPacket(event.data);
            };

            this.ws.onerror = (err) => {
                console.error("[Siren-Cast Web] Connection error:", err);
                if (this.onError) this.onError(err);
            };

            this.ws.onclose = () => {
                console.log("[Siren-Cast Web] Disconnected.");
                this.isConnected = false;
            };
        } catch (e) {
            console.error("[Siren-Cast Web] Failed to connect:", e);
            if (this.onError) this.onError(e);
        }
    }

    disconnect() {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
        this.isConnected = false;
    }

    _handleBinaryPacket(buffer) {
        if (buffer.byteLength < 1) return;
        const view = new DataView(buffer);
        const packetType = view.getUint8(0);

        if (packetType === 0x01) {
            // Handshake Packet
            this._handleHandshake(buffer, view);
        } else if (packetType === 0x02) {
            // Chunk Data Packet
            this._handleChunkData(buffer, view);
        }
    }

    _handleHandshake(buffer, view) {
        if (buffer.byteLength < 33) return;

        // [PacketType(1B)] [Magic(10B)] [Version(1B)] [Width(4B)] [Height(4B)] [FPS(4B)] [Duration(4B)] [ModelType(1B)] [ConfigLen(4B)]
        const width = view.getUint32(12, false);
        const height = view.getUint32(16, false);
        const fps = view.getFloat32(20, false);
        const chunkDuration = view.getFloat32(24, false);
        const modelType = view.getUint8(28);
        const configLen = view.getUint32(29, false);

        let config = {};
        if (configLen > 0 && buffer.byteLength >= 33 + configLen) {
            const jsonBytes = new Uint8Array(buffer, 33, configLen);
            const jsonStr = new TextDecoder().decode(jsonBytes);
            try { config = JSON.parse(jsonStr); } catch (e) {}
        }

        this.streamMetadata = {
            width,
            height,
            fps,
            chunkDuration,
            modelType,
            config,
        };
        this.chunkDuration = chunkDuration;

        console.log(`[Siren-Cast Web] Handshake received: ${width}x${height} @ ${fps.toFixed(1)} FPS`);
        if (this.onConnected) this.onConnected(this.streamMetadata);
    }

    _handleChunkData(buffer, view) {
        if (buffer.byteLength < 22) return;

        // Format: [PacketType(1B)] [Flags(1B)] [ChunkID(4B)] [Timestamp(8B)] [Duration(4B)] [PayloadLen(4B)] [Payload]
        const flags = view.getUint8(1);
        const isKeyframe = Boolean(flags & 0x01);
        const chunkId = view.getUint32(2, false);
        const timestamp = view.getFloat64(6, false);
        const duration = view.getFloat32(14, false);
        const payloadLen = view.getUint32(18, false);

        this.currentChunkIdx = chunkId;
        this.chunkDuration = duration;
        this.chunkStartWallTime = performance.now() / 1000.0;
        this.totalPackets++;
        this.currentBitrateKbps = (buffer.byteLength * 8.0 / 1000.0) / Math.max(0.01, duration);

        const payloadBuffer = buffer.slice(22, 22 + payloadLen);
        this._unpackPayload(payloadBuffer, isKeyframe);
    }

    _unpackPayload(payloadBuffer, isKeyframe) {
        // Differential INT8 unpack
        const view = new DataView(payloadBuffer);
        const codec = view.getUint8(0);
        const dataBytes = new Uint8Array(payloadBuffer, 1);

        // Simple fallback parser for raw or JSON-indexed packet
        try {
            // Check if meta length is embedded
            if (dataBytes.byteLength >= 4) {
                const metaLen = (dataBytes[0] << 24) | (dataBytes[1] << 16) | (dataBytes[2] << 8) | dataBytes[3];
                if (metaLen > 0 && metaLen < dataBytes.byteLength - 4) {
                    const metaStr = new TextDecoder().decode(dataBytes.subarray(4, 4 + metaLen));
                    const metaList = JSON.parse(metaStr);
                    const int8Raw = new Int8Array(dataBytes.buffer, dataBytes.byteOffset + 4 + metaLen);

                    // Reconstruct Float32 continuous array
                    let totalFloats = 0;
                    for (const item of metaList) totalFloats += item.l;

                    const floatArray = new Float32Array(totalFloats);
                    let dst = 0;

                    for (const item of metaList) {
                        const scale = item.c;
                        const off = item.o;
                        const len = item.l;
                        for (let k = 0; k < len; k++) {
                            const diff = int8Raw[off + k] * scale;
                            if (!isKeyframe && this.prevWeights && dst < this.prevWeights.length) {
                                floatArray[dst] = this.prevWeights[dst] + diff;
                            } else {
                                floatArray[dst] = diff;
                            }
                            dst++;
                        }
                    }

                    this.prevWeights = floatArray;
                    this.activeWeights = floatArray;

                    if (this.onChunkReceived) {
                        this.onChunkReceived(floatArray, {
                            chunkId: this.currentChunkIdx,
                            isKeyframe,
                            bitrateKbps: this.currentBitrateKbps,
                        });
                    }
                }
            }
        } catch (e) {
            console.warn("[Siren-Cast Web] Decompression note:", e);
        }
    }

    /**
     * Compute current continuous local coordinate t_local in [-1.0, 1.0]
     */
    getCurrentLocalTime() {
        if (!this.chunkStartWallTime) return 0.0;
        const elapsed = (performance.now() / 1000.0) - this.chunkStartWallTime;
        const rel = elapsed / Math.max(0.01, this.chunkDuration);
        return -1.0 + 2.0 * Math.min(1.2, rel); // Smooth extrapolation
    }
}
