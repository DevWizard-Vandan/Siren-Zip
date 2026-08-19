/**
 * Binary Parser for .neura 1.0 & .neura 2.0 Implicit Neural Containers.
 * Performs memory-mapped parsing of the 128B header, Seek Index Table, and INT8 dequantization.
 */

class NeuraParser {
    constructor() {
        this.buffer = null;
        this.view = null;
        this.version = 2;
        this.header = {};
        this.indexTable = [];
        this.audioPayload = null;
        this.chunkCache = new Map();
    }

    /**
     * Parse binary .neura file from ArrayBuffer
     * @param {ArrayBuffer} arrayBuffer
     */
    parseContainer(arrayBuffer) {
        this.buffer = arrayBuffer;
        this.view = new DataView(arrayBuffer);
        this.chunkCache.clear();

        if (arrayBuffer.byteLength < 128) {
            throw new Error("Invalid .neura file: Size is less than 128-byte header minimum.");
        }

        // Read 4-byte Magic Identifier
        const magic = String.fromCharCode(
            this.view.getUint8(0),
            this.view.getUint8(1),
            this.view.getUint8(2),
            this.view.getUint8(3)
        );

        if (magic === "NEU2") {
            this.version = 2;
            this._parseHeaderV2();
            this._parseIndexTable();
        } else if (magic === "NEUR") {
            this.version = 1;
            this._parseHeaderV1();
        } else {
            throw new Error(`Unrecognized .neura container magic bytes: '${magic}'. Expected 'NEU2' or 'NEUR'.`);
        }

        return this.header;
    }

    _parseHeaderV2() {
        const v = this.view;
        this.header = {
            magic: "NEU2",
            version: v.getUint32(4, true),
            total_chunks: v.getUint32(8, true),
            total_duration: v.getFloat64(12, true),
            fps: v.getFloat32(20, true),
            width: v.getUint32(24, true),
            height: v.getUint32(28, true),
            chunk_duration: v.getFloat32(32, true),
            hidden_layers: v.getUint32(36, true),
            hidden_features: v.getUint32(40, true),
            omega_xy: v.getFloat32(44, true),
            omega_t: v.getFloat32(48, true),
            omega_0_hidden: v.getFloat32(52, true),
            final_activation_id: v.getUint32(56, true),
            num_tensors_per_chunk: v.getUint32(60, true),
            index_table_offset: Number(v.getBigUint64(64, true)),
            index_table_size: Number(v.getBigUint64(72, true)),
            audio_codec_type: v.getUint32(80, true),
            audio_sample_rate: v.getUint32(84, true),
            audio_channels: v.getUint32(88, true),
            color_primaries: v.getUint32(92, true),
            transfer_characteristics: v.getUint32(96, true),
            audio_payload_offset: Number(v.getBigUint64(100, true)),
            audio_payload_size: Number(v.getBigUint64(108, true)),
        };
    }

    _parseHeaderV1() {
        const v = this.view;
        const frameCount = v.getUint32(8, true);
        const fps = v.getFloat32(12, true) || 24.0;
        this.header = {
            magic: "NEUR",
            version: v.getUint32(4, true),
            total_chunks: 1,
            frame_count: frameCount,
            total_duration: frameCount / fps,
            fps: fps,
            width: v.getUint32(16, true),
            height: v.getUint32(20, true),
            chunk_duration: frameCount / fps,
            hidden_layers: v.getUint32(24, true),
            hidden_features: v.getUint32(28, true),
            omega_xy: v.getFloat32(32, true),
            omega_t: v.getFloat32(36, true),
            omega_0_hidden: v.getFloat32(40, true),
            final_activation_id: v.getUint32(44, true),
            num_tensors: v.getUint32(48, true),
            payload_size_bytes: Number(v.getBigUint64(52, true)),
            index_table_offset: 0,
            index_table_size: 0,
            audio_payload_size: 0,
        };
    }

    _parseIndexTable() {
        const offset = this.header.index_table_offset;
        const totalChunks = this.header.total_chunks;
        const recordSize = 40; // 4 + 8 + 8 + 4 + 8 + 8

        this.indexTable = [];
        let ptr = offset;

        for (let i = 0; i < totalChunks; i++) {
            if (ptr + recordSize > this.buffer.byteLength) break;
            const chunk_idx = this.view.getUint32(ptr, true);
            const start_time = this.view.getFloat64(ptr + 4, true);
            const end_time = this.view.getFloat64(ptr + 12, true);
            const num_frames = this.view.getUint32(ptr + 20, true);
            const byte_offset = Number(this.view.getBigUint64(ptr + 24, true));
            const byte_size = Number(this.view.getBigUint64(ptr + 32, true));

            this.indexTable.push({
                chunk_idx,
                start_time,
                end_time,
                num_frames,
                byte_offset,
                byte_size,
            });
            ptr += recordSize;
        }
    }

    /**
     * Locate chunk index and compute continuous local coordinate t_local in [-1.0, 1.0]
     * @param {number} t_global Global timestamp in seconds
     */
    locateChunkAndLocalTime(t_global) {
        if (this.version === 1 || this.indexTable.length === 0) {
            const alpha = Math.min(1.0, Math.max(0.0, t_global / Math.max(0.001, this.header.total_duration)));
            return {
                chunkIdx: 0,
                t_local: -1.0 + 2.0 * alpha,
                record: null,
            };
        }

        const clamped_t = Math.max(0.0, Math.min(this.header.total_duration, t_global));
        let low = 0;
        let high = this.indexTable.length - 1;
        let chosenIdx = 0;

        while (low <= high) {
            const mid = (low + high) >> 1;
            const rec = this.indexTable[mid];
            if (clamped_t >= rec.start_time && clamped_t < rec.end_time) {
                chosenIdx = mid;
                break;
            } else if (clamped_t < rec.start_time) {
                chosenIdx = mid;
                high = mid - 1;
            } else {
                chosenIdx = mid;
                low = mid + 1;
            }
        }

        const record = this.indexTable[chosenIdx];
        const span = Math.max(1e-4, record.end_time - record.start_time);
        const rel = (clamped_t - record.start_time) / span;
        const t_local = -1.0 + 2.0 * Math.max(0.0, Math.min(1.0, rel));

        return {
            chunkIdx: chosenIdx,
            t_local,
            record,
        };
    }

    /**
     * Deserialize INT8 weight payload into ordered Float32Array suitable for WebGPU buffer binding
     * @param {number} chunkIdx
     */
    getChunkWeightsFloat32(chunkIdx = 0) {
        if (this.chunkCache.has(chunkIdx)) {
            return this.chunkCache.get(chunkIdx);
        }

        let payloadBuffer, payloadOffset, numTensors;

        if (this.version === 2) {
            if (chunkIdx >= this.indexTable.length) return null;
            const rec = this.indexTable[chunkIdx];
            payloadOffset = rec.byte_offset;
            numTensors = this.header.num_tensors_per_chunk;
        } else {
            payloadOffset = 128;
            numTensors = this.header.num_tensors;
        }

        const tensorMap = this._deserializeTensors(payloadOffset, numTensors);
        const packedFloat32 = this._packForWebGPUShader(tensorMap);

        this.chunkCache.set(chunkIdx, packedFloat32);
        return packedFloat32;
    }

    _deserializeTensors(startOffset, numTensors) {
        const view = this.view;
        let ptr = startOffset;
        const tensorMap = new Map();

        for (let i = 0; i < numTensors; i++) {
            if (ptr >= this.buffer.byteLength) break;
            const nameLen = view.getUint16(ptr, true);
            ptr += 2;

            let name = "";
            for (let j = 0; j < nameLen; j++) {
                name += String.fromCharCode(view.getUint8(ptr + j));
            }
            ptr += nameLen;

            const rank = view.getUint16(ptr, true);
            ptr += 2;

            const shape = [];
            for (let j = 0; j < rank; j++) {
                shape.push(view.getUint32(ptr, true));
                ptr += 4;
            }

            const scale = view.getFloat32(ptr, true);
            const zeroPoint = view.getInt32(ptr + 4, true);
            ptr += 8;

            const dataLen = view.getUint32(ptr, true);
            ptr += 4;

            const int8Data = new Int8Array(this.buffer, ptr, dataLen);
            ptr += dataLen;

            // Dequantize to float32
            const floatArray = new Float32Array(int8Data.length);
            for (let k = 0; k < int8Data.length; k++) {
                floatArray[k] = int8Data[k] * scale;
            }

            tensorMap.set(name, {
                name,
                shape,
                scale,
                data: floatArray,
            });
        }

        return tensorMap;
    }

    _packForWebGPUShader(tensorMap) {
        const H = this.header.hidden_features;
        const L = this.header.hidden_layers;

        // Calculate total floats needed in continuous buffer:
        // Layer 0: W (H * 3) + b (H) = 4H
        // Layer 1..L-1: (L - 1) * (H * H + H)
        // Layer Out: (3 * H) + 3
        const totalFloats = (H * 3 + H) + (L - 1) * (H * H + H) + (3 * H + 3);
        const packed = new Float32Array(totalFloats);
        let dstOffset = 0;

        // 1. Layer 0
        const w0 = this._findTensor(tensorMap, "net.0.linear.weight") || this._findTensor(tensorMap, "0.weight");
        const b0 = this._findTensor(tensorMap, "net.0.linear.bias") || this._findTensor(tensorMap, "0.bias");
        if (w0) { packed.set(w0.data, dstOffset); dstOffset += w0.data.length; } else { dstOffset += H * 3; }
        if (b0) { packed.set(b0.data, dstOffset); dstOffset += b0.data.length; } else { dstOffset += H; }

        // 2. Hidden Layers 1..L-1
        for (let l = 1; l < L; l++) {
            const wl = this._findTensor(tensorMap, `net.${l}.linear.weight`) || this._findTensor(tensorMap, `${l}.weight`);
            const bl = this._findTensor(tensorMap, `net.${l}.linear.bias`) || this._findTensor(tensorMap, `${l}.bias`);
            if (wl) { packed.set(wl.data, dstOffset); dstOffset += wl.data.length; } else { dstOffset += H * H; }
            if (bl) { packed.set(bl.data, dstOffset); dstOffset += bl.data.length; } else { dstOffset += H; }
        }

        // 3. Final Linear Readout
        const wFinal = this._findTensor(tensorMap, "final_linear.weight") || this._findTensor(tensorMap, "out.weight");
        const bFinal = this._findTensor(tensorMap, "final_linear.bias") || this._findTensor(tensorMap, "out.bias");
        if (wFinal) { packed.set(wFinal.data, dstOffset); dstOffset += wFinal.data.length; } else { dstOffset += 3 * H; }
        if (bFinal) { packed.set(bFinal.data, dstOffset); dstOffset += bFinal.data.length; } else { dstOffset += 3; }

        return packed;
    }

    _findTensor(map, partialName) {
        for (const [key, val] of map.entries()) {
            if (key.includes(partialName)) return val;
        }
        return null;
    }
}
