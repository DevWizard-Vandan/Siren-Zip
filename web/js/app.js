/**
 * Siren-Zip: WebGPU In-Browser Cinema Player Application Orchestrator.
 * Handles UI events, timeline scrubbing, 400x analytical zoom/pan, WebGPU loop, & live WebSockets.
 */

class SirenWebApp {
    constructor() {
        this.parser = new NeuraParser();
        this.engine = new WebGPUEngine();
        this.liveReceiver = new LiveWebReceiver();

        // Canvas & Viewport State
        this.canvas = document.getElementById("siren-canvas");
        this.viewport = { x_min: -1.0, x_max: 1.0, y_min: -1.0, y_max: 1.0 };
        this.centerX = 0.0;
        this.centerY = 0.0;
        this.zoomFactor = 1.0;

        // Playback State
        this.isPlaying = false;
        this.currentTime = 0.0;
        this.totalDuration = 12.0;
        this.playbackSpeed = 1.0;
        this.loopMode = 0; // 0: All, 1: One, 2: None
        this.toneMapMode = "aces";
        this.activeChunkIdx = -1;

        // Split View Mode
        this.isSplitMode = false;
        this.splitPos = 0.5; // normalized [0.0, 1.0]

        // Live Stream Mode
        this.isLiveStream = false;

        // Telemetry & FPS Tracking
        this.lastFrameTime = performance.now();
        this.fpsFrameCount = 0;
        this.currentFps = 60.0;
        this.fpsUpdateTime = performance.now();

        // Mouse Drag State
        this.isDragging = false;
        this.dragStartX = 0;
        this.dragStartY = 0;
        this.isDraggingSplit = false;

        this._bindElements();
        this._bindEvents();
    }

    async init() {
        try {
            // Fetch WGSL Shader Source
            const resp = await fetch("js/siren_shader.wgsl");
            if (!resp.ok) throw new Error("Could not load siren_shader.wgsl");
            const wgslCode = await resp.text();

            // Initialize WebGPU Compute Pipeline
            await this.engine.init(this.canvas, wgslCode);

            // Update Device Badge
            const devBadge = document.getElementById("hud-device-badge");
            if (devBadge) {
                devBadge.textContent = `GPU: ${this.engine.getAdapterName()}`;
            }

            // Load Sample Manifest
            this._loadSampleManifest();

            // Start Main 60 FPS Render Loop
            requestAnimationFrame(this._renderLoop.bind(this));
            console.log("[App] Siren-VLC WebGPU Player Ready.");
        } catch (err) {
            console.error("[App] Initialization failed:", err);
            this._showErrorBanner(err.message);
        }
    }

    _bindElements() {
        this.dropZone = document.getElementById("drop-zone");
        this.fileInput = document.getElementById("file-input");
        this.btnOpenFile = document.getElementById("btn-open-file");
        this.btnBrowseTrigger = document.getElementById("btn-browse-trigger");
        this.sampleSelect = document.getElementById("sample-select");
        this.btnPlayPause = document.getElementById("btn-play-pause");
        this.playIcon = document.getElementById("play-icon");
        this.timelineSlider = document.getElementById("timeline-slider");
        this.lblCurrentTime = document.getElementById("lbl-current-time");
        this.lblTotalTime = document.getElementById("lbl-total-time");
        this.selSpeed = document.getElementById("sel-playback-speed");
        this.selToneMap = document.getElementById("sel-tone-mapping");
        this.zoomSlider = document.getElementById("zoom-slider");
        this.lblZoomVal = document.getElementById("lbl-zoom-val");
        this.btnResetZoom = document.getElementById("btn-reset-zoom");
        this.btnStepBack = document.getElementById("btn-step-back");
        this.btnStepFwd = document.getElementById("btn-step-fwd");
        this.btnLoopMode = document.getElementById("btn-loop-mode");
        this.btnToggleSplit = document.getElementById("btn-toggle-split");
        this.splitDivider = document.getElementById("split-divider");
        this.btnFullscreen = document.getElementById("btn-fullscreen");
        this.btnSnapshot = document.getElementById("btn-snapshot");

        // Live Stream Modal
        this.modalLive = document.getElementById("modal-live-stream");
        this.btnOpenLiveModal = document.getElementById("btn-open-live-modal");
        this.btnCloseModal = document.getElementById("btn-close-modal");
        this.btnCancelModal = document.getElementById("btn-cancel-modal");
        this.btnConnectLive = document.getElementById("btn-connect-live-stream");
        this.txtWsUrl = document.getElementById("txt-ws-url");

        // HUD Badges
        this.hudFpsBadge = document.getElementById("hud-fps-badge");
        this.hudGopBadge = document.getElementById("hud-gop-badge");
    }

    _bindEvents() {
        // File Open & Drag & Drop
        this.btnOpenFile.addEventListener("click", () => this.fileInput.click());
        this.btnBrowseTrigger.addEventListener("click", () => this.fileInput.click());
        this.fileInput.addEventListener("change", (e) => {
            if (e.target.files.length > 0) this.loadFile(e.target.files[0]);
        });

        const viewportContainer = document.getElementById("viewport-container");
        window.addEventListener("dragover", (e) => e.preventDefault());
        window.addEventListener("drop", (e) => {
            e.preventDefault();
            if (e.dataTransfer.files.length > 0) {
                this.loadFile(e.dataTransfer.files[0]);
            }
        });

        // Sample Selector
        this.sampleSelect.addEventListener("change", (e) => {
            if (e.target.value) this.loadSample(e.target.value);
        });

        // Playback Controls
        this.btnPlayPause.addEventListener("click", () => this.togglePlay());
        this.timelineSlider.addEventListener("input", (e) => {
            const alpha = parseFloat(e.target.value) / 10000.0;
            this.seekTo(alpha * this.totalDuration);
        });

        this.selSpeed.addEventListener("change", (e) => {
            this.playbackSpeed = parseFloat(e.target.value);
        });

        this.selToneMap.addEventListener("change", (e) => {
            this.toneMapMode = e.target.value;
        });

        this.btnStepBack.addEventListener("click", () => this.seekTo(this.currentTime - 1.0));
        this.btnStepFwd.addEventListener("click", () => this.seekTo(this.currentTime + 1.0));

        this.btnLoopMode.addEventListener("click", () => {
            this.loopMode = (this.loopMode + 1) % 3;
            const labels = ["🔁 Loop: All", "🔂 Loop: One", "➡️ Loop: Off"];
            this.btnLoopMode.textContent = labels[this.loopMode];
        });

        // 400x Analytical Continuous Zoom & Pan
        this.zoomSlider.addEventListener("input", (e) => {
            this.setZoom(parseFloat(e.target.value));
        });
        this.btnResetZoom.addEventListener("click", () => this.resetView());

        this.canvas.addEventListener("wheel", (e) => {
            e.preventDefault();
            const delta = e.deltaY < 0 ? 1.15 : 0.87;
            this.setZoom(this.zoomFactor * delta);
        });

        this.canvas.addEventListener("mousedown", (e) => {
            this.isDragging = true;
            this.dragStartX = e.clientX;
            this.dragStartY = e.clientY;
        });

        window.addEventListener("mousemove", (e) => {
            if (this.isDragging && this.zoomFactor > 1.0) {
                const dx = (e.clientX - this.dragStartX) / this.canvas.clientWidth;
                const dy = (e.clientY - this.dragStartY) / this.canvas.clientHeight;
                this.dragStartX = e.clientX;
                this.dragStartY = e.clientY;

                const halfW = 1.0 / this.zoomFactor;
                const halfH = 1.0 / this.zoomFactor;
                this.centerX = Math.max(-1.0 + halfW, Math.min(1.0 - halfW, this.centerX - dx * 2.0 * halfW));
                this.centerY = Math.max(-1.0 + halfH, Math.min(1.0 - halfH, this.centerY - dy * 2.0 * halfH));
                this._updateViewportBounds();
            }

            if (this.isDraggingSplit) {
                const rect = this.canvas.getBoundingClientRect();
                const pos = Math.max(0.05, Math.min(0.95, (e.clientX - rect.left) / rect.width));
                this.splitPos = pos;
                this.splitDivider.style.left = `${pos * 100}%`;
            }
        });

        window.addEventListener("mouseup", () => {
            this.isDragging = false;
            this.isDraggingSplit = false;
        });

        // Split View Controls
        this.btnToggleSplit.addEventListener("click", () => {
            this.isSplitMode = !this.isSplitMode;
            this.splitDivider.style.display = this.isSplitMode ? "block" : "none";
            this.btnToggleSplit.classList.toggle("btn-primary", this.isSplitMode);
        });

        this.splitDivider.addEventListener("mousedown", (e) => {
            this.isDraggingSplit = true;
            e.stopPropagation();
        });

        // Fullscreen
        this.btnFullscreen.addEventListener("click", () => {
            if (!document.fullscreenElement) {
                viewportContainer.requestFullscreen().catch(() => {});
            } else {
                document.exitFullscreen().catch(() => {});
            }
        });

        // 4K Snapshot
        this.btnSnapshot.addEventListener("click", () => this.takeSnapshot());

        // Live Stream Modal Events
        this.btnOpenLiveModal.addEventListener("click", () => this.modalLive.classList.add("open"));
        this.btnCloseModal.addEventListener("click", () => this.modalLive.classList.remove("open"));
        this.btnCancelModal.addEventListener("click", () => this.modalLive.classList.remove("open"));
        this.btnConnectLive.addEventListener("click", () => {
            const url = this.txtWsUrl.value.trim();
            if (url) {
                this.modalLive.classList.remove("open");
                this.connectLiveStream(url);
            }
        });

        // Keyboard Shortcuts
        window.addEventListener("keydown", (e) => {
            if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;

            if (e.code === "Space") {
                e.preventDefault();
                this.togglePlay();
            } else if (e.code === "ArrowLeft") {
                this.seekTo(this.currentTime - 1.0);
            } else if (e.code === "ArrowRight") {
                this.seekTo(this.currentTime + 1.0);
            } else if (e.code === "KeyF") {
                this.btnFullscreen.click();
            } else if (e.code === "KeyS") {
                this.takeSnapshot();
            }
        });
    }

    async _loadSampleManifest() {
        try {
            const res = await fetch("samples/sample_manifest.json");
            if (res.ok) {
                const samples = await res.json();
                this.sampleSelect.innerHTML = '<option value="" disabled selected>📂 Load Sample Movie...</option>';
                for (const s of samples) {
                    const opt = document.createElement("option");
                    opt.value = s.file;
                    opt.textContent = `🎬 ${s.name}`;
                    this.sampleSelect.appendChild(opt);
                }
            }
        } catch (e) {}
    }

    async loadSample(url) {
        try {
            const res = await fetch(url);
            if (!res.ok) throw new Error(`Sample not found: ${url}`);
            const buf = await res.arrayBuffer();
            this.loadBuffer(buf, url.split("/").pop());
        } catch (e) {
            console.error("[App] Failed to load sample:", e);
            alert(`Could not load sample: ${e.message}`);
        }
    }

    async loadFile(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            this.loadBuffer(e.target.result, file.name);
        };
        reader.readAsArrayBuffer(file);
    }

    loadBuffer(arrayBuffer, filename = "cinema.neura") {
        try {
            if (this.isLiveStream) {
                this.liveReceiver.disconnect();
                this.isLiveStream = false;
            }

            const header = this.parser.parseContainer(arrayBuffer);
            this.totalDuration = header.total_duration || 12.0;
            this.currentTime = 0.0;
            this.activeChunkIdx = -1;

            // Hide drag & drop overlay
            this.dropZone.classList.add("hidden");

            // Format Total Time
            this.lblTotalTime.textContent = this._formatTime(this.totalDuration);
            this.lblCurrentTime.textContent = "00:00.00";
            this.timelineSlider.value = 0;

            // Page first chunk
            this._pageChunk(0);

            // Auto-start playback
            this.isPlaying = true;
            this.playIcon.textContent = "⏸ Pause (Space)";

            console.log(`[App] Loaded ${filename}: ${header.width}x${header.height} @ ${header.fps.toFixed(1)} FPS (${header.total_chunks} chunks)`);
        } catch (err) {
            alert(`Error parsing .neura container:\n${err.message}`);
        }
    }

    _pageChunk(chunkIdx) {
        if (chunkIdx === this.activeChunkIdx) return;
        const weights = this.parser.getChunkWeightsFloat32(chunkIdx);
        if (weights) {
            this.engine.updateWeights(weights);
            this.activeChunkIdx = chunkIdx;
        }
    }

    connectLiveStream(wsUrl) {
        this.isLiveStream = true;
        this.dropZone.classList.add("hidden");

        this.liveReceiver.onConnected = (meta) => {
            this.totalDuration = 0;
            this.currentTime = 0;
            this.lblTotalTime.textContent = "🔴 LIVE";
            this.lblCurrentTime.textContent = "00:00.00";
        };

        this.liveReceiver.onChunkReceived = (weights, info) => {
            this.engine.updateWeights(weights);
        };

        this.liveReceiver.connect(wsUrl);
        this.isPlaying = true;
        this.playIcon.textContent = "⏸ Pause (Space)";
    }

    togglePlay() {
        this.isPlaying = !this.isPlaying;
        this.playIcon.textContent = this.isPlaying ? "⏸ Pause (Space)" : "▶ Play (Space)";
    }

    seekTo(timeSec) {
        this.currentTime = Math.max(0.0, Math.min(this.totalDuration, timeSec));
        const alpha = this.currentTime / Math.max(0.001, this.totalDuration);
        this.timelineSlider.value = Math.floor(alpha * 10000);
        this.lblCurrentTime.textContent = this._formatTime(this.currentTime);

        const { chunkIdx } = this.parser.locateChunkAndLocalTime(this.currentTime);
        this._pageChunk(chunkIdx);
    }

    setZoom(val) {
        this.zoomFactor = Math.max(1.0, Math.min(400.0, val));
        this.zoomSlider.value = this.zoomFactor;
        this.lblZoomVal.textContent = `${this.zoomFactor.toFixed(1)}x`;
        this._updateViewportBounds();
    }

    resetView() {
        this.centerX = 0.0;
        this.centerY = 0.0;
        this.setZoom(1.0);
    }

    _updateViewportBounds() {
        const halfW = 1.0 / this.zoomFactor;
        const halfH = 1.0 / this.zoomFactor;
        this.viewport = {
            x_min: Math.max(-1.0, this.centerX - halfW),
            x_max: Math.min(1.0, this.centerX + halfW),
            y_min: Math.max(-1.0, this.centerY - halfH),
            y_max: Math.min(1.0, this.centerY + halfH),
        };
    }

    _formatTime(sec) {
        const m = Math.floor(sec / 60);
        const s = (sec % 60).toFixed(2);
        return `${String(m).padStart(2, '0')}:${String(s).padStart(5, '0')}`;
    }

    takeSnapshot() {
        // Evaluate continuous field to image
        const dataUrl = this.canvas.toDataURL("image/png");
        const a = document.createElement("a");
        a.href = dataUrl;
        a.download = `siren_snapshot_${Date.now()}.png`;
        a.click();
    }

    _renderLoop(now) {
        const dt = (now - this.lastFrameTime) / 1000.0;
        this.lastFrameTime = now;

        // Measure Frame Rate
        this.fpsFrameCount++;
        if (now - this.fpsUpdateTime >= 500.0) {
            this.currentFps = (this.fpsFrameCount / (now - this.fpsUpdateTime)) * 1000.0;
            this.fpsFrameCount = 0;
            this.fpsUpdateTime = now;
        }

        if (this.isLiveStream) {
            // Live WebSockets stream rendering
            const t_local = this.liveReceiver.getCurrentLocalTime();
            this.currentTime += dt;
            this.lblCurrentTime.textContent = this._formatTime(this.currentTime);

            this.engine.render({
                t_local,
                viewport: this.viewport,
                omega_xy: 30.0,
                omega_t: 10.0,
                omega_0_hidden: 30.0,
                hidden_features: 64,
                hidden_layers: 2,
                tone_map_mode: this.toneMapMode,
                zoom_factor: this.zoomFactor,
                split_pos: this.splitPos,
                is_split: this.isSplitMode,
            });

            this.hudFpsBadge.textContent = `🔴 LIVE | ${this.currentFps.toFixed(1)} FPS | ${this.engine.lastComputeTimeMs.toFixed(1)}ms`;
            this.hudGopBadge.textContent = `⚡ Chunk: #${this.liveReceiver.currentChunkIdx} | Bitrate: ${this.liveReceiver.currentBitrateKbps.toFixed(1)} kbps`;
        } else if (this.parser.buffer) {
            // Local .neura playback
            if (this.isPlaying) {
                this.currentTime += dt * this.playbackSpeed;

                if (this.currentTime >= this.totalDuration) {
                    if (this.loopMode === 0 || this.loopMode === 1) {
                        this.currentTime = 0.0;
                    } else {
                        this.currentTime = this.totalDuration;
                        this.isPlaying = false;
                        this.playIcon.textContent = "▶ Play (Space)";
                    }
                }

                const alpha = this.currentTime / Math.max(0.001, this.totalDuration);
                this.timelineSlider.value = Math.floor(alpha * 10000);
                this.lblCurrentTime.textContent = this._formatTime(this.currentTime);
            }

            const { chunkIdx, t_local } = this.parser.locateChunkAndLocalTime(this.currentTime);
            this._pageChunk(chunkIdx);

            const h = this.parser.header;
            this.engine.render({
                t_local,
                viewport: this.viewport,
                omega_xy: h.omega_xy,
                omega_t: h.omega_t,
                omega_0_hidden: h.omega_0_hidden,
                hidden_features: h.hidden_features,
                hidden_layers: h.hidden_layers,
                tone_map_mode: this.toneMapMode,
                zoom_factor: this.zoomFactor,
                split_pos: this.splitPos,
                is_split: this.isSplitMode,
            });

            const totalChunks = h.total_chunks || 1;
            this.hudFpsBadge.textContent = `⚡ ${this.currentFps.toFixed(1)} FPS | ${this.engine.lastComputeTimeMs.toFixed(1)}ms Compute`;
            this.hudGopBadge.textContent = `🎬 Chunk: [${String(chunkIdx + 1).padStart(2, '0')}/${String(totalChunks).padStart(2, '0')}] | Zoom: ${this.zoomFactor.toFixed(1)}x`;
        }

        requestAnimationFrame(this._renderLoop.bind(this));
    }

    _showErrorBanner(msg) {
        const dropText = document.querySelector(".drop-text");
        if (dropText) dropText.textContent = "WebGPU Error";
        const dropSub = document.querySelector(".drop-subtext");
        if (dropSub) dropSub.innerHTML = `<span style="color: #ff5555;">${msg}</span><br>Please use Google Chrome, Edge, or Safari with WebGPU enabled.`;
    }
}

// Launch application on DOM load
window.addEventListener("DOMContentLoaded", () => {
    const app = new SirenWebApp();
    app.init();
    window.sirenApp = app;
});
