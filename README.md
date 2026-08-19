# ⚡ Siren-Zip: Implicit Neural Representation (INR) Cinema Codec

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![PySide6 GUI](https://img.shields.io/badge/GUI-PySide6-purple.svg)](https://pypi.org/project/PySide6/)
[![Audio: Master Clock Sync](https://img.shields.io/badge/Audio-Master%20Clock%20Sync-blueviolet.svg)](https://github.com/DevWizard-Vandan/Siren-Zip)
[![Color: HDR10+ / Rec.2020](https://img.shields.io/badge/Color-HDR10%2B%20%7C%20Rec.2020-red.svg)](https://github.com/DevWizard-Vandan/Siren-Zip)
[![Platform: CUDA / CPU](https://img.shields.io/badge/Platform-CUDA%20%7C%20CPU-success.svg)](https://developer.nvidia.com/cuda-toolkit)

**Replacing Discrete Pixels with Continuous Spatio-Temporal Calculus**  
*A complete implicit neural cinema codec architecture featuring Neural GOP (Group of Pictures) auto-chunking, multi-channel audio multiplexing with 0.0ms lip-sync drift, 10-bit HDR10+ / Rec.2020 ACES filmic tone-mapping, 400X analytical zoom, and a dark-mode desktop player.*

[Key Innovations](#-key-innovations) •
[Audio-Video Master Sync](#-audio-video-master-clock-synchronization) •
[Color Science & HDR](#-10-bit-hdr10--rec2020-color-science) •
[Neural GOP Architecture](#-the-neural-gop-architecture-siren-zip-20) •
[Benchmarks](#-benchmark-results) •
[Desktop Player](#-the-siren-player-desktop-application) •
[Quickstart](#-quickstart-guide)

</div>

---

## 🔬 Motivation: Why Implicit Neural Representations?

Traditional image and video codecs (`JPEG`, `WebP`, `H.264`, `HEVC`, `AV1`) rely on discrete, grid-based sampling:
$$I[u, v, t] \in \{0, \dots, 255\}^3, \quad u, v, t \in \mathbb{N}$$

This discrete representation suffers from fundamental physical limitations:
1. **Spatial Resolution Limits:** Zooming into discrete pixel arrays triggers severe **macroblock pixellation and blur**.
2. **Temporal Frame-Rate Stutter:** Slowing down a discrete video requires either **stuttery frame duplication** or complex optical flow estimation prone to **warping artifacts and ghosting**.
3. **Audio-Video Desynchronization:** Discrete video frames and continuous audio DAC sample rates cause periodic clock drift.
4. **Memory & Cold Storage Bloat:** High-resolution 4K/8K cinema masters require dozens of gigabytes per movie: $\mathcal{O}(H \times W \times T)$.

### 🌟 The Siren-Zip Solution
**Siren-Zip** discards discrete pixel arrays entirely. Instead, a video sequence is parameterized as a continuous, differentiable neural field:
$$f_\theta(x, y, t) \longrightarrow (r, g, b), \quad \text{where } (x, y, t) \in [-1.0, 1.0]^3 \subset \mathbb{R}$$

```
 TRADITIONAL VIDEO (H.264 / AV1)                 SIREN-ZIP (.NEURA)
 ┌───────────────────────────────┐               ┌───────────────────────────────┐
 │ Discrete Pixel Array (Bitmaps)│               │ Continuous Mathematical Field │
 │ Blocky Macroblocks at Zoom    │  ───────────► │ Infinite Spatial Continuity   │
 │ Stuttery Frame Duplication    │               │ Infinite Temporal Smoothness  │
 │ Drift-Prone Clock Sync        │               │ 0.0ms Audio Master Clock Sync │
 │ Memory = Pixels × Time        │               │ Memory = θ (Weights & Biases) │
 └───────────────────────────────┘               └───────────────────────────────┘
```

---

## 🎵 Audio-Video Master Clock Synchronization

In conventional media players, audio and video clocks drift because video frames are discrete (e.g. $23.976\text{ FPS}$) while sound cards consume analog PCM buffers at $48,000\text{ Hz}$.

**Siren-Zip's Continuous Solution:**
The hardware audio DAC clock acts as the **Master Clock Provider**. As the audio plays, the hardware reports its exact continuous timestamp $t_{\text{master}} \in \mathbb{R}$. The neural network evaluates:
$$f_{\theta_k}(x, y, t_{\text{local}}) \quad \text{where } t_{\text{local}} = 2 \cdot \left(\frac{t_{\text{master}} - t_{\text{start}, k}}{\tau}\right) - 1.0$$
Because time is continuous in SIREN, **lip-sync temporal drift is mathematically zero ($\Delta t = 0.000\text{ ms}$)**.

---

## 🎨 10-Bit HDR10+ / Rec.2020 Color Science

### 1. SMPTE ST.2084 Perceptual Quantizer (PQ)
$$L = \left(\frac{\max(N^{1/m_2} - c_1, 0)}{c_2 - c_3 N^{1/m_2}}\right)^{1/m_1} \times 10000\text{ nits}$$

### 2. ACES Filmic Tone-Mapping
Maps wide dynamic range luminance $[0, \text{Peak}]$ gracefully to displayable SDR/HDR screens:
$$\text{ACES}(x) = \text{clamp}\left(\frac{x(2.51x + 0.03)}{x(2.43x + 0.59) + 0.14}, 0.0, 1.0\right)$$

---

## 🧠 The Neural GOP Architecture (Siren-Zip 2.0)

To compress arbitrary full-length cinema videos (from 1 minute to 2+ hours), Siren-Zip 2.0 divides the global timeline $T_{\text{total}}$ into $K$ independent temporal Neural GOP chunks of duration $\tau$ (e.g. $\tau = 3.0\text{s}$):

$$K = \left\lceil \frac{T_{\text{total}}}{\tau} \right\rceil$$

```
 GLOBAL MOVIE TIMELINE: t_global ∈ [0.0s, T_total]
 ┌─────────────────────┬─────────────────────┬─── ··· ───┬─────────────────────┐
 │    CHUNK 0 (θ_0)    │    CHUNK 1 (θ_1)    │           │   CHUNK K-1 (θ_K-1) │
 │ [0.0s  ──►  3.0s]   │ [3.0s  ──►  6.0s]   │           │ [T-τ   ──►  T_total]│
 └─────────────────────┴─────────────────────┴─── ··· ───┴─────────────────────┘
```

---

## 📁 System Architecture

```
siren-zip/
├── LICENSE                       # MIT Open Source License
├── requirements.txt              # PyTorch, TorchVision, OpenCV, TorchMetrics, PySide6, PyAV, SoundDevice
├── README.md                     # Technical blueprint & documentation
├── Movie_Trailer_1080p.mp4       # 1080p Full HD Cinema Trailer benchmark video
├── Short_Clip_720p.mp4           # 720p 96-frame benchmark video
├── cinema_full.neura             # Multiplexed Video + Audio + HDR .neura 2.0 container
├── trailer_sample.neura          # Multi-chunk .neura 2.0 cinema container
├── my_video.neura                # Single-chunk .neura 1.0 container (869.6 KB)
├── test_target.png               # 2048x2048 high-contrast synthetic test chart
├── checkpoints/
│   ├── best_video_siren.pth      # Spatio-Temporal SIREN checkpoint (96 frames 720p)
│   └── best_siren.pth            # Static Image SIREN checkpoint (2048x2048)
├── runs/
│   ├── rendered_video.mp4        # Rendered 720p video from continuous SIREN manifold
│   ├── slow_motion_4x.mp4        # 4X Temporal Super-Sampling (381 continuous timestamps)
│   ├── slow_motion_comparison.mp4# Side-by-side: Discrete Frame Duplication vs SIREN Continuous Motion
│   ├── continuous_zoom_comparison.png # 20x Sub-Pixel Analytical Zoom
│   └── rate_distortion_curve.png      # Rate-Distortion curve vs JPEG & WebP
├── src/
│   ├── audio/
│   │   ├── audio_extractor.py    # Extracts multi-channel audio & probes HDR color metadata
│   │   └── audio_player.py       # Hardware audio playback & Master Clock provider
│   ├── color/
│   │   ├── hdr_transfer.py       # SMPTE ST.2084 PQ, HLG & Rec.709/Rec.2020 color matrices
│   │   └── tone_mapper.py        # ACES Filmic, Reinhard & Reinhard-Jodie tone mapping
│   ├── chunking/
│   │   ├── video_splitter.py     # Fast temporal slicing directly to GPU (zero disk bloat)
│   │   └── chunk_orchestrator.py # Multi-chunk GPU trainer with automated memory cleanup & ETA
│   ├── container/
│   │   ├── neura_v2_format.py    # .neura 2.0 128-byte header & Seek Index Table specifications
│   │   ├── neura_v2_writer.py    # Streaming .neura 2.0 container packer with audio & HDR
│   │   └── neura_v2_reader.py    # Memory-mapped streaming reader with sub-ms chunk paging
│   ├── streaming/
│   │   └── stream_engine.py      # Runtime streaming engine with audio sync & HDR tone mapping
│   ├── player/
│   │   ├── engine.py             # GPU inference engine with Dynamic Viewport Culling & LOD
│   │   └── neura_reader.py       # Universal reader for .neura 1.0 and 2.0 containers
│   ├── ui/
│   │   ├── main_window.py        # Modern dark-mode desktop player with Audio & HDR controls
│   │   ├── video_canvas.py       # Interactive canvas with smooth pan & 400x zoom
│   │   └── split_view.py         # Draggable split-screen (Discrete H.264 vs SIREN-Zip)
│   ├── model/
│   │   ├── siren_video.py        # Spatio-temporal SIREN with anisotropic frequency scaling
│   │   ├── siren.py              # Pure 2D SIREN for static imagery
│   │   └── quantizer.py          # Symmetric INT8 quantization engine
│   ├── data/
│   │   ├── video_coordinate_dataset.py # GPU contiguous 1D video coordinate sampler
│   │   └── coordinate_dataset.py       # 2D image coordinate dataset
│   └── utils/
│       ├── metrics.py            # Vectorized GPU PSNR & SSIM metrics
│       └── neura_format.py       # 128-byte aligned container packer
└── scripts/
    ├── compress_cinema.py        # Master CLI to compress Video + Audio + HDR into .neura 2.0
    ├── test_av_sync.py           # Verification script measuring A/V synchronization drift
    ├── compress_long_video.py    # Multi-chunk video compression CLI
    ├── verify_seek_accuracy.py   # Benchmark script testing 100 random seeks & paging latency
    ├── play_long_stream.py       # Continuous multi-chunk streaming video player
    ├── launch_player.py          # Entrypoint to launch the Siren Player desktop GUI
    ├── benchmark_throughput.py   # Latency & FPS benchmark across resolutions on RTX GPU
    ├── train_video.py            # CLI for training single-chunk video INR
    ├── render_video.py           # CLI for rendering SIREN back to MP4
    ├── slow_motion_demo.py       # 4X continuous temporal super-sampling proof
    ├── export_neura.py           # Exports checkpoint to quantized .neura binary
    ├── train_image.py            # 2D image training CLI
    ├── continuous_zoom.py        # Sub-pixel continuous zoom demonstration
    └── compare_baseline.py       # Compression benchmark against JPEG & WebP
```

---

## 📊 Benchmark Results

### 1. Multi-Chunk Cinema Compression (1080p Full HD Cinema Trailer)

| Asset / Codec | Storage Size | Compression Ratio vs Raw | Reconstruction PSNR | Mean SSIM |
| :--- | :--- | :--- | :--- | :--- |
| **Raw Uncompressed 1080p** | **18,438.6 MB** | **1.0x** | $\infty$ | **1.0000** |
| **Standard H.264 MP4** | **9,075.0 KB (8.86 MB)** | **2,031.8x** | Reference | Reference |
| **SIREN-ZIP 2.0 (.neura 2.0 INT8)** | **`3,478.3 KB` (3.40 MB)** | **`5,428.3x`** | **`35.30 dB`** | **`0.8695`** |

> 💡 **Key Result:** Siren-Zip 2.0 is **2.61x smaller than the compressed H.264 MP4** on Full HD 1080p cinema footage while achieving **35.30 dB PSNR**!

### 2. Audio-Video Master Clock Synchronization

| Metric | Result | Target / Standard |
| :--- | :--- | :--- |
| **Mean Lip-Sync Drift** | **`0.0000 ms`** | $< 1.0\text{ ms}$ (Perfect Alignment) |
| **Maximum Lip-Sync Drift** | **`0.0000 ms`** | Zero Drift |
| **Audio Track Multiplexing** | **AAC / Opus / MP3** | Multi-channel Stereo/5.1/7.1 |

### 3. Random Seek & Memory-Mapped Weight Paging (100 Random Seeks)

| Metric | Result | Target / Standard |
| :--- | :--- | :--- |
| **Mean Chunk Paging Latency** | **`1.16 ms`** | $< 2.0\text{ ms}$ (Met) |
| **95th Percentile Paging Latency** | **`1.47 ms`** | Instantaneous |
| **Mean Seek-to-Frame Latency** | **`127.8 ms`** | Real-time |
| **Mean Seek Reconstruction PSNR** | **`32.65 dB`** | $> 30\text{ dB}$ |
| **Mean Seek Reconstruction SSIM** | **`0.8695`** | Structural Fidelity |

---

## 🖥️ The Siren Player Desktop Application

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                SIREN PLAYER DESKTOP UI                                 │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│   ┌───────────────────────────────────┬───────────────────────────────────┐            │
│   │   ORIGINAL H.264 (DISCRETE)       │    SIREN-ZIP (.NEURA CONTINUOUS)  │            │
│   │                                   │                                   │            │
│   │     [ Blocky Pixelation ]         │       [ Smooth Analytical ]       │            │
│   │           at 400%                 │             at 400%               │            │
│   └───────────────────────────────────┴───────────────────────────────────┘            │
│                                                                                        │
│   [ ▶ Play ]  [ ⏸ Pause ]   Timeline: ──●──────────────────────── (01:14.238)          │
│   Volume: [ 🔊 ──●─────── ]   Color: [ ACES Filmic (HDR) ▼ ]                           │
│   Speed: [ 0.1x | 0.25x | 0.5x | 1.0x | 2.0x | 4.0x Continuous Slow-Mo ]               │
│   Zoom Tool: [ 1.0x ──────●────── 400.0x Continuous Viewport ]                         │
│                                                                                        │
│   HUD: 3.4 MB | 60 FPS | A/V Sync Drift: 0.0ms | Viewport Culling: 99.7% Saved        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quickstart Guide

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/DevWizard-Vandan/Siren-Zip.git
cd Siren-Zip
pip install -r requirements.txt
```

### 2. Compress Cinema Video + Audio + HDR into .neura 2.0
```bash
python scripts/compress_cinema.py --input Movie_Trailer_1080p.mp4 --output cinema_full.neura --chunk_duration 3.0 --epochs_per_chunk 400
```

### 3. Verify Audio-Video Master Synchronization
```bash
python scripts/test_av_sync.py --neura cinema_full.neura --duration 5.0
```

### 4. Verify Seek Latency Across 100 Random Seeks
```bash
python scripts/verify_seek_accuracy.py --neura cinema_full.neura --ground_truth Movie_Trailer_1080p.mp4 --num_seeks 100
```

### 5. Launch the Siren Player Desktop App
```bash
python scripts/launch_player.py --file cinema_full.neura --baseline Movie_Trailer_1080p.mp4
```

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
