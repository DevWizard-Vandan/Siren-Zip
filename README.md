# ⚡ Siren-Zip: Implicit Neural Representation (INR) Video & Image Codec

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![PySide6 GUI](https://img.shields.io/badge/GUI-PySide6-purple.svg)](https://pypi.org/project/PySide6/)
[![Platform: CUDA / CPU](https://img.shields.io/badge/Platform-CUDA%20%7C%20CPU-success.svg)](https://developer.nvidia.com/cuda-toolkit)

**Replacing Discrete Pixels with Continuous Mathematics**  
*A complete implicit neural codec with continuous spatio-temporal representations, 400X analytical zoom, 4X continuous temporal super-sampling, and a dark-mode interactive desktop player.*

[Key Innovations](#-key-innovations) •
[Mathematical Foundations](#-mathematical-foundations) •
[Architecture](#-system-architecture) •
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
3. **Bandwidth Inefficiency:** High-resolution videos scale memory quadratically with resolution: $\mathcal{O}(H \times W \times T)$.

### 🌟 The Siren-Zip Solution
**Siren-Zip** discards the concept of discrete pixel arrays entirely. Instead, a video or image is parameterized as a compact, continuous neural network:
$$f_\theta(x, y, t) \longrightarrow (r, g, b), \quad \text{where } (x, y, t) \in [-1.0, 1.0]^3 \subset \mathbb{R}$$

```
 TRADITIONAL VIDEO (H.264 / AV1)                 SIREN-ZIP (.NEURA)
 ┌───────────────────────────────┐               ┌───────────────────────────────┐
 │ Discrete Pixel Array (Bitmaps)│               │ Continuous Mathematical Field │
 │ Blocky Macroblocks at Zoom    │  ───────────► │ Infinite Spatial Continuity   │
 │ Stuttery Frame Duplication    │               │ Infinite Temporal Smoothness  │
 │ Memory = Pixels × Time        │               │ Memory = θ (Weights & Biases) │
 └───────────────────────────────┘               └───────────────────────────────┘
```

---

## 🧠 Key Innovations

1. **Sitzmann Sinusoidal Representations (SIREN):**
   Periodic sine activations $\phi_i(\mathbf{x}) = \sin(\omega_0(\mathbf{W}_i\mathbf{x} + \mathbf{b}_i))$ paired with exact uniform weight initialization prevent spectral bias and preserve fine high-frequency derivatives.
2. **Anisotropic Spatio-Temporal Frequency Scaling:**
   Spatial structures oscillate at higher frequencies ($\omega_{xy} = 30.0$) than temporal motion across frames ($\omega_t = 10.0$), ensuring razor-sharp edges without temporal flicker.
3. **Continuous Temporal Super-Sampling (Slow-Mo without Optical Flow):**
   Sampling arbitrary floating-point timestamps $t \in \mathbb{R}$ produces continuous, buttery-smooth slow motion at any fractional frame rate without interpolation blur.
4. **Proprietary 128-Byte Aligned `.neura` Binary Container:**
   Fixed 128-byte header packed with metadata, architecture hyperparameters, and symmetric min-max INT8 quantized parameter arrays ($298\times$ compression over raw video).
5. **Dynamic Spatio-Temporal Viewport Culling:**
   When zooming up to $400\times$, the inference engine evaluates **only the visible viewport coordinates**, skipping $>99\%$ of off-screen coordinate computations for constant-latency 60 FPS playback.

---

## 📐 Mathematical Foundations

### 1. Sinusoidal Activation Layer
$$\phi_i(\mathbf{x}) = \sin\left(\omega_0 \left(\mathbf{W}_i \mathbf{x} + \mathbf{b}_i\right)\right)$$

The derivative of a SIREN layer is itself a scaled cosine representation:
$$\nabla_\mathbf{x} \phi_i(\mathbf{x}) = \omega_0 \mathbf{W}_i^T \cos\left(\omega_0 \left(\mathbf{W}_i \mathbf{x} + \mathbf{b}_i\right)\right)$$
This preserves gradient flow and prevents vanishing gradients across high-frequency boundaries.

### 2. Sitzmann Uniform Initialization
To preserve unit variance across all hidden layers:
$$\mathbf{W}_0 \sim \mathcal{U}\left(-\frac{1}{n_{\text{in}}}, \frac{1}{n_{\text{in}}}\right), \quad \mathbf{W}_i \sim \mathcal{U}\left(-\frac{\sqrt{6 / n_{\text{in}}}}{\omega_0}, \frac{\sqrt{6 / n_{\text{in}}}}{\omega_0}\right)$$

### 3. Anisotropic Input Formulation
$$\phi_0(x, y, t) = \sin\left(\mathbf{W}_{xy} \cdot [x, y]^T \cdot \omega_{xy} + \mathbf{W}_t \cdot [t]^T \cdot \omega_t + \mathbf{b}_0\right)$$

---

## 📁 System Architecture

```
siren-zip/
├── LICENSE                       # MIT Open Source License
├── requirements.txt              # PyTorch, TorchVision, OpenCV, TorchMetrics, PySide6
├── README.md                     # Technical blueprint & documentation
├── Short_Clip_720p.mp4           # 720p 96-frame benchmark video
├── my_video.neura                # Proprietary 128-byte aligned INT8 container (869.6 KB)
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
│   ├── player/
│   │   ├── engine.py             # GPU inference engine with Dynamic Viewport Culling & LOD
│   │   └── neura_reader.py       # Reads 128-byte header & dequantizes .neura container
│   ├── ui/
│   │   ├── main_window.py        # Modern dark-mode PySide6 desktop application
│   │   ├── video_canvas.py       # Interactive canvas with smooth pan & 400x zoom
│   │   └── split_view.py         # Draggable split-screen (Discrete H.264 vs SIREN-Zip)
│   ├── model/
│   │   ├── siren_video.py        # Spatio-temporal SIREN with anisotropic frequency scaling
│   │   ├── siren.py              # Pure 2D SIREN for static imagery
│   │   └── quantizer.py          # Symmetric INT8 quantization & .neura serialization engine
│   ├── data/
│   │   ├── video_coordinate_dataset.py # GPU contiguous 1D video coordinate sampler
│   │   └── coordinate_dataset.py       # 2D image coordinate dataset
│   ├── training/
│   │   ├── video_trainer.py      # High-throughput GPU video trainer with TF32
│   │   └── trainer.py            # Static image trainer
│   └── utils/
│       ├── neura_format.py       # 128-byte aligned binary container packer
│       └── metrics.py            # Vectorized GPU PSNR & SSIM metrics
└── scripts/
    ├── launch_player.py          # Entrypoint to launch the Siren Player desktop GUI
    ├── benchmark_throughput.py   # Latency & FPS benchmark across resolutions on RTX GPU
    ├── train_video.py            # CLI for training video INR
    ├── render_video.py           # CLI for rendering SIREN back to MP4
    ├── slow_motion_demo.py       # 4X continuous temporal super-sampling proof
    ├── export_neura.py           # Exports checkpoint to quantized .neura binary
    ├── train_image.py            # 2D image training CLI
    ├── continuous_zoom.py        # Sub-pixel continuous zoom demonstration
    └── compare_baseline.py       # Compression benchmark against JPEG & WebP
```

---

## 📊 Benchmark Results

### 1. Spatio-Temporal Video Compression ($1280 \times 720$, 96 Frames)

| Asset / Codec | Storage Size | Compression Ratio vs Raw | Reconstruction PSNR | Mean SSIM |
| :--- | :--- | :--- | :--- | :--- |
| **Raw Uncompressed 720p** | **253.1 MB** | **1.0x** | $\infty$ | **1.0000** |
| **Standard H.264 MP4** | **1,129.8 KB** | **229.4x** | Reference | Reference |
| **PyTorch .pth Checkpoint** | **10,446.7 KB** | **24.8x** | 29.23 dB | 0.8718 |
| **SIREN-ZIP (.neura INT8)** | **`869.6 KB`** | **`298.1x`** | **29.23 dB** | **0.8718** |

> 💡 **Key Result:** The `.neura` container is **23% smaller than the compressed H.264 MP4** while unlocking **infinite continuous resolution** and continuous temporal super-sampling!

### 2. Static 2K Benchmark Target ($2048 \times 2048$)

| Codec | File Size | Compression Ratio | Reconstruction PSNR | SSIM |
| :--- | :--- | :--- | :--- | :--- |
| **Raw 2K RGB** | **12.3 MB** | **1.0x** | $\infty$ | **1.0000** |
| **SIREN-ZIP (INT8)** | **`322.8 KB`** | **`38.1x`** | **26.75 dB** | **`0.9295`** |
| JPEG (Q=10) | 129.4 KB | 94.9x | 27.45 dB | 0.8866 |
| WebP (Q=10) | 79.7 KB | 154.2x | 34.27 dB | 0.9616 |

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
│   [ ▶ Play ]  [ ⏸ Pause ]   Timeline: ──●──────────────────────── (t = 0.4231s)        │
│   Speed: [ 0.1x | 0.25x | 0.5x | 1.0x | 2.0x | 4.0x Continuous Slow-Mo ]               │
│   Zoom Tool: [ 1.0x ──────●────── 400.0x Continuous Viewport ]                         │
│                                                                                        │
│   HUD: 869.6 KB | 60 FPS | Dynamic Viewport Culling: ACTIVE (99.7% Compute Saved)      │
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

### 2. Launch the Siren Player Desktop App
```bash
python scripts/launch_player.py --file my_video.neura --baseline Short_Clip_720p.mp4
```

### 3. Train Spatio-Temporal Video SIREN
```bash
python scripts/train_video.py --video_path Short_Clip_720p.mp4 --epochs 2000 --batch_size 65536
```

### 4. Render 4X Continuous Slow-Motion Video
```bash
python scripts/slow_motion_demo.py --checkpoint checkpoints/best_video_siren.pth --fps_multiplier 4.0
```

### 5. Export to Quantized `.neura` Container
```bash
python scripts/export_neura.py --checkpoint checkpoints/best_video_siren.pth --output my_video.neura
```

### 6. Profile GPU Inference Latency & Viewport Culling
```bash
python scripts/benchmark_throughput.py --neura my_video.neura
```

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
