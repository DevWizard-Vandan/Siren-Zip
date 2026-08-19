# PROVISIONAL PATENT APPLICATION DISCLOSURE

**UNITED STATES PATENT AND TRADEMARK OFFICE (USPTO) / PCT SPECIFICATION**

---

## TITLE OF THE INVENTION
**SYSTEM AND METHOD FOR SPATIO-TEMPORAL IMPLICIT NEURAL VIDEO COMPRESSION WITH DYNAMIC VIEWPORT CULLING AND ASYNCHRONOUS WEIGHT STREAMING**

**Inventors:** Vandan Patel et al.  
**Assignee:** Siren-Zip Technologies / Research Laboratories  

---

## I. FIELD OF THE INVENTION
The present invention relates generally to digital multimedia signal processing and neural data compression, and more particularly to systems and methods for encoding, multiplexing, and rendering video and multi-channel audio as continuous spatio-temporal implicit neural representations with hardware-synchronized master clocks and double-buffered weight prefetching.

---

## II. BACKGROUND OF THE INVENTION & PRIOR ART LIMITATIONS
Conventional video compression algorithms (e.g., MPEG-4, H.264/AVC, H.265/HEVC, VP9, AV1) encode video sequences as arrays of discrete pixels partitioned into macroblocks, relying on Discrete Cosine Transforms (DCT) and block motion vectors. 

These existing methods suffer from several critical technical deficiencies:
1. **Discrete Grid Quantization:** Visual information cannot be evaluated at continuous intermediate spatial coordinates, causing severe macroblocking and pixelation under high magnification.
2. **Fixed Frame-Rate Rigidity:** Temporal interpolation requires computationally expensive optical flow estimation, which frequently introduces visual warping and ghosting artifacts.
3. **Audio-Video Desynchronization:** Clocks drift over time due to quantized video frame boundaries vs continuous audio DAC clocks.
4. **Bandwidth & Storage Bloat:** Ultra-high-definition (4K/8K) archival requires proportional scaling of raw pixel storage ($\mathcal{O}(H \times W \times T)$).

Therefore, there is an unmet industrial need for a continuous mathematical compression architecture that decouples file size from rendering resolution while maintaining zero-drift hardware synchronization.

---

## III. SUMMARY OF THE INVENTION
The present invention provides a continuous implicit neural video codec (**Siren-Zip**) and container architecture (`.neura 2.0`) comprising:
1. **Anisotropic Spatio-Temporal Neural Manifold:** An MLP parameterized by periodic sinusoidal activations with distinct spatial ($\omega_{xy}$) and temporal ($\omega_t$) carrier frequencies.
2. **Neural GOP (Group of Pictures) Auto-Chunking:** An automated temporal decomposition engine partitioning continuous movies into discrete, independently paginated neural parameter blocks.
3. **Asynchronous Double-Buffered CUDA Prefetcher:** A lookahead streaming mechanism eliminating chunk boundary latency without video frame drops.
4. **Hardware Audio DAC Master Clock Synchronization:** An architecture that drives neural coordinate queries directly from the hardware audio clock, ensuring mathematically zero lip-sync drift.
5. **Dynamic Spatio-Temporal Viewport Culling:** A real-time rendering system evaluating coordinates strictly within the visible viewport, saving $>99.9\%$ of floating-point operations during high magnification.

---

## IV. PATENT CLAIMS (10 CLAIMS)

### We Claim:

1. **A computer-implemented method for compressing and rendering continuous video signals, comprising:**
   - Receiving an input video sequence having spatial dimensions $(W, H)$ and temporal duration $T$;
   - Partitioning said temporal duration $T$ into a plurality of $K$ temporal chunks of duration $\tau$;
   - Optimizing, for each of said $K$ temporal chunks, a set of neural network parameters $\theta_k$ comprising periodic sinusoidal activation layers with anisotropic spatial and temporal frequency scaling;
   - Serializing said plurality of parameter sets $\theta_k$ into a binary container format comprising a fixed-size header and a temporal seek index table; and
   - Rendering a video frame at an arbitrary spatial resolution $(W', H')$ and arbitrary continuous timestamp $t \in [0.0, T]$ by querying said neural parameters without discrete pixel interpolation.

2. **The method of claim 1, wherein said periodic sinusoidal activation layers comprise an initial layer:**
   $$\phi_0(x, y, t) = \sin\left( \mathbf{W}_{xy} \cdot [x, y]^T \cdot \omega_{xy} + \mathbf{W}_t \cdot [t]^T \cdot \omega_t + \mathbf{b}_0 \right)$$
   wherein spatial carrier frequency $\omega_{xy}$ is strictly greater than temporal carrier frequency $\omega_t$.

3. **The method of claim 1, further comprising:**
   - Preallocating a single neural network execution shell in a graphics processing unit (GPU) memory; and
   - Memory-mapping said binary container to page parameter sets $\theta_k$ into said execution shell in under two milliseconds upon receiving a temporal seek request.

4. **The method of claim 1, further comprising an asynchronous double-buffered prefetching process wherein:**
   - A background thread monitors active playback progress within temporal chunk $k$;
   - In response to temporal progress exceeding a predetermined threshold percentage, asynchronously pre-loading parameter set $\theta_{k+1}$ into GPU memory on a secondary compute stream prior to completion of chunk $k$;
   - Whereby playback transition between chunk $k$ and chunk $k+1$ occurs with zero dropped video frames.

5. **The method of claim 1, wherein said binary container format comprises:**
   - A 128-byte aligned container header containing container identification magic bytes, global duration, native resolution, and audio format descriptors;
   - A seek index table comprising $K$ records, each defining chunk index, start timestamp, end timestamp, frame count, byte offset, and byte size;
   - An audio payload buffer multiplexing compressed multi-channel audio data; and
   - A contiguous block of quantized INT8 weight matrices representing said plurality of parameter sets.

6. **The method of claim 5, further comprising:**
   - Decoding said multiplexed audio payload buffer into an audio digital-to-analog converter (DAC) buffer;
   - Querying an instantaneous hardware DAC timestamp $t_{\text{master}}$ from said audio hardware; and
   - Driving the continuous video evaluation timestamp $t_{\text{local}} = g(t_{\text{master}})$, wherein temporal lip-sync drift between audio and video is maintained at $0.0\text{ milliseconds}$.

7. **The method of claim 1, wherein rendering said video frame comprises Dynamic Viewport Culling, comprising:**
   - Receiving normalized user viewport bounds $(x_{\min}, x_{\max}, y_{\min}, y_{\max}) \subseteq [-1.0, 1.0]^2$;
   - Generating coordinate grid queries exclusively within said viewport bounds; and
   - Skipping forward-pass neural evaluation for coordinates outside said viewport bounds, saving greater than 90% of arithmetic tensor operations at zoom levels exceeding $4.0\times$.

8. **The method of claim 1, wherein said neural network parameters are trained directly in 10-bit SMPTE ST.2084 Perceptual Quantizer (PQ) high dynamic range (HDR) color space, and wherein rendering applies an ACES Filmic tone-mapping curve on standard dynamic range (SDR) displays.**

9. **A non-transitory computer-readable storage medium storing a compressed neural media container, comprising:**
   - A 128-byte aligned container header;
   - A seek index table mapping global timestamps to discrete byte offsets;
   - A multiplexed audio stream; and
   - A plurality of symmetric INT8 quantized weight matrices parameterizing continuous spatio-temporal sinusoidal fields.

10. **A computing apparatus for continuous neural media playback, comprising a processor, a graphics processing unit (GPU), and a memory storing instructions configured to execute the method of claim 1.**
