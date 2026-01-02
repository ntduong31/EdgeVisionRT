# EdgeVision RT

**Production-ready YOLOv8n inference system optimized for Raspberry Pi 5 **

![Performance](https://img.shields.io/badge/FPS-33.5_(No_Display)-brightgreen)
![Display FPS](https://img.shields.io/badge/FPS-29.6_(With_Display)-green)
![Latency](https://img.shields.io/badge/Latency-34ms-brightgreen)
![Status](https://img.shields.io/badge/Status-OPTIMIZED-success)

---

## Key Optimizations

This project implements radical optimizations to maximize the Cortex-A76 performance on Raspberry Pi 5:

### 1. **Hand-Tuned ARM64 Assembly Kernels**
*   **`rgb_to_chw_fp32_asm`**: Custom assembly using NEON `LD3` instructions for interleaved loading and vectorized floating-point normalization. Replaces standard C++ loops.
*   **`transpose_84x8400_asm`**: Optimized tensor transpose kernel for YOLOv8 output decoding.
*   **`memcpy_neon_asm`**: Prefetch-optimized memory copy for large buffers.

### 2. **System-Level Forensics**
*   **CPU Governor Pinning**: Scripts to force `performance` governor to prevent clock downscaling.
*   **Thread Affinity Pinning**: 
    *   **NCNN Threads (0-2)**: Pinned to specific cores to avoid context switching.
    *   **Display Thread (3)**: Isolated on a separate core to prevent cache contention with inference.

### 3. **Display Pipeline Innovation**
*   **Framebuffer Direct (`--fb`)**: Direct writing to `/dev/fb0` (DRM/KMS), bypassing the entire X11/Wayland/Qt stack. Zero-copy, zero-contention display.
*   **Async Non-Blocking**: Display thread uses `std::try_lock` to drop frames rather than blocking the inference pipeline.
*   **YUYV Hardware Acceleration**: Optimized color conversion for webcam inputs.

### 4. **NCNN Configuration**
*   **Thread Auto-Tuning**: Automatically switches between 4 threads (max throughput) and 3 threads (display mode) to minimize latency jitter.
*   **FP16/INT8 Support**: Quantization support for further speedups.

---

## Performance Benchmarks

**Target**: YOLOv8n @ 416x416 Input

| Mode | FPS (Mean) | FPS (P99) | Inference Latency | Notes |
|------|------------|-----------|-------------------|-------|
| **No Display** | **33.5** | 26.0 | 30.5 ms | Pure inference speed |
| **OpenCV Display** | 29.6 | 23.9 | 34.2 ms | With optimized threading |
| **Framebuffer** | **>30.0** | **>25.0** | 32.0 ms | Bypasses X11 overhead |

---

## Quick Start

### 1. Build
```bash
cd /home/pi/AI/EdgeVisionRT
./build.sh
```

### 2. Run with `run.sh`
The `run.sh` script handles governor settings, thread affinity, and library paths automatically.

#### Webcam Mode (NEW!)
```bash
# Webcam + OpenCV Display
./run.sh cam display

# Webcam + Framebuffer (Max FPS, bypass X11)
./run.sh cam fb

# Webcam + Class Filter
./run.sh cam display class person
```

#### Video File Mode
```bash
# Benchmark (No display)
./run.sh

# Video + Display
./run.sh display

# Video + Framebuffer
./run.sh fb
```

---

## Display Modes Explained

### 1. OpenCV Display (`display`)
*   **Standard Window**: Uses highgui (Qt/X11).
*   **Pros**: Easy to move/resize windows, works in desktop environment.
*   **Cons**: ~10-15% overhead due to X11/Qt cache pollution.
*   **Best for**: Development, debugging, desktop usage.

### 2. Framebuffer Display (`fb`)
*   **Direct Hardware Access**: Writes directly to `/dev/fb0`.
*   **Pros**: **Fastest possible display**. Zero X11 overhead. Works in console mode / headless.
*   **Cons**: Overlays on top of desktop. Requires permission (`sudo chmod 666 /dev/fb0`).
*   **Best for**: Production, embedded kiosks, max FPS requirements.

---

## Project Structure

```
EdgeVisionRT/
├── src/
│   ├── asm_kernels.S        # <--- ARM64 ASSEMBLY KERNELS
│   ├── main.cpp             # Main loop & logic
│   ├── neon_preprocess.cpp  # C++ NEON glue
│   ├── inference_engine.cpp # NCNN wrapper
│   └── ...
├── include/
│   ├── asm_kernels.h        # Assembly headers
│   ├── drm_display.h        # <--- DIRECT FRAMEBUFFER DRIVER
│   └── ...
├── models/                  # YOLOv8n NCNN models (fp32, fp16, int8)
├── run.sh                   # Smart launcher script
└── build.sh                 # CMake build script
```

---

## Troubleshooting

### Display is Black (Webcam)
*   **Cause**: Webcam outputs YUYV, but display expects BGR.
*   **Fix**: Already patched in `main.cpp` using optimized `cv::cvtColor` before push. ensuring `run.sh cam display` works.

### Permission Denied `/dev/fb0`
*   **Fix**: Run `sudo chmod 666 /dev/fb0` to allow user access to the framebuffer.

### Low FPS
*   **Check**: Ensure you are using `./run.sh` which sets the CPU governor to `performance`.
*   **Thermal**: Check if device is throttling (`vcgencmd measure_temp`).

---

## License
MIT