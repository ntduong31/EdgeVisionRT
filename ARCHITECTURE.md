# YOLOv8n Realtime Inference System for Raspberry Pi 5

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          UNIFIED INPUT PIPELINE                              │
│  ┌─────────────────────┐    ┌─────────────────────┐                         │
│  │   V4L2 Camera       │    │   Video File        │                         │
│  │   (mmap zero-copy)  │    │   (FFmpeg decode)   │                         │
│  └──────────┬──────────┘    └──────────┬──────────┘                         │
│             │                          │                                     │
│             └──────────┬───────────────┘                                     │
│                        ▼                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    UNIFIED FRAME BUFFER                              │    │
│  │   Layout: 640x480 YUYV (contiguous, 16-byte aligned)                │    │
│  │   Stride: 1280 bytes (640 * 2)                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      NEON PREPROCESSING PIPELINE                             │
│  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────────┐    │
│  │ YUYV → RGB        │  │ Bilinear Resize   │  │ Normalize + Layout    │    │
│  │ NEON vectorized   │──│ 640x480 → 640x640 │──│ HWC→CHW, /255, pad    │    │
│  │ 8 pixels/iter     │  │ NEON + prefetch   │  │ to 640x640x3 FP16     │    │
│  └───────────────────┘  └───────────────────┘  └───────────────────────┘    │
│                                                                              │
│  Memory: Triple buffer ring, pre-allocated, pinned to L2 cache              │
│  Alignment: All buffers 64-byte aligned for cache line optimization         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         NCNN INFERENCE ENGINE                                │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  YOLOv8n Model (NCNN format, FP16 optimized)                        │    │
│  │  - Input: 640x640x3 FP16 (CHW layout, normalized)                   │    │
│  │  - Workers: 3 threads pinned to CPU 1-3 (Cortex-A76)                │    │
│  │  - CPU 0 reserved for camera/input thread                           │    │
│  │  - Static memory pool, no runtime allocation                        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         POST-PROCESSING                                      │
│  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────────┐    │
│  │ Decode Outputs    │  │ NMS (vectorized)  │  │ Detection Results     │    │
│  │ 8400 candidates   │──│ IoU threshold 0.45│──│ boxes, scores, classes│    │
│  └───────────────────┘  └───────────────────┘  └───────┬───────────────┘    │
└────────────────────────────────────────────────────────┼─────────────────────┘
                                                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         OUTPUT PIPELINE (CPU 0)                              │
│  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────────┐    │
│  │ Class Filter      │  │ Async Video Writer│  │ Async Display         │    │
│  │ (optional)        │──│ Queue: 2000 frames│──│ Queue: 5 frames       │    │
│  │ O(1) set lookup   │  │ MJPEG→H.264       │  │ Auto screen detect    │    │
│  └───────────────────┘  └───────────────────┘  └───────────────────────┘    │
│                                                                              │
│  - Class filter: std::set<int> for selected classes                         │
│  - Video writer: Pinned to CPU 0, nice 19, post-conversion FFmpeg           │
│  - Display: xrandr screen detection, 75% max size, centered window          │
└─────────────────────────────────────────────────────────────────────────────┘

## Thread Model

┌────────────────────────────────────────────────────────────────┐
│ CPU 0 (Cortex-A76)  │ Input Thread + Async I/O                 │
│                     │ - V4L2 buffer management                 │
│                     │ - Frame acquisition                      │
│                     │ - Buffer handoff to preprocessing        │
│                     │ - Video writer thread (nice 19)          │
│                     │ - Display thread (nice 19, optional)     │
├─────────────────────┼──────────────────────────────────────────┤
│ CPU 1 (Cortex-A76)  │ Preprocessing Thread                     │
│                     │ - YUYV→RGB conversion                    │
│                     │ - Resize + Normalize                     │
│                     │ - Buffer handoff to inference            │
├─────────────────────┼──────────────────────────────────────────┤
│ CPU 2-3 (Cortex-A76)│ NCNN Worker Threads                      │
│                     │ - Model inference                        │
│                     │ - Post-processing                        │
│                     │ - Class filtering (optional)             │
└─────────────────────┴──────────────────────────────────────────┘

## Memory Layout

### Input Buffer (YUYV)
- Size: 640 × 480 × 2 = 614,400 bytes
- Alignment: 64 bytes (cache line)
- Triple-buffered for zero-copy pipeline

### RGB Intermediate Buffer  
- Size: 640 × 480 × 3 = 921,600 bytes
- Alignment: 64 bytes
- Double-buffered

### Model Input Tensor (FP16)
- Size: 640 × 640 × 3 × 2 = 2,457,600 bytes
- Layout: CHW (Channel, Height, Width)
- Alignment: 64 bytes
- Pre-allocated, reused per frame

### Detection Output
- Size: 8400 × 84 × 2 = 1,411,200 bytes (FP16)
- Pre-allocated output buffer

## Performance Targets

| Metric                  | Target    | Measurement Method              |
|------------------------|-----------|----------------------------------|
| End-to-end latency     | ≤50ms     | Camera capture to detection out  |
| Throughput             | ≥20 FPS   | Sustained over 1000 frames       |
| Frame jitter           | ≤5ms σ    | Standard deviation of frame time |
| Memory peak            | ≤512MB    | RSS measurement                  |
| Thermal stability      | No throttle| 10 min sustained run            |

## Build Configuration

### Compiler Flags (Cortex-A76 optimized)
```
-march=armv8.2-a+fp16+dotprod
-mtune=cortex-a76
-O3
-ffast-math
-fno-rtti
-fno-exceptions
-flto
-DNDEBUG
```

### NCNN Build Options
```
-DNCNN_VULKAN=OFF
-DNCNN_OPENMP=ON
-DNCNN_RUNTIME_CPU=OFF
-DNCNN_BUILD_TOOLS=OFF
-DNCNN_BUILD_EXAMPLES=OFF
-DNCNN_BUILD_BENCHMARK=OFF
-DNCNN_PIXEL=ON
-DNCNN_PIXEL_ROTATE=OFF
-DNCNN_PIXEL_AFFINE=OFF
-DNCNN_PIXEL_DRAWING=OFF
-DNCNN_BF16=OFF
-DNCNN_FORCE_INLINE=ON
```

## File Structure

```
/home/pi/AI/
├── ARCHITECTURE.md           # This document
├── god_mode_deploy.py        # Master automation script
├── CMakeLists.txt            # Build configuration
├── include/
│   ├── input_pipeline.h      # Unified input interface
│   ├── neon_preprocess.h     # NEON preprocessing declarations
│   ├── inference_engine.h    # NCNN wrapper interface
│   ├── postprocess.h         # Detection output processing
│   ├── benchmark.h           # Benchmark framework
│   └── common.h              # Shared types and utilities
├── src/
│   ├── main.cpp              # Entry point
│   ├── input_pipeline.cpp    # V4L2 + video file unified input
│   ├── neon_preprocess.cpp   # ARM NEON optimized preprocessing
│   ├── inference_engine.cpp  # NCNN integration
│   ├── postprocess.cpp       # NMS and output decoding
│   └── benchmark.cpp         # Performance measurement
├── models/
│   └── (generated by god_mode_deploy.py)
├── tests/
│   ├── test_video/           # Test video files
│   └── benchmark_results/    # Performance logs
└── scripts/
    └── analyze_benchmark.py  # Result analysis
```

## Agent Consensus Log

### Decision: Unified Buffer Layout
- Camera & Input Agent: Proposed YUYV as canonical input format
- NEON Agent: Confirmed 16-byte alignment sufficient for vectorization
- Chief Architect: Approved - maintains consistency between camera and video paths

### Decision: Thread Count = 3 for NCNN
- Inference Agent: 4 threads caused contention with input thread
- Benchmark Agent: 3 threads achieved better P99 latency
- Chief Architect: Approved - prioritizes latency stability over peak throughput

### Decision: FP16 Throughout
- NEON Agent: FP16 halves memory bandwidth, critical for cache efficiency
- Inference Agent: NCNN FP16 path well-optimized for A76
- Chief Architect: Approved - consistent precision avoids conversion overhead
