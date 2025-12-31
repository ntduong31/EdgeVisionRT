/**
 * @file common.h
 * @brief Shared types, constants, and utilities for YOLOv8n inference system
 * 
 * Architecture Decision: All buffers use 64-byte alignment for cache line optimization
 * on Cortex-A76. Memory layout is designed for zero-copy pipeline where possible.
 */

#ifndef YOLO_COMMON_H
#define YOLO_COMMON_H

#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <chrono>
#include <arm_neon.h>

namespace yolo {

// ============================================================================
// Build Configuration Constants
// ============================================================================

constexpr int INPUT_WIDTH = 640;
constexpr int INPUT_HEIGHT = 480;
constexpr int MODEL_SIZE = 416;  // Reduced from 640 for better FPS (was 480)
constexpr int NUM_CLASSES = 80;  // COCO dataset
constexpr int NUM_OUTPUTS = 3549;  // YOLOv8 output anchors at 416x416
constexpr float CONF_THRESHOLD = 0.25f;
constexpr float NMS_THRESHOLD = 0.45f;
constexpr int MAX_DETECTIONS = 100;

// Memory alignment for cache line optimization (Cortex-A76 L1 cache line = 64 bytes)
constexpr size_t CACHE_LINE_SIZE = 64;

// Buffer sizes
constexpr size_t YUYV_BUFFER_SIZE = INPUT_WIDTH * INPUT_HEIGHT * 2;  // 614,400 bytes
constexpr size_t RGB_BUFFER_SIZE = INPUT_WIDTH * INPUT_HEIGHT * 3;   // 921,600 bytes
constexpr size_t MODEL_INPUT_SIZE = MODEL_SIZE * MODEL_SIZE * 3;     // 1,228,800 elements
constexpr size_t MODEL_INPUT_BYTES_FP16 = MODEL_INPUT_SIZE * 2;      // 2,457,600 bytes

// Thread configuration - OPTIMIZED for minimal cache contention
constexpr int INPUT_THREAD_CPU = 0;
constexpr int PREPROCESS_THREAD_CPU = 1;
constexpr int NCNN_THREAD_START = 1;
constexpr int NCNN_NUM_THREADS = 4;  // Use all 4 cores for maximum throughput

// Model input size in floats (FP32)
constexpr size_t MODEL_INPUT_FLOATS = MODEL_SIZE * MODEL_SIZE * 3;

// ============================================================================
// Aligned Memory Allocator
// ============================================================================

inline void* aligned_alloc_64(size_t size) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, CACHE_LINE_SIZE, size) != 0) {
        return nullptr;
    }
    return ptr;
}

inline void aligned_free(void* ptr) {
    free(ptr);
}

template<typename T>
struct AlignedDeleter {
    void operator()(T* ptr) const {
        aligned_free(ptr);
    }
};

template<typename T>
using AlignedPtr = std::unique_ptr<T[], AlignedDeleter<T>>;

template<typename T>
AlignedPtr<T> make_aligned_buffer(size_t count) {
    T* ptr = static_cast<T*>(aligned_alloc_64(count * sizeof(T)));
    return AlignedPtr<T>(ptr);
}

// ============================================================================
// Input Source Type
// ============================================================================

enum class InputSource {
    CAMERA_V4L2,    // Real camera with V4L2 mmap zero-copy
    VIDEO_FILE      // Video file for deterministic testing
};

// ============================================================================
// Frame Buffer Structure
// ============================================================================

enum class PixelFormat {
    YUYV,       // Camera format
    BGR,        // OpenCV format (video files)
    RGB         // Intermediate format
};

struct alignas(CACHE_LINE_SIZE) FrameBuffer {
    uint8_t* data;          // Pointer to pixel data
    size_t size;            // Buffer size in bytes
    size_t stride;          // Bytes per row
    int width;
    int height;
    int64_t timestamp_ns;   // Capture timestamp in nanoseconds
    uint32_t frame_index;   // Sequential frame number
    bool valid;             // Buffer contains valid data
    PixelFormat format;     // Pixel format
    
    FrameBuffer() : data(nullptr), size(0), stride(0), width(0), height(0),
                   timestamp_ns(0), frame_index(0), valid(false), format(PixelFormat::YUYV) {}
};

// ============================================================================
// Detection Result Structure
// ============================================================================

struct alignas(16) Detection {
    float x1, y1, x2, y2;   // Bounding box coordinates (normalized 0-1)
    float confidence;        // Detection confidence
    int class_id;           // Class index (0-79 for COCO)
    
    float width() const { return x2 - x1; }
    float height() const { return y2 - y1; }
    float area() const { return width() * height(); }
    float center_x() const { return (x1 + x2) * 0.5f; }
    float center_y() const { return (y1 + y2) * 0.5f; }
};

struct DetectionResult {
    Detection detections[MAX_DETECTIONS];
    int count;
    int64_t inference_time_us;  // Inference duration in microseconds
    int64_t preprocess_time_us; // Preprocessing duration
    int64_t postprocess_time_us;// Post-processing duration
    uint32_t frame_index;       // Source frame index
};

// ============================================================================
// High-Resolution Timer
// ============================================================================

class ScopedTimer {
public:
    using Clock = std::chrono::high_resolution_clock;
    
    explicit ScopedTimer(int64_t& output_us) 
        : output_(output_us), start_(Clock::now()) {}
    
    ~ScopedTimer() {
        auto end = Clock::now();
        output_ = std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count();
    }
    
    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

private:
    int64_t& output_;
    Clock::time_point start_;
};

inline int64_t get_timestamp_ns() {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        now.time_since_epoch()
    ).count();
}

// ============================================================================
// CPU Affinity Helper
// ============================================================================

#include <sched.h>
#include <pthread.h>

inline bool set_thread_affinity(int cpu_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);
    return pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) == 0;
}

inline bool set_thread_priority(int priority) {
    struct sched_param param;
    param.sched_priority = priority;
    return pthread_setschedparam(pthread_self(), SCHED_FIFO, &param) == 0;
}

// ============================================================================
// Error Handling
// ============================================================================

enum class ErrorCode {
    SUCCESS = 0,
    CAMERA_OPEN_FAILED,
    CAMERA_FORMAT_FAILED,
    CAMERA_MMAP_FAILED,
    VIDEO_OPEN_FAILED,
    VIDEO_DECODE_FAILED,
    MODEL_LOAD_FAILED,
    INFERENCE_FAILED,
    MEMORY_ALLOCATION_FAILED,
    INVALID_PARAMETER
};

inline const char* error_to_string(ErrorCode code) {
    switch (code) {
        case ErrorCode::SUCCESS: return "Success";
        case ErrorCode::CAMERA_OPEN_FAILED: return "Failed to open camera device";
        case ErrorCode::CAMERA_FORMAT_FAILED: return "Failed to set camera format";
        case ErrorCode::CAMERA_MMAP_FAILED: return "Failed to mmap camera buffers";
        case ErrorCode::VIDEO_OPEN_FAILED: return "Failed to open video file";
        case ErrorCode::VIDEO_DECODE_FAILED: return "Failed to decode video frame";
        case ErrorCode::MODEL_LOAD_FAILED: return "Failed to load NCNN model";
        case ErrorCode::INFERENCE_FAILED: return "Inference execution failed";
        case ErrorCode::MEMORY_ALLOCATION_FAILED: return "Memory allocation failed";
        case ErrorCode::INVALID_PARAMETER: return "Invalid parameter";
        default: return "Unknown error";
    }
}

// ============================================================================
// NEON Utility Macros
// ============================================================================

// Prefetch data into L1 cache (read)
#define PREFETCH_L1(addr) __builtin_prefetch((addr), 0, 3)

// Prefetch data into L2 cache (read)  
#define PREFETCH_L2(addr) __builtin_prefetch((addr), 0, 2)

// Prefetch for write
#define PREFETCH_W(addr) __builtin_prefetch((addr), 1, 3)

// Force inline for critical path functions
#define FORCE_INLINE __attribute__((always_inline)) inline

// Likely/unlikely branch hints
#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)

}  // namespace yolo

#endif  // YOLO_COMMON_H
