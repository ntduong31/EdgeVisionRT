/**
 * @file input_pipeline.h
 * @brief Unified input pipeline for V4L2 camera and video file sources
 * 
 * Camera & Input Systems Agent Design:
 * - Two input branches (V4L2 camera, video file) converge to single buffer layout
 * - V4L2 uses mmap zero-copy for minimal latency
 * - Video file path uses same buffer layout to ensure benchmark validity
 * - Downstream code sees identical FrameBuffer regardless of source
 */

#ifndef YOLO_INPUT_PIPELINE_H
#define YOLO_INPUT_PIPELINE_H

#include "common.h"
#include <string>
#include <atomic>
#include <functional>

namespace yolo {

// Forward declarations
class V4L2Camera;
class VideoFileReader;

/**
 * @brief Unified input interface abstracting camera and video file sources
 * 
 * Both V4L2 camera and video file provide frames through the same interface,
 * ensuring preprocessing and inference code paths are identical.
 */
class InputPipeline {
public:
    /**
     * @brief Configuration for input pipeline
     */
    struct Config {
        InputSource source;
        std::string device_path;    // "/dev/video0" for camera, or video file path
        int width = INPUT_WIDTH;
        int height = INPUT_HEIGHT;
        int fps = 30;
        int buffer_count = 4;       // Number of mmap buffers for V4L2
        bool loop_video = true;     // Loop video file for benchmark
    };

    /**
     * @brief Frame callback signature
     * @param frame Valid frame buffer with YUYV data
     * @return true to continue capture, false to stop
     */
    using FrameCallback = std::function<bool(const FrameBuffer& frame)>;

    InputPipeline();
    ~InputPipeline();
    
    // Non-copyable
    InputPipeline(const InputPipeline&) = delete;
    InputPipeline& operator=(const InputPipeline&) = delete;

    /**
     * @brief Initialize the input pipeline
     * @param config Input configuration
     * @return Error code
     */
    ErrorCode initialize(const Config& config);

    /**
     * @brief Start frame capture
     * @param callback Function called for each frame
     * @return Error code
     */
    ErrorCode start(FrameCallback callback);

    /**
     * @brief Stop frame capture
     */
    void stop();

    /**
     * @brief Check if pipeline is running
     */
    bool is_running() const { return running_.load(std::memory_order_relaxed); }

    /**
     * @brief Get current configuration
     */
    const Config& config() const { return config_; }

    /**
     * @brief Get total frames captured
     */
    uint64_t frame_count() const { return frame_count_.load(std::memory_order_relaxed); }

    /**
     * @brief Get dropped frame count (V4L2 only)
     */
    uint64_t dropped_frames() const { return dropped_frames_.load(std::memory_order_relaxed); }

    /**
     * @brief Get video file FPS (video file input only)
     */
    double get_video_fps() const { return video_fps_; }

private:
    ErrorCode init_v4l2();
    ErrorCode init_video_file();
    
    void capture_loop_v4l2(FrameCallback callback);
    void capture_loop_video(FrameCallback callback);

    Config config_;
    std::atomic<bool> running_{false};
    std::atomic<uint64_t> frame_count_{0};
    std::atomic<uint64_t> dropped_frames_{0};
    double video_fps_ = 25.0;  // Video file FPS

    // V4L2 specific
    int v4l2_fd_ = -1;
    struct V4L2Buffer {
        void* start;
        size_t length;
    };
    V4L2Buffer* v4l2_buffers_ = nullptr;
    int v4l2_buffer_count_ = 0;

    // Video file specific
    void* video_ctx_ = nullptr;  // FFmpeg context
    AlignedPtr<uint8_t> video_frame_buffer_;
};

// ============================================================================
// V4L2 Camera Implementation (defined in input_pipeline.cpp)
// ============================================================================

/**
 * @brief V4L2 helper for camera format negotiation
 */
struct V4L2Format {
    uint32_t pixel_format;  // V4L2_PIX_FMT_YUYV
    int width;
    int height;
    int fps_numerator;
    int fps_denominator;
    int bytesperline;
    int sizeimage;
};

/**
 * @brief Query camera capabilities
 * @param fd V4L2 file descriptor
 * @param format Output format structure
 * @return true if camera supports YUYV format
 */
bool v4l2_query_format(int fd, V4L2Format* format);

/**
 * @brief Set camera format
 * @param fd V4L2 file descriptor
 * @param width Desired width
 * @param height Desired height
 * @param fps Desired FPS
 * @return true on success
 */
bool v4l2_set_format(int fd, int width, int height, int fps);

}  // namespace yolo

#endif  // YOLO_INPUT_PIPELINE_H
