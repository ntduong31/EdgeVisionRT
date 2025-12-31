/**
 * @file video_writer.h
 * @brief Async video writer with FFmpeg pipe - zero impact on inference FPS
 * 
 * Architecture:
 * - Producer (inference thread) pushes frames to lock-free queue
 * - Consumer (writer thread) encodes via FFmpeg pipe in background
 * - Pin writer to CPU 0, inference uses CPU 1-3
 * - Pre-allocated frame pool to avoid allocation during inference
 */

#ifndef YOLO_VIDEO_WRITER_H
#define YOLO_VIDEO_WRITER_H

#include "common.h"
#include "highgui_minimal.h"  // Minimal highgui for display
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <thread>
#include <atomic>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <iostream>
#include <cstdio>
#include <vector>
#include <unistd.h>

namespace yolo {

// ============================================================================
// COCO Class Names for Labels
// ============================================================================

static const char* COCO_NAMES[] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
};

// Color palette for different classes (BGR format)
static const cv::Scalar CLASS_COLORS[] = {
    cv::Scalar(255, 56, 56),   // Red
    cv::Scalar(255, 157, 151), // Light Red
    cv::Scalar(255, 112, 31),  // Orange
    cv::Scalar(255, 178, 29),  // Yellow-Orange
    cv::Scalar(207, 210, 49),  // Yellow-Green
    cv::Scalar(72, 249, 10),   // Green
    cv::Scalar(146, 204, 23),  // Light Green
    cv::Scalar(61, 219, 134),  // Teal
    cv::Scalar(26, 147, 52),   // Dark Green
    cv::Scalar(0, 212, 187),   // Cyan
    cv::Scalar(44, 153, 168),  // Dark Cyan
    cv::Scalar(0, 194, 255),   // Light Blue
    cv::Scalar(52, 69, 147),   // Dark Blue
    cv::Scalar(100, 115, 255), // Blue
    cv::Scalar(0, 24, 236),    // Bright Blue
    cv::Scalar(132, 56, 255),  // Purple
    cv::Scalar(82, 0, 133),    // Dark Purple
    cv::Scalar(203, 56, 255),  // Magenta
    cv::Scalar(255, 149, 200), // Pink
    cv::Scalar(255, 55, 199),  // Hot Pink
};

constexpr int NUM_COLORS = sizeof(CLASS_COLORS) / sizeof(CLASS_COLORS[0]);

// ============================================================================
// BBox Renderer - Optimized for minimal overhead
// ============================================================================

class BBoxRenderer {
public:
    // Draw bboxes directly on frame (in-place, minimal allocation)
    static void draw(cv::Mat& frame, const DetectionResult& result, 
                     int orig_width, int orig_height) {
        const float scale_x = static_cast<float>(frame.cols);
        const float scale_y = static_cast<float>(frame.rows);
        
        for (int i = 0; i < result.count; i++) {
            const Detection& det = result.detections[i];
            
            int x1 = static_cast<int>(det.x1 * scale_x);
            int y1 = static_cast<int>(det.y1 * scale_y);
            int x2 = static_cast<int>(det.x2 * scale_x);
            int y2 = static_cast<int>(det.y2 * scale_y);
            
            x1 = std::max(0, std::min(x1, frame.cols - 1));
            y1 = std::max(0, std::min(y1, frame.rows - 1));
            x2 = std::max(0, std::min(x2, frame.cols - 1));
            y2 = std::max(0, std::min(y2, frame.rows - 1));
            
            const cv::Scalar& color = CLASS_COLORS[det.class_id % NUM_COLORS];
            cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);
            
            char label[64];
            snprintf(label, sizeof(label), "%s %.0f%%", 
                     COCO_NAMES[det.class_id % NUM_CLASSES], 
                     det.confidence * 100);
            
            int baseline = 0;
            cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 
                                                   0.5, 1, &baseline);
            
            int label_y = std::max(y1 - label_size.height - 4, 0);
            cv::rectangle(frame, 
                         cv::Point(x1, label_y), 
                         cv::Point(x1 + label_size.width + 4, label_y + label_size.height + 4),
                         color, cv::FILLED);
            
            cv::putText(frame, label, 
                       cv::Point(x1 + 2, label_y + label_size.height + 2),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        }
    }
    
    static void draw_fps(cv::Mat& frame, float fps, float inference_ms) {
        char text[64];
        snprintf(text, sizeof(text), "FPS: %.1f | Inf: %.1fms", fps, inference_ms);
        
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 
                                             0.7, 2, &baseline);
        cv::rectangle(frame, 
                     cv::Point(10, 10), 
                     cv::Point(20 + text_size.width, 20 + text_size.height),
                     cv::Scalar(0, 0, 0), cv::FILLED);
        
        cv::putText(frame, text, cv::Point(15, 15 + text_size.height),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    }
};

// ============================================================================
// Frame Pool - Pre-allocated frames to avoid allocation during inference
// ============================================================================

class FramePool {
public:
    FramePool(int pool_size, int width, int height) : width_(width), height_(height) {
        frames_.resize(pool_size);
        for (int i = 0; i < pool_size; i++) {
            frames_[i] = cv::Mat(height, width, CV_8UC3);
            free_indices_.push(i);
        }
    }
    
    int acquire() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (free_indices_.empty()) return -1;
        int idx = free_indices_.front();
        free_indices_.pop();
        return idx;
    }
    
    void release(int idx) {
        std::lock_guard<std::mutex> lock(mutex_);
        free_indices_.push(idx);
    }
    
    cv::Mat& get(int idx) { return frames_[idx]; }
    
private:
    std::vector<cv::Mat> frames_;
    std::queue<int> free_indices_;
    std::mutex mutex_;
    int width_, height_;
};

// ============================================================================
// Frame Data (for queue)
// ============================================================================

struct FrameData {
    int frame_idx;
    cv::Mat frame;
    DetectionResult detections;
    int original_width;
    int original_height;
    float fps;
    float inference_ms;
    
    FrameData() : frame_idx(-1), original_width(0), original_height(0), fps(0), inference_ms(0) {}
};

// ============================================================================
// Async Video Writer - FFmpeg pipe with CPU isolation
// ============================================================================

class AsyncVideoWriter {
public:
    struct Config {
        std::string output_path;
        int width;
        int height;
        double fps = 25.0;
        int queue_size = 30;        // Reduced to avoid memory pressure
        bool draw_fps = true;
        int writer_cpu = 0;
        bool use_ffmpeg = true;
    };
    
    AsyncVideoWriter() : running_(false), frames_written_(0), frames_dropped_(0), 
                         ffmpeg_pipe_(nullptr) {}
    
    ~AsyncVideoWriter() {
        stop();
    }
    
    bool start(const Config& config) {
        config_ = config;
        
        // No frame pool - use simple clone to avoid memory pressure
        // Pre-allocating 30+ frames causes cache thrashing on RPi5
        
        // Use MJPEG for temp file - has proper timestamps and low CPU overhead
        // Convert to MP4 (H.264) after processing is done
        actual_output_path_ = config.output_path;
        
        // If MP4 requested, write to temp AVI first
        if (config.output_path.find(".mp4") != std::string::npos) {
            temp_avi_path_ = config.output_path.substr(0, config.output_path.rfind('.')) + "_temp.avi";
            use_temp_avi_ = true;
        } else {
            temp_avi_path_ = config.output_path;
            use_temp_avi_ = false;
        }
        
        // Use MJPEG codec - proper timestamps, fast encoding, good quality
        int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
        cv_writer_.open(temp_avi_path_, fourcc, config.fps,
                       cv::Size(config.width, config.height), true);
        
        if (!cv_writer_.isOpened()) {
            std::cerr << "Failed to open OpenCV video writer\n";
            return false;
        }
        
        use_ffmpeg_ = false;
        running_ = true;
        writer_thread_ = std::thread(&AsyncVideoWriter::writer_loop, this);
        
        return true;
    }
    
    void stop() {
        if (!running_) return;
        
        running_ = false;
        cv_.notify_all();
        
        if (writer_thread_.joinable()) {
            writer_thread_.join();
        }
        
        // Flush remaining frames
        while (!queue_.empty()) {
            write_frame(queue_.front());
            queue_.pop();
        }
        
        cv_writer_.release();
        
        // Convert temp AVI to MP4 if needed
        if (use_temp_avi_ && frames_written_ > 0) {
            std::cout << "Converting to MP4 (H.264)...\n";
            char cmd[1024];
            // Use -vsync cfr to ensure constant frame rate for smooth playback
            // Use -r to force exact output frame rate matching input
            snprintf(cmd, sizeof(cmd),
                "ffmpeg -y -r %.3f -i \"%s\" -c:v libx264 -preset fast -crf 23 "
                "-pix_fmt yuv420p -r %.3f -vsync cfr -movflags +faststart \"%s\" 2>/dev/null && rm -f \"%s\"",
                config_.fps, temp_avi_path_.c_str(), config_.fps, actual_output_path_.c_str(), temp_avi_path_.c_str());
            int ret = system(cmd);
            if (ret == 0) {
                std::cout << "MP4 conversion complete: " << actual_output_path_ << "\n";
            } else {
                std::cerr << "MP4 conversion failed, keeping AVI: " << temp_avi_path_ << "\n";
            }
        }
    }
    
    bool push(const cv::Mat& frame, const DetectionResult& result, 
              int orig_width, int orig_height, float fps = 0, float inference_ms = 0) {
        
        FrameData fd;
        fd.frame = frame.clone();  // Simple clone, no pool
        fd.frame_idx = -1;
        fd.detections = result;
        fd.original_width = orig_width;
        fd.original_height = orig_height;
        fd.fps = fps;
        fd.inference_ms = inference_ms;
        
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (queue_.size() >= static_cast<size_t>(config_.queue_size)) {
                frames_dropped_++;
                return false;
            }
            queue_.push(std::move(fd));
        }
        cv_.notify_one();
        
        return true;
    }
    
    int frames_written() const { return frames_written_.load(); }
    int frames_dropped() const { return frames_dropped_.load(); }
    int queue_size() const {
        std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(mutex_));
        return queue_.size();
    }

private:
    void writer_loop() {
        // Pin writer thread to CPU 0 (isolated from inference on CPU 1-3)
        set_thread_affinity(config_.writer_cpu);
        
        // Set lowest priority
        struct sched_param param;
        param.sched_priority = 0;
        pthread_setschedparam(pthread_self(), SCHED_OTHER, &param);
        
        // Set nice value to lowest priority
        nice(19);
        
        while (running_ || !queue_.empty()) {
            FrameData fd;
            
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait_for(lock, std::chrono::milliseconds(50), [this] {
                    return !queue_.empty() || !running_;
                });
                
                if (queue_.empty()) continue;
                
                fd = std::move(queue_.front());
                queue_.pop();
            }
            
            write_frame(fd);
            // No pool to release
        }
    }
    
    void write_frame(FrameData& fd) {
        cv::Mat& frame = fd.frame;
        
        if (frame.empty()) return;
        
        cv::Mat output_frame;
        if (frame.cols != config_.width || frame.rows != config_.height) {
            cv::resize(frame, output_frame, cv::Size(config_.width, config_.height));
        } else {
            output_frame = frame;
        }
        
        BBoxRenderer::draw(output_frame, fd.detections, 
                          fd.original_width, fd.original_height);
        
        if (config_.draw_fps && fd.fps > 0) {
            BBoxRenderer::draw_fps(output_frame, fd.fps, fd.inference_ms);
        }
        
        if (use_ffmpeg_ && ffmpeg_pipe_) {
            fwrite(output_frame.data, 1, output_frame.total() * output_frame.elemSize(), ffmpeg_pipe_);
        } else {
            cv_writer_.write(output_frame);
        }
        
        frames_written_++;
    }
    
    Config config_;
    bool use_ffmpeg_ = false;
    FILE* ffmpeg_pipe_;
    cv::VideoWriter cv_writer_;
    // No frame pool - simple clone is faster for small queue
    
    std::string temp_avi_path_;
    std::string actual_output_path_;
    bool use_temp_avi_ = false;
    
    std::thread writer_thread_;
    std::atomic<bool> running_;
    std::atomic<int> frames_written_;
    std::atomic<int> frames_dropped_;
    
    std::queue<FrameData> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
};

// ============================================================================
// Async Display - Zero FPS impact with smooth rendering
// ============================================================================

class AsyncDisplay {
public:
    struct Config {
        std::string window_name = "YOLOv8n Detection";
        int queue_size = 5;         // Small queue for low latency
        bool draw_fps = true;
        bool draw_bbox = true;
        int display_cpu = 0;        // Pin to CPU 0
        float max_screen_ratio = 0.8f;  // Max 80% of screen size
    };
    
    AsyncDisplay() : running_(false), frames_displayed_(0), frames_dropped_(0),
                     display_width_(640), display_height_(480) {}
    
    ~AsyncDisplay() {
        stop();
    }
    
    bool start(const Config& config, int frame_width, int frame_height) {
        config_ = config;
        frame_width_ = frame_width;
        frame_height_ = frame_height;
        
        // Set DISPLAY environment variable
        setenv("DISPLAY", ":0", 1);
        
        // Suppress Qt timer warnings (harmless, caused by multi-threaded highgui)
        setenv("QT_LOGGING_RULES", "qt.qpa.*=false", 0);
        
        // Detect screen size and calculate display size
        calculate_display_size();
        
        running_ = true;
        display_thread_ = std::thread(&AsyncDisplay::display_loop, this);
        
        return true;
    }
    
    void stop() {
        if (!running_) return;
        
        running_ = false;
        cv_.notify_all();
        
        if (display_thread_.joinable()) {
            display_thread_.join();
        }
        // Window cleanup handled in display_loop()
    }
    
    bool push(const cv::Mat& frame, const DetectionResult& result,
              int orig_width, int orig_height, float fps = 0, float inference_ms = 0) {
        
        FrameData fd;
        fd.frame = frame.clone();
        fd.detections = result;
        fd.original_width = orig_width;
        fd.original_height = orig_height;
        fd.fps = fps;
        fd.inference_ms = inference_ms;
        
        {
            std::lock_guard<std::mutex> lock(mutex_);
            // Drop oldest frame if queue is full (keep latest for smooth display)
            while (queue_.size() >= static_cast<size_t>(config_.queue_size)) {
                queue_.pop();
                frames_dropped_++;
            }
            queue_.push(std::move(fd));
        }
        cv_.notify_one();
        
        return true;
    }
    
    int frames_displayed() const { return frames_displayed_.load(); }
    int frames_dropped() const { return frames_dropped_.load(); }
    int display_width() const { return display_width_; }
    int display_height() const { return display_height_; }

private:
    void calculate_display_size() {
        // Try to get screen size from X11
        int screen_width = 1920;
        int screen_height = 1080;
        
        // Try xrandr to get actual screen resolution
        FILE* pipe = popen("xrandr 2>/dev/null | grep '\\*' | head -1 | awk '{print $1}'", "r");
        if (pipe) {
            char buffer[128];
            if (fgets(buffer, sizeof(buffer), pipe)) {
                int w, h;
                if (sscanf(buffer, "%dx%d", &w, &h) == 2) {
                    screen_width = w;
                    screen_height = h;
                }
            }
            pclose(pipe);
        }
        
        // Calculate display size (max 80% of screen, maintain aspect ratio)
        float max_width = screen_width * config_.max_screen_ratio;
        float max_height = screen_height * config_.max_screen_ratio;
        
        float scale = std::min(max_width / frame_width_, max_height / frame_height_);
        scale = std::min(scale, 1.0f);  // Don't upscale
        
        display_width_ = static_cast<int>(frame_width_ * scale);
        display_height_ = static_cast<int>(frame_height_ * scale);
        
        // Calculate window position (center of screen)
        window_x_ = (screen_width - display_width_) / 2;
        window_y_ = (screen_height - display_height_) / 2;
        
        std::cout << "Display: " << display_width_ << "x" << display_height_ 
                  << " (screen: " << screen_width << "x" << screen_height << ")\n";
    }
    
    void display_loop() {
        // Pin to CPU 0 (isolated from inference)
        set_thread_affinity(config_.display_cpu);
        
        // Low priority
        nice(10);
        
        // Create window
        cv::namedWindow(config_.window_name, cv::WINDOW_NORMAL);
        cv::resizeWindow(config_.window_name, display_width_, display_height_);
        cv::moveWindow(config_.window_name, window_x_, window_y_);
        
        cv::Mat display_frame;
        
        while (running_) {
            FrameData fd;
            
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait_for(lock, std::chrono::milliseconds(30), [this] {
                    return !queue_.empty() || !running_;
                });
                
                if (queue_.empty()) {
                    if (!running_) break;
                    continue;
                }
                
                // Get latest frame (skip old ones for smooth display)
                while (queue_.size() > 1) {
                    queue_.pop();
                    frames_dropped_++;
                }
                
                fd = std::move(queue_.front());
                queue_.pop();
            }
            
            if (fd.frame.empty()) continue;
            
            // Resize for display
            if (fd.frame.cols != display_width_ || fd.frame.rows != display_height_) {
                cv::resize(fd.frame, display_frame, cv::Size(display_width_, display_height_));
            } else {
                display_frame = fd.frame;
            }
            
            // Draw bboxes
            if (config_.draw_bbox) {
                BBoxRenderer::draw(display_frame, fd.detections,
                                  fd.original_width, fd.original_height);
            }
            
            // Draw FPS overlay
            if (config_.draw_fps && fd.fps > 0) {
                BBoxRenderer::draw_fps(display_frame, fd.fps, fd.inference_ms);
            }
            
            // Show frame
            cv::imshow(config_.window_name, display_frame);
            
            // Handle key events (non-blocking)
            int key = cv::waitKey(1);
            if (key == 27 || key == 'q' || key == 'Q') {  // ESC or Q to quit
                running_ = false;
                break;
            }
            
            frames_displayed_++;
        }
        
        // Clean up window - call waitKey to process final events
        cv::waitKey(1);
        cv::destroyWindow(config_.window_name);
        cv::waitKey(1);  // Process destroy event
    }
    
    Config config_;
    int frame_width_, frame_height_;
    int display_width_, display_height_;
    int window_x_, window_y_;
    
    std::thread display_thread_;
    std::atomic<bool> running_;
    std::atomic<int> frames_displayed_;
    std::atomic<int> frames_dropped_;
    
    std::queue<FrameData> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
};

}  // namespace yolo

#endif  // YOLO_VIDEO_WRITER_H
