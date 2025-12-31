/**
 * @file input_pipeline.cpp
 * @brief Unified input pipeline for V4L2 camera and video files
 * 
 * Camera & Input Systems Agent Implementation:
 * - V4L2 mmap zero-copy capture
 * - Video file decoding via FFmpeg/OpenCV
 * - Both paths produce identical FrameBuffer output
 */

#include "input_pipeline.h"

#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>
#include <errno.h>
#include <cstring>
#include <thread>

// OpenCV for video file reading (simpler than raw FFmpeg)
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>

namespace yolo {

// ============================================================================
// V4L2 Helper Functions
// ============================================================================

static int xioctl(int fd, unsigned long request, void* arg) {
    int r;
    do {
        r = ioctl(fd, request, arg);
    } while (r == -1 && errno == EINTR);
    return r;
}

bool v4l2_query_format(int fd, V4L2Format* format) {
    struct v4l2_capability cap;
    if (xioctl(fd, VIDIOC_QUERYCAP, &cap) == -1) {
        return false;
    }
    
    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
        return false;
    }
    
    if (!(cap.capabilities & V4L2_CAP_STREAMING)) {
        return false;
    }
    
    // Query current format
    struct v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    
    if (xioctl(fd, VIDIOC_G_FMT, &fmt) == -1) {
        return false;
    }
    
    format->pixel_format = fmt.fmt.pix.pixelformat;
    format->width = fmt.fmt.pix.width;
    format->height = fmt.fmt.pix.height;
    format->bytesperline = fmt.fmt.pix.bytesperline;
    format->sizeimage = fmt.fmt.pix.sizeimage;
    
    return true;
}

bool v4l2_set_format(int fd, int width, int height, int fps) {
    // Set pixel format
    struct v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = width;
    fmt.fmt.pix.height = height;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;
    
    if (xioctl(fd, VIDIOC_S_FMT, &fmt) == -1) {
        return false;
    }
    
    // Verify format was set
    if (fmt.fmt.pix.pixelformat != V4L2_PIX_FMT_YUYV) {
        return false;
    }
    
    // Set frame rate
    struct v4l2_streamparm parm;
    memset(&parm, 0, sizeof(parm));
    parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    parm.parm.capture.timeperframe.numerator = 1;
    parm.parm.capture.timeperframe.denominator = fps;
    
    xioctl(fd, VIDIOC_S_PARM, &parm);  // Ignore error, some cameras don't support
    
    return true;
}

// ============================================================================
// InputPipeline Implementation
// ============================================================================

InputPipeline::InputPipeline() = default;

InputPipeline::~InputPipeline() {
    stop();
    
    // Cleanup V4L2
    if (v4l2_buffers_) {
        for (int i = 0; i < v4l2_buffer_count_; i++) {
            if (v4l2_buffers_[i].start != MAP_FAILED) {
                munmap(v4l2_buffers_[i].start, v4l2_buffers_[i].length);
            }
        }
        delete[] v4l2_buffers_;
    }
    
    if (v4l2_fd_ >= 0) {
        close(v4l2_fd_);
    }
}

ErrorCode InputPipeline::initialize(const Config& config) {
    config_ = config;
    
    if (config.source == InputSource::CAMERA_V4L2) {
        return init_v4l2();
    } else {
        return init_video_file();
    }
}

ErrorCode InputPipeline::init_v4l2() {
    // Open device
    v4l2_fd_ = open(config_.device_path.c_str(), O_RDWR | O_NONBLOCK);
    if (v4l2_fd_ < 0) {
        return ErrorCode::CAMERA_OPEN_FAILED;
    }
    
    // Set format
    if (!v4l2_set_format(v4l2_fd_, config_.width, config_.height, config_.fps)) {
        close(v4l2_fd_);
        v4l2_fd_ = -1;
        return ErrorCode::CAMERA_FORMAT_FAILED;
    }
    
    // Request buffers
    struct v4l2_requestbuffers req;
    memset(&req, 0, sizeof(req));
    req.count = config_.buffer_count;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    
    if (xioctl(v4l2_fd_, VIDIOC_REQBUFS, &req) == -1) {
        close(v4l2_fd_);
        v4l2_fd_ = -1;
        return ErrorCode::CAMERA_MMAP_FAILED;
    }
    
    v4l2_buffer_count_ = req.count;
    v4l2_buffers_ = new V4L2Buffer[v4l2_buffer_count_];
    
    // Map buffers
    for (int i = 0; i < v4l2_buffer_count_; i++) {
        struct v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        
        if (xioctl(v4l2_fd_, VIDIOC_QUERYBUF, &buf) == -1) {
            return ErrorCode::CAMERA_MMAP_FAILED;
        }
        
        v4l2_buffers_[i].length = buf.length;
        v4l2_buffers_[i].start = mmap(
            nullptr, buf.length,
            PROT_READ | PROT_WRITE,
            MAP_SHARED,
            v4l2_fd_, buf.m.offset
        );
        
        if (v4l2_buffers_[i].start == MAP_FAILED) {
            return ErrorCode::CAMERA_MMAP_FAILED;
        }
    }
    
    return ErrorCode::SUCCESS;
}

ErrorCode InputPipeline::init_video_file() {
    // Allocate frame buffer for video decoding
    video_frame_buffer_ = make_aligned_buffer<uint8_t>(YUYV_BUFFER_SIZE);
    if (!video_frame_buffer_) {
        return ErrorCode::MEMORY_ALLOCATION_FAILED;
    }
    
    // OpenCV will be initialized in capture loop
    return ErrorCode::SUCCESS;
}

ErrorCode InputPipeline::start(FrameCallback callback) {
    if (running_.load()) {
        return ErrorCode::SUCCESS;
    }
    
    running_.store(true);
    
    if (config_.source == InputSource::CAMERA_V4L2) {
        // Queue all buffers
        for (int i = 0; i < v4l2_buffer_count_; i++) {
            struct v4l2_buffer buf;
            memset(&buf, 0, sizeof(buf));
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;
            buf.index = i;
            
            if (xioctl(v4l2_fd_, VIDIOC_QBUF, &buf) == -1) {
                running_.store(false);
                return ErrorCode::CAMERA_MMAP_FAILED;
            }
        }
        
        // Start streaming
        enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (xioctl(v4l2_fd_, VIDIOC_STREAMON, &type) == -1) {
            running_.store(false);
            return ErrorCode::CAMERA_MMAP_FAILED;
        }
        
        // Run capture loop in current thread
        capture_loop_v4l2(callback);
    } else {
        capture_loop_video(callback);
    }
    
    return ErrorCode::SUCCESS;
}

void InputPipeline::stop() {
    running_.store(false);
    
    if (config_.source == InputSource::CAMERA_V4L2 && v4l2_fd_ >= 0) {
        enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        xioctl(v4l2_fd_, VIDIOC_STREAMOFF, &type);
    }
}

void InputPipeline::capture_loop_v4l2(FrameCallback callback) {
    // Set thread affinity to CPU 0
    set_thread_affinity(INPUT_THREAD_CPU);
    
    fd_set fds;
    struct timeval tv;
    
    while (running_.load(std::memory_order_relaxed)) {
        FD_ZERO(&fds);
        FD_SET(v4l2_fd_, &fds);
        
        tv.tv_sec = 0;
        tv.tv_usec = 100000;  // 100ms timeout
        
        int r = select(v4l2_fd_ + 1, &fds, nullptr, nullptr, &tv);
        
        if (r == -1) {
            if (errno == EINTR) continue;
            break;
        }
        
        if (r == 0) {
            // Timeout
            continue;
        }
        
        // Dequeue buffer
        struct v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        
        if (xioctl(v4l2_fd_, VIDIOC_DQBUF, &buf) == -1) {
            if (errno == EAGAIN) continue;
            break;
        }
        
        // Create frame buffer
        FrameBuffer frame;
        frame.data = static_cast<uint8_t*>(v4l2_buffers_[buf.index].start);
        frame.size = buf.bytesused;
        frame.stride = config_.width * 2;
        frame.width = config_.width;
        frame.height = config_.height;
        frame.timestamp_ns = buf.timestamp.tv_sec * 1000000000LL + buf.timestamp.tv_usec * 1000LL;
        frame.frame_index = frame_count_.load();
        frame.valid = true;
        frame.format = PixelFormat::YUYV;  // Camera uses YUYV
        
        // Invoke callback
        bool continue_capture = callback(frame);
        frame_count_.fetch_add(1);
        
        // Re-queue buffer
        if (xioctl(v4l2_fd_, VIDIOC_QBUF, &buf) == -1) {
            break;
        }
        
        if (!continue_capture) {
            break;
        }
    }
    
    running_.store(false);
}

void InputPipeline::capture_loop_video(FrameCallback callback) {
    // Set thread affinity to CPU 0
    set_thread_affinity(INPUT_THREAD_CPU);
    
    cv::VideoCapture cap(config_.device_path);
    
    if (!cap.isOpened()) {
        running_.store(false);
        return;
    }
    
    // Get video properties
    int video_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int video_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double video_fps = cap.get(cv::CAP_PROP_FPS);
    
    // Store FPS for external access
    if (video_fps > 0 && video_fps < 1000) {
        const_cast<InputPipeline*>(this)->video_fps_ = video_fps;
    }
    
    cv::Mat bgr_frame;

    while (running_.load(std::memory_order_relaxed)) {
        if (!cap.read(bgr_frame)) {
            if (config_.loop_video) {
                cap.set(cv::CAP_PROP_POS_FRAMES, 0);
                continue;
            }
            break;
        }
        
        // Resize if necessary (but keep BGR format!)
        if (bgr_frame.cols != config_.width || bgr_frame.rows != config_.height) {
            cv::resize(bgr_frame, bgr_frame, cv::Size(config_.width, config_.height));
        }
        
        // OPTIMIZATION: Pass BGR directly - NO YUYV conversion!
        // Create frame buffer pointing to BGR data
        FrameBuffer frame;
        frame.data = bgr_frame.data;
        frame.size = bgr_frame.total() * bgr_frame.elemSize();
        frame.stride = bgr_frame.step[0];
        frame.width = config_.width;
        frame.height = config_.height;
        frame.timestamp_ns = get_timestamp_ns();
        frame.frame_index = frame_count_.load();
        frame.valid = true;
        frame.format = PixelFormat::BGR;  // Mark as BGR format
        
        // Invoke callback
        bool continue_capture = callback(frame);
        frame_count_.fetch_add(1);
        
        if (!continue_capture) {
            break;
        }
    }
    
    running_.store(false);
}

}  // namespace yolo
