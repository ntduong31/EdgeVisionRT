/**
 * @file drm_display.h
 * @brief Direct Framebuffer display - bypasses X11 for zero contention
 * 
 * This provides direct framebuffer access without X11/Qt overhead.
 * NEON-optimized pixel blitting for maximum throughput.
 * 
 * Usage: Run from console (not X11) or use: sudo chvt 1 first
 */

#ifndef YOLO_DRM_DISPLAY_H
#define YOLO_DRM_DISPLAY_H

#include "common.h"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <atomic>
#include <algorithm>

// Linux headers
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <linux/fb.h>
#include <arm_neon.h>

namespace yolo {

/**
 * @brief Framebuffer display - zero X11 overhead
 * 
 * Writes directly to /dev/fb0 for minimal latency.
 * No Qt, no X11, just raw NEON-accelerated pixel pushing.
 */
class FramebufferDisplay {
public:
    struct Config {
        int target_width = 640;
        int target_height = 480;
        bool draw_fps = true;
        bool draw_bbox = true;
    };
    
    FramebufferDisplay() : fd_(-1), fb_ptr_(nullptr), fb_size_(0), 
                           running_(false), frames_displayed_(0) {}
    
    ~FramebufferDisplay() {
        stop();
    }
    
    bool start(const Config& config) {
        config_ = config;
        
        // Open framebuffer
        fd_ = open("/dev/fb0", O_RDWR);
        if (fd_ < 0) {
            perror("Cannot open /dev/fb0 (try: sudo chmod 666 /dev/fb0)");
            return false;
        }
        
        // Get framebuffer info
        struct fb_var_screeninfo vinfo;
        struct fb_fix_screeninfo finfo;
        
        if (ioctl(fd_, FBIOGET_VSCREENINFO, &vinfo) < 0) {
            perror("FBIOGET_VSCREENINFO");
            close(fd_);
            return false;
        }
        
        if (ioctl(fd_, FBIOGET_FSCREENINFO, &finfo) < 0) {
            perror("FBIOGET_FSCREENINFO");
            close(fd_);
            return false;
        }
        
        screen_width_ = vinfo.xres;
        screen_height_ = vinfo.yres;
        bits_per_pixel_ = vinfo.bits_per_pixel;
        line_length_ = finfo.line_length;
        
        fb_size_ = finfo.smem_len;
        
        // Map framebuffer to memory
        fb_ptr_ = static_cast<uint8_t*>(mmap(nullptr, fb_size_, 
                                             PROT_READ | PROT_WRITE, 
                                             MAP_SHARED, fd_, 0));
        
        if (fb_ptr_ == MAP_FAILED) {
            perror("mmap framebuffer");
            close(fd_);
            return false;
        }
        
        // Calculate display offset (top-left, not centered to avoid complex math)
        offset_x_ = 0;
        offset_y_ = 0;
        
        // Clamp target size to screen
        display_width_ = std::min(config_.target_width, screen_width_);
        display_height_ = std::min(config_.target_height, screen_height_);
        
        printf("Framebuffer: %dx%d @%dbpp (display: %dx%d)\n",
               screen_width_, screen_height_, bits_per_pixel_,
               display_width_, display_height_);
        
        running_ = true;
        return true;
    }
    
    void stop() {
        running_ = false;
        
        if (fb_ptr_ && fb_ptr_ != MAP_FAILED) {
            munmap(fb_ptr_, fb_size_);
            fb_ptr_ = nullptr;
        }
        
        if (fd_ >= 0) {
            close(fd_);
            fd_ = -1;
        }
    }
    
    // Push BGR frame directly - NEON optimized, minimal latency
    bool push_bgr(const uint8_t* bgr_data, int width, int height, int stride,
                  const DetectionResult& result, float fps = 0, float inference_ms = 0) {
        if (!running_ || !fb_ptr_) return false;
        
        // Direct blit BGR -> framebuffer
        if (bits_per_pixel_ == 32) {
            blit_bgr_to_bgra32_neon(bgr_data, width, height, stride);
        } else if (bits_per_pixel_ == 16) {
            blit_bgr_to_rgb565_neon(bgr_data, width, height, stride);
        }
        
        // Draw bounding boxes directly on framebuffer
        if (config_.draw_bbox && bits_per_pixel_ == 32) {
            draw_detections_fb(result, width, height);
        }
        
        // Draw FPS text (simple)
        if (config_.draw_fps && fps > 0 && bits_per_pixel_ == 32) {
            draw_fps_fb(fps, inference_ms);
        }
        
        frames_displayed_++;
        return true;
    }
    
    int frames_displayed() const { return frames_displayed_.load(); }
    bool is_running() const { return running_.load(); }

private:
    // NEON-optimized BGR to BGRA32 blit
    void blit_bgr_to_bgra32_neon(const uint8_t* bgr, int w, int h, int stride) {
        int out_w = std::min(w, display_width_);
        int out_h = std::min(h, display_height_);
        
        const uint8x16_t alpha = vdupq_n_u8(0xFF);
        
        for (int y = 0; y < out_h; y++) {
            const uint8_t* src = bgr + y * stride;
            uint8_t* dst = fb_ptr_ + (y + offset_y_) * line_length_ + offset_x_ * 4;
            
            int x = 0;
            // NEON: process 16 pixels at once
            for (; x + 16 <= out_w; x += 16) {
                // Load 48 bytes (16 BGR pixels)
                uint8x16x3_t bgr_pixels = vld3q_u8(src + x * 3);
                
                // Rearrange to BGRA (framebuffer typically expects BGRA or ARGB)
                uint8x16x4_t bgra;
                bgra.val[0] = bgr_pixels.val[0];  // B
                bgra.val[1] = bgr_pixels.val[1];  // G
                bgra.val[2] = bgr_pixels.val[2];  // R
                bgra.val[3] = alpha;               // A
                
                // Store 64 bytes (16 BGRA pixels)
                vst4q_u8(dst + x * 4, bgra);
            }
            
            // Scalar remainder
            for (; x < out_w; x++) {
                dst[x * 4 + 0] = src[x * 3 + 0];  // B
                dst[x * 4 + 1] = src[x * 3 + 1];  // G
                dst[x * 4 + 2] = src[x * 3 + 2];  // R
                dst[x * 4 + 3] = 0xFF;            // A
            }
        }
    }
    
    // NEON-optimized BGR to RGB565 blit
    void blit_bgr_to_rgb565_neon(const uint8_t* bgr, int w, int h, int stride) {
        int out_w = std::min(w, display_width_);
        int out_h = std::min(h, display_height_);
        
        for (int y = 0; y < out_h; y++) {
            const uint8_t* src = bgr + y * stride;
            uint16_t* dst = reinterpret_cast<uint16_t*>(
                fb_ptr_ + (y + offset_y_) * line_length_
            ) + offset_x_;
            
            int x = 0;
            // NEON: process 8 pixels at once
            for (; x + 8 <= out_w; x += 8) {
                uint8x8x3_t bgr_pixels = vld3_u8(src + x * 3);
                
                // Convert to RGB565
                uint16x8_t r = vshrq_n_u16(vmovl_u8(bgr_pixels.val[2]), 3);  // R >> 3
                uint16x8_t g = vshrq_n_u16(vmovl_u8(bgr_pixels.val[1]), 2);  // G >> 2
                uint16x8_t b = vshrq_n_u16(vmovl_u8(bgr_pixels.val[0]), 3);  // B >> 3
                
                // Pack: (R << 11) | (G << 5) | B
                uint16x8_t rgb565 = vorrq_u16(vorrq_u16(vshlq_n_u16(r, 11), vshlq_n_u16(g, 5)), b);
                
                vst1q_u16(dst + x, rgb565);
            }
            
            // Scalar remainder
            for (; x < out_w; x++) {
                uint8_t b = src[x * 3];
                uint8_t g = src[x * 3 + 1];
                uint8_t r = src[x * 3 + 2];
                dst[x] = ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3);
            }
        }
    }
    
    // Draw simple bounding box on framebuffer (BGRA32)
    void draw_box_fb(int x1, int y1, int x2, int y2, uint32_t color) {
        x1 = std::max(0, std::min(x1, display_width_ - 1));
        x2 = std::max(0, std::min(x2, display_width_ - 1));
        y1 = std::max(0, std::min(y1, display_height_ - 1));
        y2 = std::max(0, std::min(y2, display_height_ - 1));
        
        // Top and bottom edges
        for (int x = x1; x <= x2; x++) {
            *reinterpret_cast<uint32_t*>(fb_ptr_ + y1 * line_length_ + x * 4) = color;
            *reinterpret_cast<uint32_t*>(fb_ptr_ + y2 * line_length_ + x * 4) = color;
        }
        // Left and right edges
        for (int y = y1; y <= y2; y++) {
            *reinterpret_cast<uint32_t*>(fb_ptr_ + y * line_length_ + x1 * 4) = color;
            *reinterpret_cast<uint32_t*>(fb_ptr_ + y * line_length_ + x2 * 4) = color;
        }
    }
    
    void draw_detections_fb(const DetectionResult& result, int orig_w, int orig_h) {
        const uint32_t GREEN = 0xFF00FF00;  // ARGB green
        
        for (int i = 0; i < result.count; i++) {
            const auto& det = result.detections[i];
            
            // Scale to display size
            int x1 = static_cast<int>(det.x1 * display_width_ / orig_w);
            int y1 = static_cast<int>(det.y1 * display_height_ / orig_h);
            int x2 = static_cast<int>(det.x2 * display_width_ / orig_w);
            int y2 = static_cast<int>(det.y2 * display_height_ / orig_h);
            
            draw_box_fb(x1, y1, x2, y2, GREEN);
        }
    }
    
    void draw_fps_fb(float fps, float inference_ms) {
        // Simple: draw a colored rectangle in top-left corner
        // Green if FPS >= 30, Yellow if >= 20, Red otherwise
        uint32_t color;
        if (fps >= 30.0f) color = 0xFF00FF00;      // Green
        else if (fps >= 20.0f) color = 0xFFFFFF00; // Yellow
        else color = 0xFFFF0000;                    // Red
        
        // Draw 10x10 indicator box
        for (int y = 5; y < 15 && y < display_height_; y++) {
            for (int x = 5; x < 15 && x < display_width_; x++) {
                *reinterpret_cast<uint32_t*>(fb_ptr_ + y * line_length_ + x * 4) = color;
            }
        }
    }
    
    Config config_;
    int fd_;
    uint8_t* fb_ptr_;
    size_t fb_size_;
    int screen_width_, screen_height_;
    int display_width_, display_height_;
    int bits_per_pixel_;
    int line_length_;
    int offset_x_, offset_y_;
    
    std::atomic<bool> running_;
    std::atomic<int> frames_displayed_;
};

}  // namespace yolo

#endif // YOLO_DRM_DISPLAY_H
