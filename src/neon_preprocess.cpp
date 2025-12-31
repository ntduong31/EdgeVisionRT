/**
 * @file neon_preprocess.cpp
 * @brief ARM NEON optimized preprocessing implementation - RADICAL OPTIMIZATION
 * 
 * CRITICAL CHANGES for 20+ FPS:
 * 1. Pre-allocated static buffers - NO malloc in hot path
 * 2. Direct BGR path for video files - bypass YUYV conversion
 * 3. Output FP32 directly - no FP16 intermediate
 * 4. Fused letterbox + normalize where possible
 * 5. Aggressive prefetching and cache optimization
 */

#include "neon_preprocess.h"
#include <cmath>
#include <algorithm>
#include <atomic>

namespace yolo {
namespace neon {

// ============================================================================
// Static Pre-allocated Buffers (NO MALLOC IN HOT PATH)
// ============================================================================

static uint8_t* g_rgb_buffer = nullptr;        // 640*480*3 = 921KB
static uint8_t* g_letterbox_buffer = nullptr;  // 640*640*3 = 1.2MB
static uint8_t* g_resize_temp = nullptr;       // For letterbox resize
static std::atomic<bool> g_buffers_initialized{false};

void init_preprocess_buffers() {
    if (g_buffers_initialized.exchange(true)) {
        return;  // Already initialized
    }
    
    g_rgb_buffer = static_cast<uint8_t*>(aligned_alloc_64(INPUT_WIDTH * INPUT_HEIGHT * 3));
    g_letterbox_buffer = static_cast<uint8_t*>(aligned_alloc_64(MODEL_SIZE * MODEL_SIZE * 3));
    g_resize_temp = static_cast<uint8_t*>(aligned_alloc_64(MODEL_SIZE * MODEL_SIZE * 3));
    
    // Touch pages to ensure they're mapped
    if (g_rgb_buffer) memset(g_rgb_buffer, 0, INPUT_WIDTH * INPUT_HEIGHT * 3);
    if (g_letterbox_buffer) memset(g_letterbox_buffer, 114, MODEL_SIZE * MODEL_SIZE * 3);
    if (g_resize_temp) memset(g_resize_temp, 0, MODEL_SIZE * MODEL_SIZE * 3);
}

void cleanup_preprocess_buffers() {
    if (!g_buffers_initialized.exchange(false)) {
        return;
    }
    
    aligned_free(g_rgb_buffer);
    aligned_free(g_letterbox_buffer);
    aligned_free(g_resize_temp);
    
    g_rgb_buffer = nullptr;
    g_letterbox_buffer = nullptr;
    g_resize_temp = nullptr;
}

// ============================================================================
// YUYV to RGB Conversion
// ============================================================================

// YUV to RGB conversion coefficients (fixed-point Q8.8)
// R = Y + 1.402 * (V - 128)   -> Y + (359 * (V - 128)) >> 8
// G = Y - 0.344 * (U - 128) - 0.714 * (V - 128) -> Y - (88 * (U - 128) + 183 * (V - 128)) >> 8
// B = Y + 1.772 * (U - 128)   -> Y + (454 * (U - 128)) >> 8

void yuyv_to_rgb_neon(
    const uint8_t* __restrict src,
    uint8_t* __restrict dst,
    int width,
    int height,
    int src_stride,
    int dst_stride
) {
    // Conversion constants
    const int16x8_t v_128 = vdupq_n_s16(128);
    const int16x8_t v_359 = vdupq_n_s16(359);  // 1.402 * 256
    const int16x8_t v_88 = vdupq_n_s16(88);    // 0.344 * 256
    const int16x8_t v_183 = vdupq_n_s16(183);  // 0.714 * 256
    const int16x8_t v_454 = vdupq_n_s16(454);  // 1.772 * 256
    const int16x8_t v_zero = vdupq_n_s16(0);
    const int16x8_t v_255 = vdupq_n_s16(255);
    
    for (int y = 0; y < height; y++) {
        const uint8_t* src_row = src + y * src_stride;
        uint8_t* dst_row = dst + y * dst_stride;
        
        // Prefetch next row
        if (y + 1 < height) {
            PREFETCH_L1(src + (y + 1) * src_stride);
        }
        
        // Process 16 pixels (32 YUYV bytes) per iteration
        int x = 0;
        for (; x <= width - 16; x += 16) {
            // Load 32 bytes of YUYV data
            // Format: Y0 U0 Y1 V0 Y2 U1 Y3 V1 ...
            uint8x16x2_t yuyv = vld2q_u8(src_row + x * 2);
            
            // yuyv.val[0] = Y values (16 of them)
            // yuyv.val[1] = U,V interleaved (U0,V0,U1,V1,...)
            
            // Deinterleave U and V
            uint8x8x2_t uv_low = vuzp_u8(vget_low_u8(yuyv.val[1]), vget_high_u8(yuyv.val[1]));
            // uv_low.val[0] = U values (8 of them, each used for 2 pixels)
            // uv_low.val[1] = V values
            
            // Expand U and V to match Y (duplicate each value)
            uint8x16_t u_expanded = vcombine_u8(
                vzip1_u8(uv_low.val[0], uv_low.val[0]),
                vzip2_u8(uv_low.val[0], uv_low.val[0])
            );
            uint8x16_t v_expanded = vcombine_u8(
                vzip1_u8(uv_low.val[1], uv_low.val[1]),
                vzip2_u8(uv_low.val[1], uv_low.val[1])
            );
            
            // Process first 8 pixels
            {
                int16x8_t y_s = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(yuyv.val[0])));
                int16x8_t u_s = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(u_expanded)));
                int16x8_t v_s = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v_expanded)));
                
                // Center U and V around 0
                int16x8_t u_centered = vsubq_s16(u_s, v_128);
                int16x8_t v_centered = vsubq_s16(v_s, v_128);
                
                // Calculate RGB
                // R = Y + (359 * V') >> 8
                int16x8_t r = vaddq_s16(y_s, vshrq_n_s16(vmulq_s16(v_359, v_centered), 8));
                
                // G = Y - (88 * U' + 183 * V') >> 8
                int16x8_t g = vsubq_s16(y_s, vshrq_n_s16(
                    vaddq_s16(vmulq_s16(v_88, u_centered), vmulq_s16(v_183, v_centered)), 8));
                
                // B = Y + (454 * U') >> 8
                int16x8_t b = vaddq_s16(y_s, vshrq_n_s16(vmulq_s16(v_454, u_centered), 8));
                
                // Clamp to [0, 255]
                r = vmaxq_s16(vminq_s16(r, v_255), v_zero);
                g = vmaxq_s16(vminq_s16(g, v_255), v_zero);
                b = vmaxq_s16(vminq_s16(b, v_255), v_zero);
                
                // Pack to uint8
                uint8x8_t r8 = vqmovun_s16(r);
                uint8x8_t g8 = vqmovun_s16(g);
                uint8x8_t b8 = vqmovun_s16(b);
                
                // Interleave and store RGB
                uint8x8x3_t rgb;
                rgb.val[0] = r8;
                rgb.val[1] = g8;
                rgb.val[2] = b8;
                vst3_u8(dst_row + x * 3, rgb);
            }
            
            // Process second 8 pixels
            {
                int16x8_t y_s = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(yuyv.val[0])));
                int16x8_t u_s = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(u_expanded)));
                int16x8_t v_s = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(v_expanded)));
                
                int16x8_t u_centered = vsubq_s16(u_s, v_128);
                int16x8_t v_centered = vsubq_s16(v_s, v_128);
                
                int16x8_t r = vaddq_s16(y_s, vshrq_n_s16(vmulq_s16(v_359, v_centered), 8));
                int16x8_t g = vsubq_s16(y_s, vshrq_n_s16(
                    vaddq_s16(vmulq_s16(v_88, u_centered), vmulq_s16(v_183, v_centered)), 8));
                int16x8_t b = vaddq_s16(y_s, vshrq_n_s16(vmulq_s16(v_454, u_centered), 8));
                
                r = vmaxq_s16(vminq_s16(r, v_255), v_zero);
                g = vmaxq_s16(vminq_s16(g, v_255), v_zero);
                b = vmaxq_s16(vminq_s16(b, v_255), v_zero);
                
                uint8x8_t r8 = vqmovun_s16(r);
                uint8x8_t g8 = vqmovun_s16(g);
                uint8x8_t b8 = vqmovun_s16(b);
                
                uint8x8x3_t rgb;
                rgb.val[0] = r8;
                rgb.val[1] = g8;
                rgb.val[2] = b8;
                vst3_u8(dst_row + (x + 8) * 3, rgb);
            }
        }
        
        // Handle remaining pixels (scalar)
        for (; x < width; x += 2) {
            int y0 = src_row[x * 2];
            int u = src_row[x * 2 + 1];
            int y1 = src_row[x * 2 + 2];
            int v = src_row[x * 2 + 3];
            
            int u_c = u - 128;
            int v_c = v - 128;
            
            auto clamp = [](int val) { return std::max(0, std::min(255, val)); };
            
            // Pixel 0
            dst_row[x * 3] = clamp(y0 + (359 * v_c >> 8));
            dst_row[x * 3 + 1] = clamp(y0 - ((88 * u_c + 183 * v_c) >> 8));
            dst_row[x * 3 + 2] = clamp(y0 + (454 * u_c >> 8));
            
            // Pixel 1
            if (x + 1 < width) {
                dst_row[(x + 1) * 3] = clamp(y1 + (359 * v_c >> 8));
                dst_row[(x + 1) * 3 + 1] = clamp(y1 - ((88 * u_c + 183 * v_c) >> 8));
                dst_row[(x + 1) * 3 + 2] = clamp(y1 + (454 * u_c >> 8));
            }
        }
    }
}

// ============================================================================
// Bilinear Resize
// ============================================================================

void bilinear_resize_rgb_neon(
    const uint8_t* __restrict src,
    uint8_t* __restrict dst,
    int src_width,
    int src_height,
    int dst_width,
    int dst_height,
    int src_stride,
    int dst_stride
) {
    // Scale factors in fixed-point Q16.16
    const int x_scale = ((src_width - 1) << 16) / (dst_width - 1);
    const int y_scale = ((src_height - 1) << 16) / (dst_height - 1);
    
    for (int dy = 0; dy < dst_height; dy++) {
        // Source Y coordinate in Q16.16
        int sy_fixed = dy * y_scale;
        int sy = sy_fixed >> 16;
        int fy = sy_fixed & 0xFFFF;  // Fractional part
        
        // Clamp source Y
        int sy0 = std::min(sy, src_height - 1);
        int sy1 = std::min(sy + 1, src_height - 1);
        
        const uint8_t* src_row0 = src + sy0 * src_stride;
        const uint8_t* src_row1 = src + sy1 * src_stride;
        uint8_t* dst_row = dst + dy * dst_stride;
        
        // Prefetch
        PREFETCH_L1(src_row0);
        PREFETCH_L1(src_row1);
        
        // Weight vectors for Y interpolation
        int wy1 = fy >> 8;  // 0-255 range
        int wy0 = 256 - wy1;
        
        int16x8_t v_wy0 = vdupq_n_s16(wy0);
        int16x8_t v_wy1 = vdupq_n_s16(wy1);
        
        // Process 8 destination pixels per iteration
        int dx = 0;
        for (; dx <= dst_width - 8; dx += 8) {
            // Calculate source X coordinates for 8 pixels
            int sx_coords[8];
            int fx_coords[8];
            
            for (int i = 0; i < 8; i++) {
                int sx_fixed = (dx + i) * x_scale;
                sx_coords[i] = std::min(sx_fixed >> 16, src_width - 2);
                fx_coords[i] = (sx_fixed & 0xFFFF) >> 8;  // 0-255
            }
            
            // Load and interpolate for each channel
            for (int c = 0; c < 3; c++) {
                int16_t result[8];
                
                for (int i = 0; i < 8; i++) {
                    int sx = sx_coords[i];
                    int wx1 = fx_coords[i];
                    int wx0 = 256 - wx1;
                    
                    // Bilinear interpolation
                    int p00 = src_row0[sx * 3 + c];
                    int p01 = src_row0[(sx + 1) * 3 + c];
                    int p10 = src_row1[sx * 3 + c];
                    int p11 = src_row1[(sx + 1) * 3 + c];
                    
                    int top = (p00 * wx0 + p01 * wx1) >> 8;
                    int bot = (p10 * wx0 + p11 * wx1) >> 8;
                    result[i] = (top * wy0 + bot * wy1) >> 8;
                }
                
                // Store results
                for (int i = 0; i < 8; i++) {
                    dst_row[(dx + i) * 3 + c] = result[i];
                }
            }
        }
        
        // Handle remaining pixels
        for (; dx < dst_width; dx++) {
            int sx_fixed = dx * x_scale;
            int sx = std::min(sx_fixed >> 16, src_width - 2);
            int fx = sx_fixed & 0xFFFF;
            
            int wx1 = fx >> 8;
            int wx0 = 256 - wx1;
            
            for (int c = 0; c < 3; c++) {
                int p00 = src_row0[sx * 3 + c];
                int p01 = src_row0[(sx + 1) * 3 + c];
                int p10 = src_row1[sx * 3 + c];
                int p11 = src_row1[(sx + 1) * 3 + c];
                
                int top = (p00 * wx0 + p01 * wx1) >> 8;
                int bot = (p10 * wx0 + p11 * wx1) >> 8;
                dst_row[dx * 3 + c] = (top * wy0 + bot * wy1) >> 8;
            }
        }
    }
}

// ============================================================================
// Letterbox Resize
// ============================================================================

void letterbox_resize_neon(
    const uint8_t* __restrict src,
    uint8_t* __restrict dst,
    int src_width,
    int src_height,
    int dst_size,
    float* scale,
    int* pad_x,
    int* pad_y
) {
    // Calculate scale to fit within dst_size
    float scale_x = static_cast<float>(dst_size) / src_width;
    float scale_y = static_cast<float>(dst_size) / src_height;
    *scale = std::min(scale_x, scale_y);
    
    int new_width = static_cast<int>(src_width * (*scale));
    int new_height = static_cast<int>(src_height * (*scale));
    
    // Calculate padding
    *pad_x = (dst_size - new_width) / 2;
    *pad_y = (dst_size - new_height) / 2;
    
    // Fill entire destination with gray (YOLO convention: 114)
    const uint8_t gray = 114;
    size_t dst_stride = dst_size * 3;
    
    // Use NEON to fill with gray
    uint8x16_t v_gray = vdupq_n_u8(gray);
    size_t total_bytes = dst_size * dst_stride;
    
    uint8_t* fill_ptr = dst;
    size_t i = 0;
    for (; i + 16 <= total_bytes; i += 16) {
        vst1q_u8(fill_ptr + i, v_gray);
    }
    for (; i < total_bytes; i++) {
        fill_ptr[i] = gray;
    }
    
    // Allocate temporary buffer for resized image
    size_t resized_stride = new_width * 3;
    size_t resized_size = new_height * resized_stride;
    uint8_t* resized = static_cast<uint8_t*>(aligned_alloc_64(resized_size));
    
    if (!resized) {
        return;  // Allocation failed
    }
    
    // Resize source to new dimensions
    bilinear_resize_rgb_neon(
        src, resized,
        src_width, src_height,
        new_width, new_height,
        src_width * 3, resized_stride
    );
    
    // Copy resized image to center of destination
    for (int y = 0; y < new_height; y++) {
        const uint8_t* src_row = resized + y * resized_stride;
        uint8_t* dst_row = dst + (y + *pad_y) * dst_stride + (*pad_x) * 3;
        
        // Use NEON memcpy for efficiency
        size_t row_bytes = new_width * 3;
        size_t j = 0;
        for (; j + 16 <= row_bytes; j += 16) {
            uint8x16_t v = vld1q_u8(src_row + j);
            vst1q_u8(dst_row + j, v);
        }
        for (; j < row_bytes; j++) {
            dst_row[j] = src_row[j];
        }
    }
    
    aligned_free(resized);
}

// ============================================================================
// RGB to CHW FP16
// ============================================================================

void rgb_to_chw_fp16_neon(
    const uint8_t* __restrict src,
    __fp16* __restrict dst,
    int width,
    int height,
    int stride
) {
    const float scale = 1.0f / 255.0f;
    float32x4_t v_scale = vdupq_n_f32(scale);
    
    size_t channel_size = width * height;
    __fp16* dst_r = dst;
    __fp16* dst_g = dst + channel_size;
    __fp16* dst_b = dst + channel_size * 2;
    
    for (int y = 0; y < height; y++) {
        const uint8_t* src_row = src + y * stride;
        __fp16* dst_r_row = dst_r + y * width;
        __fp16* dst_g_row = dst_g + y * width;
        __fp16* dst_b_row = dst_b + y * width;
        
        // Prefetch next row
        if (y + 1 < height) {
            PREFETCH_L1(src + (y + 1) * stride);
        }
        
        // Process 8 pixels per iteration
        int x = 0;
        for (; x <= width - 8; x += 8) {
            // Load 8 RGB pixels (24 bytes)
            uint8x8x3_t rgb = vld3_u8(src_row + x * 3);
            
            // Convert R channel
            {
                uint16x8_t r16 = vmovl_u8(rgb.val[0]);
                float32x4_t r_lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(r16)));
                float32x4_t r_hi = vcvtq_f32_u32(vmovl_u16(vget_high_u16(r16)));
                
                r_lo = vmulq_f32(r_lo, v_scale);
                r_hi = vmulq_f32(r_hi, v_scale);
                
                // Convert to FP16 and store
                float16x4_t r16_lo = vcvt_f16_f32(r_lo);
                float16x4_t r16_hi = vcvt_f16_f32(r_hi);
                vst1_f16(dst_r_row + x, r16_lo);
                vst1_f16(dst_r_row + x + 4, r16_hi);
            }
            
            // Convert G channel
            {
                uint16x8_t g16 = vmovl_u8(rgb.val[1]);
                float32x4_t g_lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(g16)));
                float32x4_t g_hi = vcvtq_f32_u32(vmovl_u16(vget_high_u16(g16)));
                
                g_lo = vmulq_f32(g_lo, v_scale);
                g_hi = vmulq_f32(g_hi, v_scale);
                
                float16x4_t g16_lo = vcvt_f16_f32(g_lo);
                float16x4_t g16_hi = vcvt_f16_f32(g_hi);
                vst1_f16(dst_g_row + x, g16_lo);
                vst1_f16(dst_g_row + x + 4, g16_hi);
            }
            
            // Convert B channel
            {
                uint16x8_t b16 = vmovl_u8(rgb.val[2]);
                float32x4_t b_lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(b16)));
                float32x4_t b_hi = vcvtq_f32_u32(vmovl_u16(vget_high_u16(b16)));
                
                b_lo = vmulq_f32(b_lo, v_scale);
                b_hi = vmulq_f32(b_hi, v_scale);
                
                float16x4_t b16_lo = vcvt_f16_f32(b_lo);
                float16x4_t b16_hi = vcvt_f16_f32(b_hi);
                vst1_f16(dst_b_row + x, b16_lo);
                vst1_f16(dst_b_row + x + 4, b16_hi);
            }
        }
        
        // Handle remaining pixels
        for (; x < width; x++) {
            dst_r_row[x] = static_cast<__fp16>(src_row[x * 3] * scale);
            dst_g_row[x] = static_cast<__fp16>(src_row[x * 3 + 1] * scale);
            dst_b_row[x] = static_cast<__fp16>(src_row[x * 3 + 2] * scale);
        }
    }
}

// ============================================================================
// Combined Preprocessing Pipeline
// ============================================================================

void preprocess_frame_neon(
    const uint8_t* __restrict yuyv_src,
    __fp16* __restrict fp16_dst,
    float* scale,
    int* pad_x,
    int* pad_y
) {
    // Allocate temporary buffers
    size_t rgb_size = INPUT_WIDTH * INPUT_HEIGHT * 3;
    size_t letterbox_size = MODEL_SIZE * MODEL_SIZE * 3;
    
    uint8_t* rgb_buf = static_cast<uint8_t*>(aligned_alloc_64(rgb_size));
    uint8_t* letterbox_buf = static_cast<uint8_t*>(aligned_alloc_64(letterbox_size));
    
    if (!rgb_buf || !letterbox_buf) {
        aligned_free(rgb_buf);
        aligned_free(letterbox_buf);
        return;
    }
    
    // Step 1: YUYV to RGB
    yuyv_to_rgb_neon(
        yuyv_src, rgb_buf,
        INPUT_WIDTH, INPUT_HEIGHT,
        INPUT_WIDTH * 2,  // YUYV stride
        INPUT_WIDTH * 3   // RGB stride
    );
    
    // Step 2: Letterbox resize
    letterbox_resize_neon(
        rgb_buf, letterbox_buf,
        INPUT_WIDTH, INPUT_HEIGHT,
        MODEL_SIZE,
        scale, pad_x, pad_y
    );
    
    // Step 3: RGB to CHW FP16
    rgb_to_chw_fp16_neon(
        letterbox_buf, fp16_dst,
        MODEL_SIZE, MODEL_SIZE,
        MODEL_SIZE * 3
    );
    
    aligned_free(rgb_buf);
    aligned_free(letterbox_buf);
}

/**
 * @brief NEON Memcpy
 */

void memcpy_neon(
    void* __restrict dst,
    const void* __restrict src,
    size_t size
) {
    uint8_t* d = static_cast<uint8_t*>(dst);
    const uint8_t* s = static_cast<const uint8_t*>(src);
    
    // Prefetch source data
    for (size_t i = 0; i < size; i += 64) {
        PREFETCH_L1(s + i);
    }
    
    // Copy 64 bytes per iteration (one cache line)
    size_t i = 0;
    for (; i + 64 <= size; i += 64) {
        uint8x16_t v0 = vld1q_u8(s + i);
        uint8x16_t v1 = vld1q_u8(s + i + 16);
        uint8x16_t v2 = vld1q_u8(s + i + 32);
        uint8x16_t v3 = vld1q_u8(s + i + 48);
        
        vst1q_u8(d + i, v0);
        vst1q_u8(d + i + 16, v1);
        vst1q_u8(d + i + 32, v2);
        vst1q_u8(d + i + 48, v3);
    }
    
    // Handle remaining bytes
    for (; i + 16 <= size; i += 16) {
        uint8x16_t v = vld1q_u8(s + i);
        vst1q_u8(d + i, v);
    }
    
    for (; i < size; i++) {
        d[i] = s[i];
    }
}

// ============================================================================
// OPTIMIZED: Direct BGR to FP32 CHW with Letterbox
// Bypasses ALL unnecessary conversions for video path
// ============================================================================

static void bilinear_resize_bgr_to_rgb_neon(
    const uint8_t* __restrict src,
    uint8_t* __restrict dst,
    int src_width,
    int src_height,
    int src_stride,
    int dst_width,
    int dst_height
) {
    const int x_scale = ((src_width - 1) << 16) / (dst_width - 1);
    const int y_scale = ((src_height - 1) << 16) / (dst_height - 1);
    const int dst_stride = dst_width * 3;
    
    for (int dy = 0; dy < dst_height; dy++) {
        int sy_fixed = dy * y_scale;
        int sy = sy_fixed >> 16;
        int fy = (sy_fixed & 0xFFFF) >> 8;
        
        int sy0 = std::min(sy, src_height - 1);
        int sy1 = std::min(sy + 1, src_height - 1);
        
        const uint8_t* src_row0 = src + sy0 * src_stride;
        const uint8_t* src_row1 = src + sy1 * src_stride;
        uint8_t* dst_row = dst + dy * dst_stride;
        
        PREFETCH_L1(src_row0 + 64);
        PREFETCH_L1(src_row1 + 64);
        
        int wy1 = fy;
        int wy0 = 256 - wy1;
        
        for (int dx = 0; dx < dst_width; dx++) {
            int sx_fixed = dx * x_scale;
            int sx = std::min(sx_fixed >> 16, src_width - 2);
            int wx1 = (sx_fixed & 0xFFFF) >> 8;
            int wx0 = 256 - wx1;
            
            // BGR to RGB conversion + bilinear in one pass
            for (int c = 0; c < 3; c++) {
                int src_c = 2 - c;  // BGR->RGB swap
                int p00 = src_row0[sx * 3 + src_c];
                int p01 = src_row0[(sx + 1) * 3 + src_c];
                int p10 = src_row1[sx * 3 + src_c];
                int p11 = src_row1[(sx + 1) * 3 + src_c];
                
                int top = (p00 * wx0 + p01 * wx1) >> 8;
                int bot = (p10 * wx0 + p11 * wx1) >> 8;
                dst_row[dx * 3 + c] = (top * wy0 + bot * wy1) >> 8;
            }
        }
    }
}

static void rgb_to_chw_fp32_neon(
    const uint8_t* __restrict src,
    float* __restrict dst,
    int width,
    int height,
    int stride
) {
    const float scale = 1.0f / 255.0f;
    float32x4_t v_scale = vdupq_n_f32(scale);
    
    size_t channel_size = width * height;
    float* dst_r = dst;
    float* dst_g = dst + channel_size;
    float* dst_b = dst + channel_size * 2;
    
    for (int y = 0; y < height; y++) {
        const uint8_t* src_row = src + y * stride;
        float* dst_r_row = dst_r + y * width;
        float* dst_g_row = dst_g + y * width;
        float* dst_b_row = dst_b + y * width;
        
        if (y + 1 < height) {
            PREFETCH_L1(src + (y + 1) * stride);
        }
        
        int x = 0;
        for (; x <= width - 8; x += 8) {
            uint8x8x3_t rgb = vld3_u8(src_row + x * 3);
            
            // R channel
            {
                uint16x8_t r16 = vmovl_u8(rgb.val[0]);
                float32x4_t r_lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(r16)));
                float32x4_t r_hi = vcvtq_f32_u32(vmovl_u16(vget_high_u16(r16)));
                r_lo = vmulq_f32(r_lo, v_scale);
                r_hi = vmulq_f32(r_hi, v_scale);
                vst1q_f32(dst_r_row + x, r_lo);
                vst1q_f32(dst_r_row + x + 4, r_hi);
            }
            
            // G channel
            {
                uint16x8_t g16 = vmovl_u8(rgb.val[1]);
                float32x4_t g_lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(g16)));
                float32x4_t g_hi = vcvtq_f32_u32(vmovl_u16(vget_high_u16(g16)));
                g_lo = vmulq_f32(g_lo, v_scale);
                g_hi = vmulq_f32(g_hi, v_scale);
                vst1q_f32(dst_g_row + x, g_lo);
                vst1q_f32(dst_g_row + x + 4, g_hi);
            }
            
            // B channel
            {
                uint16x8_t b16 = vmovl_u8(rgb.val[2]);
                float32x4_t b_lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(b16)));
                float32x4_t b_hi = vcvtq_f32_u32(vmovl_u16(vget_high_u16(b16)));
                b_lo = vmulq_f32(b_lo, v_scale);
                b_hi = vmulq_f32(b_hi, v_scale);
                vst1q_f32(dst_b_row + x, b_lo);
                vst1q_f32(dst_b_row + x + 4, b_hi);
            }
        }
        
        for (; x < width; x++) {
            dst_r_row[x] = src_row[x * 3] * scale;
            dst_g_row[x] = src_row[x * 3 + 1] * scale;
            dst_b_row[x] = src_row[x * 3 + 2] * scale;
        }
    }
}

void preprocess_bgr_direct(
    const uint8_t* __restrict bgr_src,
    float* __restrict fp32_dst,
    int src_width,
    int src_height,
    int src_stride,
    float* scale,
    int* pad_x,
    int* pad_y
) {
    // Ensure buffers are initialized
    if (!g_buffers_initialized.load()) {
        init_preprocess_buffers();
    }
    
    // Calculate letterbox parameters
    float scale_x = static_cast<float>(MODEL_SIZE) / src_width;
    float scale_y = static_cast<float>(MODEL_SIZE) / src_height;
    *scale = std::min(scale_x, scale_y);
    
    int new_width = static_cast<int>(src_width * (*scale));
    int new_height = static_cast<int>(src_height * (*scale));
    
    *pad_x = (MODEL_SIZE - new_width) / 2;
    *pad_y = (MODEL_SIZE - new_height) / 2;
    
    // Fill letterbox buffer with gray (114/255 normalized)
    const uint8_t gray = 114;
    memset(g_letterbox_buffer, gray, MODEL_SIZE * MODEL_SIZE * 3);
    
    // Resize BGR->RGB directly to center of letterbox buffer
    uint8_t* letterbox_center = g_letterbox_buffer + 
                                (*pad_y) * MODEL_SIZE * 3 + 
                                (*pad_x) * 3;
    
    // Resize with BGR->RGB conversion
    bilinear_resize_bgr_to_rgb_neon(
        bgr_src, g_resize_temp,
        src_width, src_height, src_stride,
        new_width, new_height
    );
    
    // Copy resized image to letterbox center
    for (int y = 0; y < new_height; y++) {
        memcpy(g_letterbox_buffer + (y + *pad_y) * MODEL_SIZE * 3 + (*pad_x) * 3,
               g_resize_temp + y * new_width * 3,
               new_width * 3);
    }
    
    // Convert to FP32 CHW
    rgb_to_chw_fp32_neon(
        g_letterbox_buffer, fp32_dst,
        MODEL_SIZE, MODEL_SIZE,
        MODEL_SIZE * 3
    );
}

// ============================================================================
// OPTIMIZED: YUYV to FP32 (for camera path)
// ============================================================================

void preprocess_yuyv_to_fp32(
    const uint8_t* __restrict yuyv_src,
    float* __restrict fp32_dst,
    float* scale,
    int* pad_x,
    int* pad_y
) {
    if (!g_buffers_initialized.load()) {
        init_preprocess_buffers();
    }
    
    // Step 1: YUYV to RGB
    yuyv_to_rgb_neon(
        yuyv_src, g_rgb_buffer,
        INPUT_WIDTH, INPUT_HEIGHT,
        INPUT_WIDTH * 2,
        INPUT_WIDTH * 3
    );
    
    // Step 2: Letterbox resize
    *scale = static_cast<float>(MODEL_SIZE) / INPUT_WIDTH;
    float scale_y = static_cast<float>(MODEL_SIZE) / INPUT_HEIGHT;
    if (scale_y < *scale) *scale = scale_y;
    
    int new_width = static_cast<int>(INPUT_WIDTH * (*scale));
    int new_height = static_cast<int>(INPUT_HEIGHT * (*scale));
    
    *pad_x = (MODEL_SIZE - new_width) / 2;
    *pad_y = (MODEL_SIZE - new_height) / 2;
    
    // Fill with gray
    memset(g_letterbox_buffer, 114, MODEL_SIZE * MODEL_SIZE * 3);
    
    // Resize
    bilinear_resize_rgb_neon(
        g_rgb_buffer, g_resize_temp,
        INPUT_WIDTH, INPUT_HEIGHT,
        new_width, new_height,
        INPUT_WIDTH * 3, new_width * 3
    );
    
    // Copy to letterbox center
    for (int y = 0; y < new_height; y++) {
        memcpy(g_letterbox_buffer + (y + *pad_y) * MODEL_SIZE * 3 + (*pad_x) * 3,
               g_resize_temp + y * new_width * 3,
               new_width * 3);
    }
    
    // Convert to FP32 CHW
    rgb_to_chw_fp32_neon(
        g_letterbox_buffer, fp32_dst,
        MODEL_SIZE, MODEL_SIZE,
        MODEL_SIZE * 3
    );
}

}  // namespace neon
}  // namespace yolo
