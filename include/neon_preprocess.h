/**
 * @file neon_preprocess.h
 * @brief ARM NEON optimized preprocessing functions
 * 
 * NEON & Data-Path Optimization Agent Decision Log:
 * - All functions process 8 pixels per iteration minimum
 * - No branches in inner loops
 * - Prefetch distance: 256 bytes ahead (4 cache lines)
 * - Input/output buffers must be 64-byte aligned
 */

#ifndef YOLO_NEON_PREPROCESS_H
#define YOLO_NEON_PREPROCESS_H

#include "common.h"

namespace yolo {
namespace neon {

/**
 * @brief Convert YUYV (YUV422) to RGB888
 * 
 * Processes 16 YUYV bytes (8 pixels) per iteration using NEON.
 * Formula: 
 *   R = Y + 1.402 * (V - 128)
 *   G = Y - 0.344 * (U - 128) - 0.714 * (V - 128)
 *   B = Y + 1.772 * (U - 128)
 * 
 * @param src Source YUYV buffer (must be 16-byte aligned)
 * @param dst Destination RGB buffer (must be 16-byte aligned)
 * @param width Image width in pixels (must be multiple of 16)
 * @param height Image height in pixels
 * @param src_stride Source stride in bytes
 * @param dst_stride Destination stride in bytes
 */
void yuyv_to_rgb_neon(
    const uint8_t* __restrict src,
    uint8_t* __restrict dst,
    int width,
    int height,
    int src_stride,
    int dst_stride
);

/**
 * @brief Bilinear resize with NEON optimization
 * 
 * Handles non-square resize (640x480 -> 640x640 with letterboxing)
 * Uses fixed-point arithmetic for coordinate calculation.
 * 
 * @param src Source RGB buffer
 * @param dst Destination RGB buffer
 * @param src_width Source width
 * @param src_height Source height
 * @param dst_width Destination width
 * @param dst_height Destination height
 * @param src_stride Source stride in bytes
 * @param dst_stride Destination stride in bytes
 */
void bilinear_resize_rgb_neon(
    const uint8_t* __restrict src,
    uint8_t* __restrict dst,
    int src_width,
    int src_height,
    int dst_width,
    int dst_height,
    int src_stride,
    int dst_stride
);

/**
 * @brief Letterbox resize maintaining aspect ratio
 * 
 * Resizes image to fit within target dimensions while maintaining
 * aspect ratio. Fills padding with gray (114, 114, 114) as per YOLO convention.
 * 
 * @param src Source RGB buffer
 * @param dst Destination RGB buffer (will be filled including padding)
 * @param src_width Source width
 * @param src_height Source height  
 * @param dst_size Target size (square, e.g., 640)
 * @param scale Output: scale factor applied
 * @param pad_x Output: horizontal padding (left)
 * @param pad_y Output: vertical padding (top)
 */
void letterbox_resize_neon(
    const uint8_t* __restrict src,
    uint8_t* __restrict dst,
    int src_width,
    int src_height,
    int dst_size,
    float* scale,
    int* pad_x,
    int* pad_y
);

/**
 * @brief Convert RGB uint8 to normalized FP16 in CHW layout
 * 
 * Performs:
 * 1. HWC -> CHW transpose
 * 2. uint8 -> FP16 conversion
 * 3. Normalization to [0, 1] range (divide by 255)
 * 
 * @param src Source RGB buffer (HWC layout)
 * @param dst Destination FP16 buffer (CHW layout)
 * @param width Image width
 * @param height Image height
 * @param stride Source stride in bytes
 */
void rgb_to_chw_fp16_neon(
    const uint8_t* __restrict src,
    __fp16* __restrict dst,
    int width,
    int height,
    int stride
);

/**
 * @brief Combined preprocessing: YUYV -> RGB -> Resize -> Normalize
 * 
 * Single-pass optimized pipeline combining all preprocessing steps.
 * Minimizes memory bandwidth by fusing operations where possible.
 * 
 * @param yuyv_src Source YUYV frame (640x480)
 * @param fp16_dst Destination FP16 tensor (3x640x640 CHW)
 * @param scale Output: scale factor for coordinate mapping
 * @param pad_x Output: horizontal padding for coordinate mapping
 * @param pad_y Output: vertical padding for coordinate mapping
 */
void preprocess_frame_neon(
    const uint8_t* __restrict yuyv_src,
    __fp16* __restrict fp16_dst,
    float* scale,
    int* pad_x,
    int* pad_y
);

/**
 * @brief Prefetch-optimized memcpy for frame buffers
 * 
 * Uses NEON load/store with prefetch for maximum bandwidth.
 * 
 * @param dst Destination buffer (64-byte aligned)
 * @param src Source buffer (64-byte aligned)
 * @param size Number of bytes to copy (multiple of 64)
 */
void memcpy_neon(
    void* __restrict dst,
    const void* __restrict src,
    size_t size
);

/**
 * @brief Initialize pre-allocated static buffers
 * Must be called once before using preprocess functions.
 * Thread-safe, can be called multiple times.
 */
void init_preprocess_buffers();

/**
 * @brief Free pre-allocated static buffers
 * Call on shutdown.
 */
void cleanup_preprocess_buffers();

/**
 * @brief OPTIMIZED: Direct BGR to FP32 CHW with letterbox
 * 
 * Bypasses YUYV conversion entirely for video files.
 * Output is FP32 for direct NCNN consumption (no FP16 intermediate).
 * Uses pre-allocated buffers - NO malloc in hot path.
 * 
 * @param bgr_src Source BGR buffer (OpenCV format)
 * @param fp32_dst Destination FP32 tensor (3x640x640 CHW)
 * @param src_width Source image width
 * @param src_height Source image height
 * @param src_stride Source stride in bytes
 * @param scale Output: scale factor for coordinate mapping
 * @param pad_x Output: horizontal padding for coordinate mapping  
 * @param pad_y Output: vertical padding for coordinate mapping
 */
void preprocess_bgr_direct(
    const uint8_t* __restrict bgr_src,
    float* __restrict fp32_dst,
    int src_width,
    int src_height,
    int src_stride,
    float* scale,
    int* pad_x,
    int* pad_y
);

/**
 * @brief OPTIMIZED: YUYV to FP32 CHW with letterbox
 * 
 * For camera path - outputs FP32 directly for NCNN.
 * Uses pre-allocated buffers - NO malloc in hot path.
 * 
 * @param yuyv_src Source YUYV frame (640x480)
 * @param fp32_dst Destination FP32 tensor (3x640x640 CHW)
 * @param scale Output: scale factor for coordinate mapping
 * @param pad_x Output: horizontal padding for coordinate mapping
 * @param pad_y Output: vertical padding for coordinate mapping
 */
void preprocess_yuyv_to_fp32(
    const uint8_t* __restrict yuyv_src,
    float* __restrict fp32_dst,
    float* scale,
    int* pad_x,
    int* pad_y
);

// ============================================================================
// Inline NEON Utilities
// ============================================================================

/**
 * @brief Clamp float32x4 vector to [0, 255] range and convert to uint8x8
 */
FORCE_INLINE uint8x8_t clamp_and_narrow_f32x4x2(float32x4_t a, float32x4_t b) {
    // Clamp to [0, 255]
    float32x4_t min_val = vdupq_n_f32(0.0f);
    float32x4_t max_val = vdupq_n_f32(255.0f);
    
    a = vmaxq_f32(vminq_f32(a, max_val), min_val);
    b = vmaxq_f32(vminq_f32(b, max_val), min_val);
    
    // Convert to int32
    int32x4_t ai = vcvtq_s32_f32(a);
    int32x4_t bi = vcvtq_s32_f32(b);
    
    // Narrow to int16
    int16x4_t a16 = vmovn_s32(ai);
    int16x4_t b16 = vmovn_s32(bi);
    int16x8_t ab16 = vcombine_s16(a16, b16);
    
    // Narrow to uint8
    return vqmovun_s16(ab16);
}

/**
 * @brief Convert uint8x8 to float32x4 (lower half)
 */
FORCE_INLINE float32x4_t u8_to_f32_low(uint8x8_t v) {
    uint16x8_t v16 = vmovl_u8(v);
    uint32x4_t v32 = vmovl_u16(vget_low_u16(v16));
    return vcvtq_f32_u32(v32);
}

/**
 * @brief Convert uint8x8 to float32x4 (upper half)
 */
FORCE_INLINE float32x4_t u8_to_f32_high(uint8x8_t v) {
    uint16x8_t v16 = vmovl_u8(v);
    uint32x4_t v32 = vmovl_u16(vget_high_u16(v16));
    return vcvtq_f32_u32(v32);
}

}  // namespace neon
}  // namespace yolo

#endif  // YOLO_NEON_PREPROCESS_H
