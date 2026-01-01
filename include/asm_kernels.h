/**
 * @file asm_kernels.h
 * @brief Declarations for ARM64 assembly kernels
 * 
 * These are hand-tuned assembly routines for Cortex-A76.
 * They provide maximum performance for hot path operations.
 */

#ifndef YOLO_ASM_KERNELS_H
#define YOLO_ASM_KERNELS_H

#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Convert RGB interleaved to FP32 planar CHW with normalization
 * 
 * Hand-tuned ARM64 assembly with:
 * - LD3 for efficient RGB deinterleave
 * - Vectorized int->float conversion
 * - Fused normalization (divide by 255)
 * - Aggressive prefetching
 * 
 * @param src Source RGB buffer (HWC layout)
 * @param dst_r Output R channel (float*)
 * @param dst_g Output G channel (float*)
 * @param dst_b Output B channel (float*)
 * @param width Image width
 * @param height Image height
 * @param src_stride Source stride in bytes
 */
void rgb_to_chw_fp32_asm(
    const uint8_t* src,
    float* dst_r,
    float* dst_g,
    float* dst_b,
    int width,
    int height,
    int src_stride
);

/**
 * @brief High-performance memcpy using NEON
 * 
 * Optimized for large buffers with cache-line aligned transfers.
 * 
 * @param dst Destination buffer
 * @param src Source buffer
 * @param size Number of bytes to copy
 */
void memcpy_neon_asm(
    void* dst,
    const void* src,
    size_t size
);

/**
 * @brief Transpose tensor [84, 8400] -> [8400, 84]
 * 
 * Optimized for YOLOv8 output tensor transpose.
 * 
 * @param src Source tensor (float*, [num_rows][num_cols])
 * @param dst Destination tensor (float*, [num_cols][num_rows])
 * @param num_cols Number of columns (8400 for YOLOv8 @416)
 * @param num_rows Number of rows (84 for YOLOv8)
 */
void transpose_84x8400_asm(
    const float* src,
    float* dst,
    int num_cols,
    int num_rows
);

#ifdef __cplusplus
}
#endif

#endif // YOLO_ASM_KERNELS_H
