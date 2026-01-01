/**
 * @file inference_engine.cpp
 * @brief NCNN-based YOLOv8n inference engine - RADICALLY OPTIMIZED
 * 
 * CRITICAL OPTIMIZATIONS:
 * - Direct FP32 input (no FP16 conversion overhead)
 * - Optimized thread count (2 threads for better cache sharing)
 * - Vectorized output transpose
 * - Zero-copy where possible
 * - INT8 quantization support for 2-4x speedup
 * - Vulkan GPU support for VideoCore VII (RPi5)
 */

#include "inference_engine.h"
#include "postprocess.h"
#include "asm_kernels.h"

#include <net.h>
#include <cpu.h>
#if NCNN_VULKAN
#include <gpu.h>
#endif
#include <cstring>
#include <arm_neon.h>
#include <iostream>

namespace yolo {

// ============================================================================
// InferenceEngine Implementation
// ============================================================================

InferenceEngine::InferenceEngine() = default;

InferenceEngine::~InferenceEngine() {
    if (net_) {
        delete net_;
    }
    if (blob_pool_allocator_) {
        delete blob_pool_allocator_;
    }
    if (workspace_allocator_) {
        delete workspace_allocator_;
    }
#if NCNN_VULKAN
    if (blob_vkallocator_) {
        delete blob_vkallocator_;
    }
    if (staging_vkallocator_) {
        delete staging_vkallocator_;
    }
#endif
}

ErrorCode InferenceEngine::initialize(const Config& config) {
    config_ = config;
    
    // Configure NCNN global settings
    ncnn::set_cpu_powersave(0);  // Use all cores at full speed
    ncnn::set_omp_num_threads(config.num_threads);
    
#if NCNN_VULKAN
    // Initialize Vulkan if requested
    if (config.use_vulkan) {
        int gpu_count = ncnn::get_gpu_count();
        if (gpu_count > 0 && config.gpu_device < gpu_count) {
            using_vulkan_ = true;
            std::cout << "Vulkan GPU: " << ncnn::get_gpu_info(config.gpu_device).device_name() 
                      << " (device " << config.gpu_device << "/" << gpu_count << ")\n";
        } else {
            std::cout << "Vulkan requested but no GPU found, falling back to CPU\n";
        }
    }
#endif

    // Create NCNN pool allocators
    blob_pool_allocator_ = new ncnn::PoolAllocator();
    workspace_allocator_ = new ncnn::UnlockedPoolAllocator();
    
    // Pre-allocate pools to avoid runtime allocation
    blob_pool_allocator_->set_size_compare_ratio(0.0f);  // Exact match only
    workspace_allocator_->set_size_compare_ratio(0.0f);
    
#if NCNN_VULKAN
    if (using_vulkan_) {
        ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(config.gpu_device);
        blob_vkallocator_ = new ncnn::VkBlobAllocator(vkdev);
        staging_vkallocator_ = new ncnn::VkStagingAllocator(vkdev);
    }
#endif

    // Create and configure network
    net_ = new ncnn::Net();
    
#if NCNN_VULKAN
    if (using_vulkan_) {
        net_->opt.use_vulkan_compute = true;
        net_->set_vulkan_device(config.gpu_device);
    }
#endif
    
    net_->opt.lightmode = config.light_mode;
    net_->opt.num_threads = config.num_threads;
    net_->opt.use_packing_layout = config.use_packing;
    
    // INT8 specific settings
    if (config.use_int8) {
        using_int8_ = true;
        net_->opt.use_int8_inference = true;
        net_->opt.use_int8_storage = true;
        net_->opt.use_int8_arithmetic = true;
        // Disable FP16 when using INT8
        net_->opt.use_fp16_packed = false;
        net_->opt.use_fp16_storage = false;
        net_->opt.use_fp16_arithmetic = false;
        std::cout << "INT8 quantization enabled\n";
    } else {
        net_->opt.use_fp16_packed = config.use_fp16;
        net_->opt.use_fp16_storage = config.use_fp16;
        net_->opt.use_fp16_arithmetic = config.use_fp16;
    }
    
    net_->opt.blob_allocator = blob_pool_allocator_;
    net_->opt.workspace_allocator = workspace_allocator_;
    
#if NCNN_VULKAN
    if (using_vulkan_) {
        net_->opt.blob_vkallocator = blob_vkallocator_;
        net_->opt.workspace_vkallocator = blob_vkallocator_;
        net_->opt.staging_vkallocator = staging_vkallocator_;
    }
#endif
    
    // Load model
    int ret = net_->load_param(config.param_path.c_str());
    if (ret != 0) {
        return ErrorCode::MODEL_LOAD_FAILED;
    }
    
    ret = net_->load_model(config.bin_path.c_str());
    if (ret != 0) {
        return ErrorCode::MODEL_LOAD_FAILED;
    }
    
    // Get input/output blob names - auto-detect from model
    const auto& input_names = net_->input_names();
    const auto& output_names = net_->output_names();
    
    if (!input_names.empty()) {
        input_name_ = input_names[0];
    } else {
        input_name_ = "images";
    }
    
    if (!output_names.empty()) {
        output_name_ = output_names[0];
    } else {
        output_name_ = "output0";
    }
    
    // Allocate output buffer for transposed data
    output_buffer_size_ = NUM_OUTPUTS * (4 + NUM_CLASSES);
    output_buffer_ = make_aligned_buffer<float>(output_buffer_size_);
    
    if (!output_buffer_) {
        return ErrorCode::MEMORY_ALLOCATION_FAILED;
    }
    
    initialized_ = true;
    return ErrorCode::SUCCESS;
}

// ============================================================================
// OPTIMIZED: Direct FP32 inference (no conversion!)
// ============================================================================

ErrorCode InferenceEngine::infer_fp32(const float* input_data, DetectionResult& result) {
    if (!initialized_) {
        return ErrorCode::MODEL_LOAD_FAILED;
    }
    
    result.count = 0;
    
    // Create NCNN Mat directly from FP32 data (ZERO-COPY!)
    // NCNN Mat uses CHW layout, same as our input
    ncnn::Mat input(MODEL_SIZE, MODEL_SIZE, 3, (void*)input_data);
    
    // Run inference
    ncnn::Extractor ex = net_->create_extractor();
    ex.set_light_mode(config_.light_mode);
    
    ex.input(input_name_.c_str(), input);
    
    ncnn::Mat output;
    int ret = ex.extract(output_name_.c_str(), output);
    
    if (ret != 0) {
        return ErrorCode::INFERENCE_FAILED;
    }
    
    // OPTIMIZED: Assembly transpose [84, 8400] -> [8400, 84]
    int num_proposals = output.w;
    int num_channels = output.h;
    
    // Use hand-tuned assembly for transpose
    transpose_84x8400_asm(
        (const float*)output.data,
        output_buffer_.get(),
        num_proposals,
        num_channels
    );
    
    // Decode outputs
    result.count = decode_yolov8_output(
        output_buffer_.get(),
        num_proposals,
        CONF_THRESHOLD,
        result.detections,
        MAX_DETECTIONS
    );
    
    // Apply NMS
    if (result.count > 0) {
        sort_detections_by_confidence(result.detections, result.count);
        nms_sorted(result.detections, result.count, NMS_THRESHOLD);
    }
    
    // Map coordinates back to original image space
    for (int i = 0; i < result.count; i++) {
        map_detection_to_original(
            result.detections[i],
            scale_, pad_x_, pad_y_,
            INPUT_WIDTH, INPUT_HEIGHT
        );
    }
    
    return ErrorCode::SUCCESS;
}

// Legacy FP16 interface (now just converts and calls FP32)
ErrorCode InferenceEngine::infer(const __fp16* input_data, DetectionResult& result) {
    // Convert FP16 to FP32 for NCNN
    AlignedPtr<float> fp32_input = make_aligned_buffer<float>(MODEL_INPUT_SIZE);
    
    const size_t channel_size = MODEL_SIZE * MODEL_SIZE;
    
    for (int c = 0; c < 3; c++) {
        float* dst = fp32_input.get() + c * channel_size;
        const __fp16* src = input_data + c * channel_size;
        
        size_t i = 0;
        for (; i + 8 <= channel_size; i += 8) {
            float16x8_t fp16_vec = vld1q_f16(src + i);
            float32x4_t fp32_lo = vcvt_f32_f16(vget_low_f16(fp16_vec));
            float32x4_t fp32_hi = vcvt_f32_f16(vget_high_f16(fp16_vec));
            vst1q_f32(dst + i, fp32_lo);
            vst1q_f32(dst + i + 4, fp32_hi);
        }
        for (; i < channel_size; i++) {
            dst[i] = static_cast<float>(src[i]);
        }
    }
    
    return infer_fp32(fp32_input.get(), result);
}

void InferenceEngine::warmup(int iterations) {
    if (!initialized_) return;
    
    // Create dummy input
    AlignedPtr<float> dummy_input = make_aligned_buffer<float>(MODEL_INPUT_SIZE);
    memset(dummy_input.get(), 0, MODEL_INPUT_SIZE * sizeof(float));
    
    DetectionResult dummy_result;
    
    for (int i = 0; i < iterations; i++) {
        infer_fp32(dummy_input.get(), dummy_result);
    }
}

}  // namespace yolo
