/**
 * @file inference_engine.h
 * @brief NCNN-based YOLOv8n inference engine
 * 
 * Inference & NCNN Agent Design:
 * - Static NCNN build for Cortex-A76 with FP16
 * - Thread pool pinned to CPU 2-3 (avoiding input thread on CPU 0-1)
 * - Pre-allocated blob memory to avoid runtime allocation
 * - Custom allocator for deterministic memory behavior
 */

#ifndef YOLO_INFERENCE_ENGINE_H
#define YOLO_INFERENCE_ENGINE_H

#include "common.h"
#include <string>
#include <net.h>

namespace yolo {

/**
 * @brief YOLOv8n inference engine using NCNN
 */
class InferenceEngine {
public:
    /**
     * @brief Engine configuration
     */
    struct Config {
        std::string param_path;     // Path to .param file
        std::string bin_path;       // Path to .bin file
        int num_threads = NCNN_NUM_THREADS;
        bool use_fp16 = true;       // Use FP16 inference
        bool use_packing = true;    // Use NEON packing
        int light_mode = 1;         // NCNN light mode
    };

    InferenceEngine();
    ~InferenceEngine();

    // Non-copyable
    InferenceEngine(const InferenceEngine&) = delete;
    InferenceEngine& operator=(const InferenceEngine&) = delete;

    /**
     * @brief Initialize the inference engine
     * @param config Engine configuration
     * @return Error code
     */
    ErrorCode initialize(const Config& config);

    /**
     * @brief Run inference on preprocessed input
     * @param input_data FP16 tensor in CHW layout (3x640x640)
     * @param result Output detection result
     * @return Error code
     */
    ErrorCode infer(const __fp16* input_data, DetectionResult& result);

    /**
     * @brief OPTIMIZED: Run inference with FP32 input directly
     * @param input_data FP32 tensor in CHW layout (3x640x640)
     * @param result Output detection result
     * @return Error code
     */
    ErrorCode infer_fp32(const float* input_data, DetectionResult& result);

    /**
     * @brief Set letterbox parameters for coordinate mapping
     */
    void set_letterbox_params(float scale, int pad_x, int pad_y) {
        scale_ = scale;
        pad_x_ = pad_x;
        pad_y_ = pad_y;
    }

    /**
     * @brief Check if engine is initialized
     */
    bool is_initialized() const { return initialized_; }

    /**
     * @brief Get model input dimensions
     */
    void get_input_size(int& width, int& height) const {
        width = MODEL_SIZE;
        height = MODEL_SIZE;
    }

    /**
     * @brief Warm up the engine (run dummy inference to initialize caches)
     * @param iterations Number of warmup iterations
     */
    void warmup(int iterations = 3);

private:
    /**
     * @brief Decode YOLOv8 output tensor to detections
     */
    void decode_outputs(const float* output_data, int output_size, DetectionResult& result);

    /**
     * @brief Apply Non-Maximum Suppression
     */
    void apply_nms(DetectionResult& result);

    bool initialized_ = false;
    Config config_;
    
    ncnn::Net* net_ = nullptr;
    ncnn::PoolAllocator* blob_pool_allocator_ = nullptr;
    ncnn::UnlockedPoolAllocator* workspace_allocator_ = nullptr;
    
    // Pre-allocated buffers
    AlignedPtr<float> output_buffer_;
    size_t output_buffer_size_ = 0;
    
    // Input/output blob names
    std::string input_name_;
    std::string output_name_;
    
    // Coordinate mapping for letterbox
    float scale_ = 1.0f;
    int pad_x_ = 0;
    int pad_y_ = 0;
};

}  // namespace yolo

#endif  // YOLO_INFERENCE_ENGINE_H
