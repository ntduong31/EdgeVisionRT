/**
 * @file main.cpp
 * @brief Entry point for YOLOv8n realtime inference system - OPTIMIZED
 * 
 * RADICAL OPTIMIZATIONS:
 * - Direct BGR path for video (no YUYV conversion)
 * - FP32 throughout (no FP16 overhead)
 * - Pre-allocated buffers
 * - Optimized thread configuration
 */

#include "common.h"
#include "input_pipeline.h"
#include "neon_preprocess.h"
#include "inference_engine.h"
#include "postprocess.h"
#include "benchmark.h"
#include "video_writer.h"

#include <iostream>
#include <string>
#include <atomic>
#include <signal.h>
#include <getopt.h>
#include <set>
#include <sstream>
#include <algorithm>

using namespace yolo;

// ============================================================================
// Global State
// ============================================================================

std::atomic<bool> g_running{true};

void signal_handler(int sig) {
    g_running.store(false);
}

// ============================================================================
// Command Line Options
// ============================================================================

struct Options {
    std::string mode = "benchmark";      // benchmark, camera, video
    std::string device = "/dev/video0";  // Camera device or video file
    std::string param_path;              // Model param file
    std::string bin_path;                // Model bin file
    int frames = 1000;                   // Frames to process
    int warmup_frames = 30;              // Warmup frames
    bool verbose = false;                // Verbose output
    bool test_model = false;             // Test model loading only
    bool test_inference = false;         // Test single inference only
    bool test_camera = false;            // Test camera capture only
    std::string output_csv;              // Output CSV path
    std::string output_video;            // Output video path (with bbox)
    bool show_fps = true;                // Show FPS overlay in output video
    std::set<int> class_filter;          // Classes to detect (empty = all)
    std::string class_filter_str;        // Original class names string
    bool display_enabled = false;        // Enable display window
};

void print_usage(const char* program) {
    std::cout << "YOLOv8n Realtime Inference System for Raspberry Pi 5\n\n";
    std::cout << "Usage: " << program << " [options]\n\n";
    std::cout << "Modes:\n";
    std::cout << "  --benchmark          Run benchmark with video or synthetic data\n";
    std::cout << "  --camera DEVICE      Run with camera input\n";
    std::cout << "  --video FILE         Run with video file input\n\n";
    std::cout << "Model:\n";
    std::cout << "  --param FILE         Path to NCNN .param file\n";
    std::cout << "  --bin FILE           Path to NCNN .bin file\n\n";
    std::cout << "Options:\n";
    std::cout << "  --frames N           Number of frames to process (default: 1000)\n";
    std::cout << "  --warmup N           Warmup frames (default: 30)\n";
    std::cout << "  --output FILE        Export results to CSV\n";
    std::cout << "  --output-video FILE  Save video with bounding boxes\n";
    std::cout << "  --no-fps             Don't show FPS overlay in output video\n";
    std::cout << "  --class NAMES        Filter classes (comma-separated, e.g., 'person,car,dog')\n";
    std::cout << "  --display            Show detection results in window (auto DISPLAY=:0)\n";
    std::cout << "  --verbose            Print per-frame results\n\n";
    std::cout << "Testing:\n";
    std::cout << "  --test-model         Test model loading\n";
    std::cout << "  --test-inference     Test single frame inference\n";
    std::cout << "  --test-camera        Test camera capture\n\n";
}

bool parse_options(int argc, char* argv[], Options& opts) {
    static struct option long_options[] = {
        {"benchmark", no_argument, 0, 'b'},
        {"camera", required_argument, 0, 'c'},
        {"video", required_argument, 0, 'v'},
        {"param", required_argument, 0, 'p'},
        {"bin", required_argument, 0, 'm'},
        {"frames", required_argument, 0, 'n'},
        {"warmup", required_argument, 0, 'w'},
        {"output", required_argument, 0, 'o'},
        {"output-video", required_argument, 0, 'O'},
        {"no-fps", no_argument, 0, 'F'},
        {"class", required_argument, 0, 'C'},
        {"display", no_argument, 0, 'D'},
        {"verbose", no_argument, 0, 'V'},
        {"test-model", no_argument, 0, '1'},
        {"test-inference", no_argument, 0, '2'},
        {"test-camera", no_argument, 0, '3'},
        {"device", required_argument, 0, 'd'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "bc:v:p:m:n:w:o:O:FC:DVd:h", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'b':
                opts.mode = "benchmark";
                break;
            case 'c':
                opts.mode = "camera";
                opts.device = optarg;
                break;
            case 'v':
                opts.mode = "video";
                opts.device = optarg;
                break;
            case 'p':
                opts.param_path = optarg;
                break;
            case 'm':
                opts.bin_path = optarg;
                break;
            case 'n':
                opts.frames = std::stoi(optarg);
                break;
            case 'w':
                opts.warmup_frames = std::stoi(optarg);
                break;
            case 'o':
                opts.output_csv = optarg;
                break;
            case 'O':
                opts.output_video = optarg;
                break;
            case 'F':
                opts.show_fps = false;
                break;
            case 'C':
                opts.class_filter_str = optarg;
                break;
            case 'D':
                opts.display_enabled = true;
                break;
            case 'V':
                opts.verbose = true;
                break;
            case 'd':
                opts.device = optarg;
                break;
            case '1':
                opts.test_model = true;
                break;
            case '2':
                opts.test_inference = true;
                break;
            case '3':
                opts.test_camera = true;
                break;
            case 'h':
                print_usage(argv[0]);
                return false;
            default:
                print_usage(argv[0]);
                return false;
        }
    }

    // Validate required options
    if (!opts.test_camera && (opts.param_path.empty() || opts.bin_path.empty())) {
        std::cerr << "Error: --param and --bin are required\n";
        return false;
    }

    // Parse class filter if specified
    if (!opts.class_filter_str.empty()) {
        // COCO class names (must match video_writer.h)
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
        constexpr int NUM_COCO_CLASSES = 80;
        
        std::stringstream ss(opts.class_filter_str);
        std::string class_name;
        while (std::getline(ss, class_name, ',')) {
            // Trim whitespace
            class_name.erase(0, class_name.find_first_not_of(" \t"));
            class_name.erase(class_name.find_last_not_of(" \t") + 1);
            
            // Convert to lowercase for comparison
            std::string class_lower = class_name;
            std::transform(class_lower.begin(), class_lower.end(), class_lower.begin(), ::tolower);
            
            bool found = false;
            for (int i = 0; i < NUM_COCO_CLASSES; i++) {
                std::string coco_lower = COCO_NAMES[i];
                std::transform(coco_lower.begin(), coco_lower.end(), coco_lower.begin(), ::tolower);
                if (class_lower == coco_lower) {
                    opts.class_filter.insert(i);
                    found = true;
                    break;
                }
            }
            if (!found) {
                std::cerr << "Warning: Unknown class '" << class_name << "' ignored\n";
                std::cerr << "Available classes: person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, etc.\n";
            }
        }
        
        if (!opts.class_filter.empty()) {
            std::cout << "Class filter: ";
            for (int id : opts.class_filter) {
                std::cout << COCO_NAMES[id] << " ";
            }
            std::cout << "(" << opts.class_filter.size() << " classes)\n";
        }
    }

    return true;
}

// ============================================================================
// Test Functions
// ============================================================================

int test_model_loading(const Options& opts) {
    std::cout << "Testing model loading...\n";
    
    InferenceEngine engine;
    InferenceEngine::Config config;
    config.param_path = opts.param_path;
    config.bin_path = opts.bin_path;
    config.num_threads = NCNN_NUM_THREADS;
    config.use_fp16 = true;
    
    ErrorCode err = engine.initialize(config);
    if (err != ErrorCode::SUCCESS) {
        std::cerr << "Failed to load model: " << error_to_string(err) << "\n";
        return 1;
    }
    
    std::cout << "Model loaded successfully\n";
    return 0;
}

int test_single_inference(const Options& opts) {
    std::cout << "Testing single frame inference...\n";
    
    // Initialize engine
    InferenceEngine engine;
    InferenceEngine::Config config;
    config.param_path = opts.param_path;
    config.bin_path = opts.bin_path;
    config.num_threads = NCNN_NUM_THREADS;
    config.use_fp16 = true;
    
    ErrorCode err = engine.initialize(config);
    if (err != ErrorCode::SUCCESS) {
        std::cerr << "Failed to load model: " << error_to_string(err) << "\n";
        return 1;
    }
    
    // Warmup
    std::cout << "Warming up...\n";
    engine.warmup(3);
    
    // Create test input (all zeros)
    AlignedPtr<__fp16> input = make_aligned_buffer<__fp16>(MODEL_INPUT_SIZE);
    memset(input.get(), 0, MODEL_INPUT_SIZE * sizeof(__fp16));
    
    // Run inference
    DetectionResult result;
    int64_t inference_time;
    
    {
        ScopedTimer timer(inference_time);
        err = engine.infer(input.get(), result);
    }
    
    if (err != ErrorCode::SUCCESS) {
        std::cerr << "Inference failed: " << error_to_string(err) << "\n";
        return 1;
    }
    
    std::cout << "Inference completed in " << inference_time << " us\n";
    std::cout << "Detections: " << result.count << "\n";
    
    return 0;
}

int test_camera_capture(const Options& opts) {
    std::cout << "Testing camera capture from " << opts.device << "...\n";
    
    InputPipeline pipeline;
    InputPipeline::Config config;
    config.source = InputSource::CAMERA_V4L2;
    config.device_path = opts.device;
    config.width = INPUT_WIDTH;
    config.height = INPUT_HEIGHT;
    config.fps = 30;
    
    ErrorCode err = pipeline.initialize(config);
    if (err != ErrorCode::SUCCESS) {
        std::cerr << "Failed to initialize camera: " << error_to_string(err) << "\n";
        return 1;
    }
    
    int frames_captured = 0;
    int target_frames = opts.frames > 0 ? opts.frames : 30;
    
    err = pipeline.start([&](const FrameBuffer& frame) -> bool {
        frames_captured++;
        std::cout << "Frame " << frames_captured << ": " 
                  << frame.width << "x" << frame.height 
                  << ", " << frame.size << " bytes\n";
        return frames_captured < target_frames && g_running.load();
    });
    
    if (err != ErrorCode::SUCCESS) {
        std::cerr << "Camera capture failed: " << error_to_string(err) << "\n";
        return 1;
    }
    
    std::cout << "Captured " << frames_captured << " frames\n";
    return 0;
}

// ============================================================================
// Main Processing Loop
// ============================================================================

int run_inference_pipeline(const Options& opts) {
    std::cout << "Starting OPTIMIZED inference pipeline...\n";
    std::cout << "Mode: " << opts.mode << "\n";
    std::cout << "Device/File: " << opts.device << "\n";
    if (!opts.output_video.empty()) {
        std::cout << "Output video: " << opts.output_video << "\n";
    }
    
    // Initialize preprocessing buffers ONCE
    neon::init_preprocess_buffers();
    
    // Initialize inference engine
    InferenceEngine engine;
    InferenceEngine::Config engine_config;
    engine_config.param_path = opts.param_path;
    engine_config.bin_path = opts.bin_path;
    
    // Get thread count from environment or use default
    const char* omp_threads = getenv("OMP_NUM_THREADS");
    if (omp_threads) {
        engine_config.num_threads = std::atoi(omp_threads);
    } else {
        engine_config.num_threads = NCNN_NUM_THREADS;
    }
    
    engine_config.use_fp16 = true;  // Internal NCNN FP16, but we feed FP32
    
    ErrorCode err = engine.initialize(engine_config);
    if (err != ErrorCode::SUCCESS) {
        std::cerr << "Failed to load model: " << error_to_string(err) << "\n";
        neon::cleanup_preprocess_buffers();
        return 1;
    }
    
    std::cout << "Model loaded, warming up (NCNN threads=" << engine_config.num_threads << ")...\n";
    engine.warmup(30);  // Extended warmup for JIT/cache priming and stability
    
    // Initialize input pipeline
    InputPipeline pipeline;
    InputPipeline::Config input_config;
    
    if (opts.mode == "camera") {
        input_config.source = InputSource::CAMERA_V4L2;
        input_config.device_path = opts.device;
    } else {
        input_config.source = InputSource::VIDEO_FILE;
        input_config.device_path = opts.device;
        input_config.loop_video = opts.output_video.empty();  // Don't loop if saving video
    }
    
    input_config.width = INPUT_WIDTH;
    input_config.height = INPUT_HEIGHT;
    input_config.fps = 30;
    
    err = pipeline.initialize(input_config);
    if (err != ErrorCode::SUCCESS) {
        std::cerr << "Failed to initialize input: " << error_to_string(err) << "\n";
        neon::cleanup_preprocess_buffers();
        return 1;
    }
    
    // Initialize async video writer (if output video specified)
    std::unique_ptr<AsyncVideoWriter> video_writer;
    int video_width = INPUT_WIDTH;   // Use pipeline resolution (already resized)
    int video_height = INPUT_HEIGHT;
    
    if (!opts.output_video.empty()) {
        video_writer = std::make_unique<AsyncVideoWriter>();
        AsyncVideoWriter::Config writer_config;
        writer_config.output_path = opts.output_video;
        writer_config.width = video_width;
        writer_config.height = video_height;
        writer_config.fps = pipeline.get_video_fps();  // Match input video FPS
        // Queue size: buffer all frames (inference faster than encoding)
        // For typical videos: ~30fps * 60sec = 1800 frames max (~500MB)
        writer_config.queue_size = 2000;
        writer_config.draw_fps = opts.show_fps;
        
        if (!video_writer->start(writer_config)) {
            std::cerr << "Failed to initialize video writer\n";
            neon::cleanup_preprocess_buffers();
            return 1;
        }
        
        std::cout << "Video writer initialized: " << video_width << "x" << video_height << "\n";
    }
    
    // Initialize async display (if enabled)
    std::unique_ptr<AsyncDisplay> display;
    if (opts.display_enabled) {
        display = std::make_unique<AsyncDisplay>();
        AsyncDisplay::Config display_config;
        display_config.window_name = "YOLOv8n Detection";
        display_config.queue_size = 3;  // Very small for low latency
        display_config.draw_fps = opts.show_fps;
        display_config.draw_bbox = true;
        display_config.max_screen_ratio = 0.75f;  // 75% of screen max
        
        if (!display->start(display_config, INPUT_WIDTH, INPUT_HEIGHT)) {
            std::cerr << "Failed to initialize display\n";
            neon::cleanup_preprocess_buffers();
            return 1;
        }
    }
    
    // Initialize benchmark
    Benchmark benchmark;
    BenchmarkConfig bench_config;
    bench_config.warmup_frames = opts.warmup_frames;
    bench_config.test_frames = opts.frames;
    bench_config.verbose = opts.verbose;
    benchmark.configure(bench_config);
    
    // Allocate FP32 model input buffer (pre-allocated, reused)
    AlignedPtr<float> model_input = make_aligned_buffer<float>(MODEL_INPUT_FLOATS);
    if (!model_input) {
        std::cerr << "Failed to allocate input buffer\n";
        neon::cleanup_preprocess_buffers();
        return 1;
    }
    
    std::cout << "Starting frame processing...\n";
    std::cout << "Warmup: " << opts.warmup_frames << " frames\n";
    std::cout << "Test: " << opts.frames << " frames\n\n";
    
    // Track FPS for overlay
    float rolling_fps = 0;
    float rolling_inference_ms = 0;
    
    // Frame processing callback
    auto process_frame = [&](const FrameBuffer& frame) -> bool {
        FrameTiming timing;
        timing.frame_index = frame.frame_index;
        
        int64_t total_start = get_timestamp_ns();
        
        timing.capture_time_us = 0;
        
        // Preprocessing - dispatch based on pixel format
        float scale;
        int pad_x, pad_y;
        int64_t preprocess_time = 0;
        
        {
            ScopedTimer timer(preprocess_time);
            
            if (frame.format == PixelFormat::BGR) {
                // OPTIMIZED: Direct BGR path (video files)
                neon::preprocess_bgr_direct(
                    frame.data,
                    model_input.get(),
                    frame.width,
                    frame.height,
                    frame.stride,
                    &scale, &pad_x, &pad_y
                );
            } else {
                // Camera path (YUYV)
                neon::preprocess_yuyv_to_fp32(
                    frame.data,
                    model_input.get(),
                    &scale, &pad_x, &pad_y
                );
            }
        }
        timing.preprocess_time_us = preprocess_time;
        
        // Set letterbox params for coordinate mapping
        engine.set_letterbox_params(scale, pad_x, pad_y);
        
        // Inference - DIRECT FP32
        DetectionResult result;
        int64_t inference_time = 0;
        
        {
            ScopedTimer timer(inference_time);
            
            ErrorCode err = engine.infer_fp32(model_input.get(), result);
            if (err != ErrorCode::SUCCESS) {
                std::cerr << "Inference failed on frame " << frame.frame_index << "\n";
            }
        }
        timing.inference_time_us = inference_time;
        
        timing.postprocess_time_us = 0;
        
        // Filter detections by class if specified
        if (!opts.class_filter.empty()) {
            int write_idx = 0;
            for (int i = 0; i < result.count; i++) {
                if (opts.class_filter.count(result.detections[i].class_id) > 0) {
                    if (write_idx != i) {
                        result.detections[write_idx] = result.detections[i];
                    }
                    write_idx++;
                }
            }
            result.count = write_idx;
        }
        
        // Total time
        int64_t total_end = get_timestamp_ns();
        timing.total_time_us = (total_end - total_start) / 1000;
        timing.detection_count = result.count;
        
        // Update rolling stats for FPS overlay
        float current_fps = 1000000.0f / timing.total_time_us;
        float current_inference_ms = inference_time / 1000.0f;
        rolling_fps = rolling_fps * 0.9f + current_fps * 0.1f;
        rolling_inference_ms = rolling_inference_ms * 0.9f + current_inference_ms * 0.1f;
        
        // Push frame to async writer (non-blocking, done AFTER inference)
        if (video_writer && frame.format == PixelFormat::BGR) {
            // Create cv::Mat wrapper (no copy, just wrap existing data)
            cv::Mat bgr_frame(frame.height, frame.width, CV_8UC3, frame.data, frame.stride);
            
            // Push to async queue (bbox drawing happens in writer thread)
            video_writer->push(bgr_frame, result, frame.width, frame.height,
                             rolling_fps, rolling_inference_ms);
        }
        
        // Push frame to async display (non-blocking)
        if (display && frame.format == PixelFormat::BGR) {
            cv::Mat bgr_frame(frame.height, frame.width, CV_8UC3, frame.data, frame.stride);
            display->push(bgr_frame, result, frame.width, frame.height,
                         rolling_fps, rolling_inference_ms);
        }
        
        // Record timing
        benchmark.record_frame(timing);
        
        // Progress indicator
        if (!opts.verbose && benchmark.current_frame() % 100 == 0) {
            std::cout << "Frame " << benchmark.current_frame() 
                      << " | " << timing.total_time_us << "us"
                      << " | " << current_fps << " FPS";
            if (video_writer) {
                std::cout << " | Queue: " << video_writer->queue_size();
            }
            std::cout << "\r" << std::flush;
        }
        
        return !benchmark.is_complete() && g_running.load();
    };
    
    // Run pipeline
    err = pipeline.start(process_frame);
    
    std::cout << "\n";
    
    // Stop video writer (flushes remaining frames)
    if (video_writer) {
        std::cout << "Flushing video writer...\n";
        video_writer->stop();
        std::cout << "Video saved: " << opts.output_video << "\n";
        std::cout << "  Frames written: " << video_writer->frames_written() << "\n";
        std::cout << "  Frames dropped: " << video_writer->frames_dropped() << "\n";
    }
    
    // Stop async display
    if (display) {
        display->stop();
    }
    
    // Print results
    benchmark.print_summary();
    
    // Export CSV if requested
    if (!opts.output_csv.empty()) {
        benchmark.export_csv(opts.output_csv);
        std::cout << "Results exported to " << opts.output_csv << "\n";
    }
    
    // Cleanup
    neon::cleanup_preprocess_buffers();
    
    // Determine exit code based on validation
    BenchmarkStats stats = benchmark.calculate_stats();
    
    if (stats.is_valid()) {
        std::cout << "\n✓ SYSTEM MEETS ALL PERFORMANCE REQUIREMENTS\n";
        std::cout << "  FPS (P99): " << stats.fps_p99 << " >= 20\n";
        return 0;
    } else {
        std::cout << "\n✗ SYSTEM DOES NOT MEET PERFORMANCE REQUIREMENTS\n";
        std::cout << "  FPS (P99): " << stats.fps_p99 << " < 20\n";
        return 1;
    }
}

// ============================================================================
// Main Entry Point
// ============================================================================

int main(int argc, char* argv[]) {
    // Set up signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    // Parse options
    Options opts;
    if (!parse_options(argc, argv, opts)) {
        return 1;
    }
    
    // Run appropriate test/mode
    if (opts.test_model) {
        return test_model_loading(opts);
    }
    
    if (opts.test_inference) {
        return test_single_inference(opts);
    }
    
    if (opts.test_camera) {
        return test_camera_capture(opts);
    }
    
    // Run full pipeline
    return run_inference_pipeline(opts);
}
