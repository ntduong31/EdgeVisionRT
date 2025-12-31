/**
 * @file benchmark.h
 * @brief Benchmark and testing framework
 * 
 * Testing, Benchmark & Truth Agent Design:
 * - Video test: measures pure throughput (preprocessing + inference)
 * - Camera test: measures latency and stability
 * - Results must pass both tests for system validation
 * - Statistical analysis with percentile reporting
 */

#ifndef YOLO_BENCHMARK_H
#define YOLO_BENCHMARK_H

#include "common.h"
#include <vector>
#include <string>
#include <fstream>

namespace yolo {

/**
 * @brief Benchmark configuration
 */
struct BenchmarkConfig {
    int warmup_frames = 30;         // Frames to skip for warmup
    int test_frames = 1000;         // Number of frames to measure
    std::string output_path;        // Path for result CSV
    bool verbose = false;           // Print per-frame results
    bool check_memory = true;       // Track peak memory usage
};

/**
 * @brief Single frame timing result
 */
struct FrameTiming {
    uint32_t frame_index;
    int64_t capture_time_us;        // Time to acquire frame
    int64_t preprocess_time_us;     // YUYV->RGB->Resize->Normalize
    int64_t inference_time_us;      // NCNN forward pass
    int64_t postprocess_time_us;    // Decode + NMS
    int64_t total_time_us;          // End-to-end latency
    int detection_count;            // Number of detections
};

/**
 * @brief Aggregate benchmark statistics
 */
struct BenchmarkStats {
    // Frame timing stats (microseconds)
    double mean_total_us;
    double std_total_us;
    double min_total_us;
    double max_total_us;
    double p50_total_us;    // Median
    double p90_total_us;
    double p95_total_us;
    double p99_total_us;
    
    // Component breakdown
    double mean_capture_us;
    double mean_preprocess_us;
    double mean_inference_us;
    double mean_postprocess_us;
    
    // Derived metrics
    double fps_mean;
    double fps_p50;
    double fps_p99;
    
    // Frame analysis
    int total_frames;
    int dropped_frames;
    int frames_over_50ms;   // Frames exceeding realtime budget
    
    // System metrics
    size_t peak_memory_kb;
    double cpu_utilization;
    
    // Pass/fail criteria
    bool meets_20fps_requirement() const { return fps_p99 >= 20.0; }
    bool meets_latency_requirement() const { return p99_total_us <= 50000; }
    bool meets_jitter_requirement() const { return std_total_us <= 5000; }
    bool is_valid() const {
        return meets_20fps_requirement() && 
               meets_latency_requirement() && 
               meets_jitter_requirement();
    }
};

/**
 * @brief Benchmark framework for performance measurement
 */
class Benchmark {
public:
    Benchmark();
    ~Benchmark();

    /**
     * @brief Initialize benchmark with configuration
     */
    void configure(const BenchmarkConfig& config);

    /**
     * @brief Reset all measurements
     */
    void reset();

    /**
     * @brief Record a frame timing
     */
    void record_frame(const FrameTiming& timing);

    /**
     * @brief Calculate aggregate statistics
     */
    BenchmarkStats calculate_stats() const;

    /**
     * @brief Export results to CSV
     */
    void export_csv(const std::string& path) const;

    /**
     * @brief Print summary to stdout
     */
    void print_summary() const;

    /**
     * @brief Get raw timing data
     */
    const std::vector<FrameTiming>& timings() const { return timings_; }

    /**
     * @brief Check if warmup phase is complete
     */
    bool warmup_complete() const { 
        return current_frame_ >= config_.warmup_frames; 
    }

    /**
     * @brief Check if benchmark is complete
     * If test_frames is 0, never complete (run until video ends)
     */
    bool is_complete() const {
        if (config_.test_frames == 0) return false;  // Run forever until video ends
        return current_frame_ >= config_.warmup_frames + config_.test_frames;
    }

    /**
     * @brief Get current frame index
     */
    int current_frame() const { return current_frame_; }

private:
    BenchmarkConfig config_;
    std::vector<FrameTiming> timings_;
    int current_frame_ = 0;
    
    // Memory tracking
    size_t initial_memory_kb_ = 0;
    size_t peak_memory_kb_ = 0;
    
    void update_memory_stats();
};

/**
 * @brief Get current process memory usage
 * @return RSS in kilobytes
 */
size_t get_memory_usage_kb();

/**
 * @brief Calculate percentile from sorted array
 * @param values Sorted array of values
 * @param count Number of values
 * @param percentile Percentile (0-100)
 * @return Value at percentile
 */
double calculate_percentile(const double* values, int count, double percentile);

/**
 * @brief Format benchmark result as table string
 */
std::string format_stats_table(const BenchmarkStats& stats);

/**
 * @brief Compare two benchmark runs
 * @param baseline Previous run statistics
 * @param current Current run statistics
 * @return Formatted comparison string
 */
std::string compare_benchmarks(const BenchmarkStats& baseline, const BenchmarkStats& current);

}  // namespace yolo

#endif  // YOLO_BENCHMARK_H
