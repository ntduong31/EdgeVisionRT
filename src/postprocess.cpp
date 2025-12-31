/**
 * @file postprocess.cpp
 * @brief YOLOv8 output post-processing implementation
 */

#include "postprocess.h"
#include <algorithm>
#include <cmath>

namespace yolo {

// ============================================================================
// Output Decoding
// ============================================================================

int decode_yolov8_output(
    const float* output,
    int num_outputs,
    float conf_threshold,
    Detection* detections,
    int max_detections
) {
    int count = 0;
    
    // YOLOv8 output format (transposed): [8400, 84]
    // Each row: [cx, cy, w, h, class_scores[80]]
    
    for (int i = 0; i < num_outputs && count < max_detections; i++) {
        const float* row = output + i * (4 + NUM_CLASSES);
        
        // Get box coordinates
        float cx = row[0];
        float cy = row[1];
        float w = row[2];
        float h = row[3];
        
        // Find best class
        int best_class = 0;
        float best_score = row[4];
        
        for (int c = 1; c < NUM_CLASSES; c++) {
            if (row[4 + c] > best_score) {
                best_score = row[4 + c];
                best_class = c;
            }
        }
        
        // Check confidence threshold
        if (best_score < conf_threshold) {
            continue;
        }
        
        // Convert to x1, y1, x2, y2 format (normalized to model size)
        float x1 = (cx - w * 0.5f) / MODEL_SIZE;
        float y1 = (cy - h * 0.5f) / MODEL_SIZE;
        float x2 = (cx + w * 0.5f) / MODEL_SIZE;
        float y2 = (cy + h * 0.5f) / MODEL_SIZE;
        
        // Clamp to [0, 1]
        x1 = std::max(0.0f, std::min(1.0f, x1));
        y1 = std::max(0.0f, std::min(1.0f, y1));
        x2 = std::max(0.0f, std::min(1.0f, x2));
        y2 = std::max(0.0f, std::min(1.0f, y2));
        
        // Store detection
        detections[count].x1 = x1;
        detections[count].y1 = y1;
        detections[count].x2 = x2;
        detections[count].y2 = y2;
        detections[count].confidence = best_score;
        detections[count].class_id = best_class;
        
        count++;
    }
    
    return count;
}

// ============================================================================
// Non-Maximum Suppression
// ============================================================================

void sort_detections_by_confidence(Detection* detections, int count) {
    // Use std::sort for small arrays, insertion sort for very small
    if (count <= 8) {
        // Insertion sort
        for (int i = 1; i < count; i++) {
            Detection key = detections[i];
            int j = i - 1;
            while (j >= 0 && detections[j].confidence < key.confidence) {
                detections[j + 1] = detections[j];
                j--;
            }
            detections[j + 1] = key;
        }
    } else {
        std::sort(detections, detections + count,
            [](const Detection& a, const Detection& b) {
                return a.confidence > b.confidence;
            });
    }
}

void nms_sorted(Detection* detections, int& count, float nms_threshold) {
    if (count <= 1) return;
    
    // Mark detections to keep
    bool* keep = new bool[count];
    for (int i = 0; i < count; i++) {
        keep[i] = true;
    }
    
    // For each detection (already sorted by confidence)
    for (int i = 0; i < count; i++) {
        if (!keep[i]) continue;
        
        const Detection& ref = detections[i];
        
        // Suppress lower-confidence detections with high IoU
        for (int j = i + 1; j < count; j++) {
            if (!keep[j]) continue;
            
            // Only suppress same class
            if (detections[j].class_id != ref.class_id) continue;
            
            float iou = calculate_iou(ref, detections[j]);
            if (iou > nms_threshold) {
                keep[j] = false;
            }
        }
    }
    
    // Compact array
    int new_count = 0;
    for (int i = 0; i < count; i++) {
        if (keep[i]) {
            if (new_count != i) {
                detections[new_count] = detections[i];
            }
            new_count++;
        }
    }
    
    count = new_count;
    delete[] keep;
}

// ============================================================================
// Coordinate Mapping
// ============================================================================

void map_detection_to_original(
    Detection& det,
    float scale,
    int pad_x,
    int pad_y,
    int orig_width,
    int orig_height
) {
    // Convert from normalized [0,1] to model pixel coordinates
    float x1_model = det.x1 * MODEL_SIZE;
    float y1_model = det.y1 * MODEL_SIZE;
    float x2_model = det.x2 * MODEL_SIZE;
    float y2_model = det.y2 * MODEL_SIZE;
    
    // Remove padding
    x1_model -= pad_x;
    y1_model -= pad_y;
    x2_model -= pad_x;
    y2_model -= pad_y;
    
    // Unscale to original image coordinates
    float x1_orig = x1_model / scale;
    float y1_orig = y1_model / scale;
    float x2_orig = x2_model / scale;
    float y2_orig = y2_model / scale;
    
    // Normalize back to [0,1] based on original image size
    det.x1 = std::max(0.0f, std::min(1.0f, x1_orig / orig_width));
    det.y1 = std::max(0.0f, std::min(1.0f, y1_orig / orig_height));
    det.x2 = std::max(0.0f, std::min(1.0f, x2_orig / orig_width));
    det.y2 = std::max(0.0f, std::min(1.0f, y2_orig / orig_height));
}

// ============================================================================
// Batch IoU (NEON optimized)
// ============================================================================

void batch_iou_neon(
    const Detection& reference,
    const Detection* candidates,
    int count,
    float* iou_out
) {
    // Reference box coordinates
    float32x4_t ref_x1 = vdupq_n_f32(reference.x1);
    float32x4_t ref_y1 = vdupq_n_f32(reference.y1);
    float32x4_t ref_x2 = vdupq_n_f32(reference.x2);
    float32x4_t ref_y2 = vdupq_n_f32(reference.y2);
    float ref_area = reference.area();
    float32x4_t v_ref_area = vdupq_n_f32(ref_area);
    float32x4_t v_zero = vdupq_n_f32(0.0f);
    
    // Process 4 candidates at a time
    int i = 0;
    for (; i + 4 <= count; i += 4) {
        // Load 4 candidate boxes
        float x1[4], y1[4], x2[4], y2[4], area[4];
        for (int j = 0; j < 4; j++) {
            x1[j] = candidates[i + j].x1;
            y1[j] = candidates[i + j].y1;
            x2[j] = candidates[i + j].x2;
            y2[j] = candidates[i + j].y2;
            area[j] = candidates[i + j].area();
        }
        
        float32x4_t cand_x1 = vld1q_f32(x1);
        float32x4_t cand_y1 = vld1q_f32(y1);
        float32x4_t cand_x2 = vld1q_f32(x2);
        float32x4_t cand_y2 = vld1q_f32(y2);
        float32x4_t cand_area = vld1q_f32(area);
        
        // Calculate intersection
        float32x4_t inter_x1 = vmaxq_f32(ref_x1, cand_x1);
        float32x4_t inter_y1 = vmaxq_f32(ref_y1, cand_y1);
        float32x4_t inter_x2 = vminq_f32(ref_x2, cand_x2);
        float32x4_t inter_y2 = vminq_f32(ref_y2, cand_y2);
        
        float32x4_t inter_w = vmaxq_f32(v_zero, vsubq_f32(inter_x2, inter_x1));
        float32x4_t inter_h = vmaxq_f32(v_zero, vsubq_f32(inter_y2, inter_y1));
        float32x4_t inter_area = vmulq_f32(inter_w, inter_h);
        
        // Calculate union
        float32x4_t union_area = vsubq_f32(vaddq_f32(v_ref_area, cand_area), inter_area);
        
        // Calculate IoU
        float32x4_t iou = vdivq_f32(inter_area, union_area);
        
        // Store results
        vst1q_f32(iou_out + i, iou);
    }
    
    // Handle remaining candidates
    for (; i < count; i++) {
        iou_out[i] = calculate_iou(reference, candidates[i]);
    }
}

}  // namespace yolo
