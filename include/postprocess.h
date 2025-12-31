/**
 * @file postprocess.h
 * @brief YOLOv8 output post-processing with NMS
 * 
 * Handles decoding of YOLOv8 output format and non-maximum suppression.
 */

#ifndef YOLO_POSTPROCESS_H
#define YOLO_POSTPROCESS_H

#include "common.h"

namespace yolo {

/**
 * @brief Decode YOLOv8 raw output to detection boxes
 * 
 * YOLOv8 output format: [batch, 84, 8400]
 * - 84 = 4 (box coords) + 80 (class scores)
 * - 8400 = number of anchor points
 * 
 * @param output Raw model output (transposed to [8400, 84])
 * @param num_outputs Number of output anchors (8400)
 * @param conf_threshold Confidence threshold for filtering
 * @param detections Output detection array
 * @param max_detections Maximum detections to return
 * @return Number of detections found
 */
int decode_yolov8_output(
    const float* output,
    int num_outputs,
    float conf_threshold,
    Detection* detections,
    int max_detections
);

/**
 * @brief Apply Non-Maximum Suppression to detections
 * 
 * Uses vectorized IoU calculation for efficiency.
 * 
 * @param detections Detection array (modified in place)
 * @param count Number of detections (updated after NMS)
 * @param nms_threshold IoU threshold for suppression
 */
void nms_sorted(
    Detection* detections,
    int& count,
    float nms_threshold
);

/**
 * @brief Sort detections by confidence (descending)
 * 
 * Uses insertion sort for small arrays, quicksort for larger.
 * 
 * @param detections Detection array
 * @param count Number of detections
 */
void sort_detections_by_confidence(
    Detection* detections,
    int count
);

/**
 * @brief Calculate IoU between two detections
 * 
 * @param a First detection
 * @param b Second detection
 * @return Intersection over Union value
 */
inline float calculate_iou(const Detection& a, const Detection& b) {
    float x1 = std::max(a.x1, b.x1);
    float y1 = std::max(a.y1, b.y1);
    float x2 = std::min(a.x2, b.x2);
    float y2 = std::min(a.y2, b.y2);
    
    float inter_width = std::max(0.0f, x2 - x1);
    float inter_height = std::max(0.0f, y2 - y1);
    float inter_area = inter_width * inter_height;
    
    float union_area = a.area() + b.area() - inter_area;
    
    return (union_area > 0) ? (inter_area / union_area) : 0.0f;
}

/**
 * @brief Map detection coordinates from model space to original image space
 * 
 * Accounts for letterbox padding and scaling.
 * 
 * @param det Detection to transform (modified in place)
 * @param scale Scale factor used in letterbox
 * @param pad_x Horizontal padding
 * @param pad_y Vertical padding
 * @param orig_width Original image width
 * @param orig_height Original image height
 */
void map_detection_to_original(
    Detection& det,
    float scale,
    int pad_x,
    int pad_y,
    int orig_width,
    int orig_height
);

/**
 * @brief NEON-optimized batch IoU calculation
 * 
 * Calculates IoU between one detection and an array of detections.
 * 
 * @param reference Reference detection
 * @param candidates Array of candidate detections
 * @param count Number of candidates
 * @param iou_out Output IoU values
 */
void batch_iou_neon(
    const Detection& reference,
    const Detection* candidates,
    int count,
    float* iou_out
);

}  // namespace yolo

#endif  // YOLO_POSTPROCESS_H
