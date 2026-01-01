#!/bin/bash
#
# Quick Run Script for EdgeVision RT
#
# Usage:
#   ./run.sh                           # Run basic benchmark (FP32)
#   ./run.sh int8                      # Run with INT8 quantization (faster!)
#   ./run.sh vulkan                    # Run with Vulkan GPU
#   ./run.sh int8 vulkan               # INT8 + Vulkan (fastest!)
#   ./run.sh display                   # Show display window
#   ./run.sh class person,car          # Filter classes
#   ./run.sh display class person      # Display + filter
#   ./run.sh video output.mp4          # Save to video
#   ./run.sh video out.mp4 class car   # Video + filter
#   ./run.sh int8 display class person # INT8 + display + filter
#   ./run.sh cam                       # Use webcam (/dev/video0)
#   ./run.sh cam display               # Webcam + display
#   ./run.sh cam fb class person       # Webcam + framebuffer + filter
#   ./run.sh all                       # Display + video + all classes
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
BINARY="${BUILD_DIR}/yolo_inference"
VIDEO="${SCRIPT_DIR}/tests/human.mp4"

# Default: FP32 model
PARAM="${SCRIPT_DIR}/models/yolov8n_ncnn_model/model.ncnn.param"
BIN="${SCRIPT_DIR}/models/yolov8n_ncnn_model/model.ncnn.bin"

# INT8 model paths
PARAM_INT8="${SCRIPT_DIR}/models/yolov8n_ncnn_model/model.int8.param"
BIN_INT8="${SCRIPT_DIR}/models/yolov8n_ncnn_model/model.int8.bin"

# FP16 model paths
PARAM_FP16="${SCRIPT_DIR}/models/yolov8n_ncnn_model/model.fp16.param"
BIN_FP16="${SCRIPT_DIR}/models/yolov8n_ncnn_model/model.fp16.bin"

# Check if binary exists
if [ ! -f "${BINARY}" ]; then
    echo "Error: Binary not found. Run ./build.sh first"
    exit 1
fi

# Default frames count
FRAMES=2000

# Flags for tracking options
ENABLE_DISPLAY=false
ENABLE_FB=false
ENABLE_VIDEO=false
ENABLE_CAM=false
ENABLE_CLASS_FILTER=false
ENABLE_INT8=false
ENABLE_FP16=false
ENABLE_VULKAN=false
VIDEO_OUTPUT=""
CLASS_FILTER=""
CAMERA_DEVICE="/dev/video0"

# Parse all arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        help|--help|-h)
            echo "EdgeVision RT - Quick Run Script"
            echo ""
            echo "Usage:"
            echo "  ./run.sh                           # Basic benchmark FP32 (video file)"
            echo "  ./run.sh cam                       # Use webcam (/dev/video0)"
            echo "  ./run.sh cam display               # Webcam + display"
            echo "  ./run.sh cam fb class person       # Webcam + framebuffer + filter"
            echo "  ./run.sh display                   # Show display window (video file)"
            echo "  ./run.sh fb                        # Framebuffer display (no X11)"
            echo "  ./run.sh class person,car          # Filter specific classes"
            echo "  ./run.sh video output.mp4          # Save video output"
            echo ""
            echo "Acceleration options:"
            echo "  fp16                               # Use FP16 optimized model"
            echo "  int8                               # Use INT8 quantized model"
            echo "  vulkan                             # Use Vulkan GPU compute"
            echo ""
            echo "Display options:"
            echo "  display                            # OpenCV display (X11)"
            echo "  fb                                 # Framebuffer display (no X11, faster!)"
            echo ""
            echo "Examples:"
            echo "  ./run.sh cam display class person  # Webcam + display + filter"
            echo "  ./run.sh cam fb                    # Webcam + framebuffer (max FPS!)"
            echo "  ./run.sh display class person      # Video file + display + filter"
            echo ""
            exit 0
            ;;
        int8)
            ENABLE_INT8=true
            shift
            ;;
        fp16)
            ENABLE_FP16=true
            shift
            ;;
        vulkan)
            ENABLE_VULKAN=true
            shift
            ;;
        display)
            ENABLE_DISPLAY=true
            shift
            ;;
        cam|camera|webcam)
            ENABLE_CAM=true
            # Check if next arg is a device path
            if [[ $# -gt 1 && "$2" == /dev/* ]]; then
                CAMERA_DEVICE="$2"
                shift
            fi
            shift
            ;;
        fb)
            ENABLE_FB=true
            shift
            ;;
        video)
            ENABLE_VIDEO=true
            VIDEO_OUTPUT="${2:-output.mp4}"
            shift 2
            ;;
        class)
            ENABLE_CLASS_FILTER=true
            CLASS_FILTER="${2:-person}"
            shift 2
            ;;
        all)
            ENABLE_DISPLAY=true
            ENABLE_VIDEO=true
            VIDEO_OUTPUT="${2:-output_all.mp4}"
            shift
            if [[ $# -gt 0 && "$1" != "class" && "$1" != "display" ]]; then
                shift  # Skip optional video filename
            fi
            ;;
        benchmark|"")
            # Default mode, do nothing
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run './run.sh help' for usage"
            exit 1
            ;;
    esac
done

# Select model based on precision flags
if [ "$ENABLE_INT8" = true ]; then
    if [ -f "${PARAM_INT8}" ] && [ -f "${BIN_INT8}" ]; then
        PARAM="${PARAM_INT8}"
        BIN="${BIN_INT8}"
    else
        echo "Warning: INT8 model not found, falling back to FP32"
        ENABLE_INT8=false
    fi
elif [ "$ENABLE_FP16" = true ]; then
    if [ -f "${PARAM_FP16}" ] && [ -f "${BIN_FP16}" ]; then
        PARAM="${PARAM_FP16}"
        BIN="${BIN_FP16}"
    else
        echo "Warning: FP16 model not found, falling back to FP32"
        ENABLE_FP16=false
    fi
fi

# Check if model exists
if [ ! -f "${PARAM}" ] || [ ! -f "${BIN}" ]; then
    echo "Error: Model files not found"
    exit 1
fi

# Build base args - camera or video mode
if [ "$ENABLE_CAM" = true ]; then
    ARGS=(
        "--camera" "${CAMERA_DEVICE}"
        "--param" "${PARAM}"
        "--bin" "${BIN}"
    )
    MODE_DESC="CAM"
    # Camera mode: run indefinitely (frames=0) or limited by --frames
    if [ "$FRAMES" -eq 2000 ]; then
        FRAMES=0  # Run indefinitely for camera
    fi
else
    ARGS=(
        "--video" "${VIDEO}"
        "--param" "${PARAM}"
        "--bin" "${BIN}"
    )
    MODE_DESC="FP32"
fi

# Add INT8 flag
if [ "$ENABLE_INT8" = true ]; then
    ARGS+=("--int8")
    MODE_DESC="INT8"
elif [ "$ENABLE_FP16" = true ]; then
    MODE_DESC="FP16"
fi

# Add Vulkan flag
if [ "$ENABLE_VULKAN" = true ]; then
    ARGS+=("--vulkan")
    MODE_DESC="${MODE_DESC}+Vulkan"
fi

if [ "$ENABLE_DISPLAY" = true ]; then
    export DISPLAY=:0
    ARGS+=("--display")
    MODE_DESC="${MODE_DESC}+Display"
fi

# Framebuffer display - bypasses X11 for maximum FPS
if [ "$ENABLE_FB" = true ]; then
    ARGS+=("--fb")
    MODE_DESC="${MODE_DESC}+Framebuffer"
    # Check framebuffer access
    if [ ! -w /dev/fb0 ]; then
        echo "Note: Need framebuffer access. Run: sudo chmod 666 /dev/fb0"
    fi
fi

if [ "$ENABLE_VIDEO" = true ]; then
    ARGS+=("--output-video" "${VIDEO_OUTPUT}")
    FRAMES=0  # Set frames to 0 for full video
    MODE_DESC="${MODE_DESC}+Video"
fi

if [ "$ENABLE_CLASS_FILTER" = true ]; then
    ARGS+=("--class" "${CLASS_FILTER}")
    MODE_DESC="${MODE_DESC}+Filter:${CLASS_FILTER}"
fi

# Add frames argument at the end
ARGS+=("--frames" "${FRAMES}")

# Print configuration
echo "Mode: ${MODE_DESC}"
if [ "$ENABLE_VIDEO" = true ]; then
    echo "Video output: ${VIDEO_OUTPUT}"
fi
if [ "$ENABLE_CLASS_FILTER" = true ]; then
    echo "Class filter: ${CLASS_FILTER}"
fi

# OPTIMIZATION: Set CPU governor to performance
if [ -w /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]; then
    echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null 2>&1 || true
fi

# Thread configuration
# When display is enabled, let main.cpp auto-select optimal threads (3)
# When no display, use 4 threads for max throughput
if [ "$ENABLE_DISPLAY" = true ]; then
    # Don't set OMP_NUM_THREADS - let main.cpp auto-optimize to 3 threads
    unset OMP_NUM_THREADS
    echo "Threads: auto (3 for display mode)"
else
    if [ -z "${OMP_NUM_THREADS}" ]; then
        export OMP_NUM_THREADS=4
    fi
    echo "Threads: ${OMP_NUM_THREADS}"
fi
echo ""

# Debug: show command
# echo "Command: ${BINARY} ${ARGS[@]}"

# Run (stay in current dir for relative paths in output)
exec "${BINARY}" "${ARGS[@]}"
