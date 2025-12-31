#!/bin/bash
#
# Quick Build Script for YOLOv8n Inference System
#
# Usage:
#   ./build.sh          # Standard build
#   ./build.sh clean    # Clean and rebuild
#   ./build.sh debug    # Debug build
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
DEPS_DIR="${SCRIPT_DIR}/deps"
NCNN_INSTALL="${DEPS_DIR}/ncnn-install"

# Parse arguments
BUILD_TYPE="Release"
CLEAN=0

for arg in "$@"; do
    case $arg in
        clean)
            CLEAN=1
            ;;
        debug)
            BUILD_TYPE="Debug"
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: $0 [clean] [debug]"
            exit 1
            ;;
    esac
done

# Clean if requested
if [ $CLEAN -eq 1 ]; then
    echo "Cleaning build directory..."
    rm -rf "${BUILD_DIR}"
fi

# Create build directory
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Check for NCNN
if [ -d "${NCNN_INSTALL}" ]; then
    NCNN_DIR="${NCNN_INSTALL}/lib/cmake/ncnn"
else
    # Try system NCNN
    NCNN_DIR=""
fi

echo "================================="
echo "YOLOv8n Inference System Build"
echo "================================="
echo "Build type: ${BUILD_TYPE}"
echo "Build dir:  ${BUILD_DIR}"
if [ -n "${NCNN_DIR}" ]; then
    echo "NCNN dir:   ${NCNN_DIR}"
fi
echo ""

# Configure
CMAKE_ARGS=(
    "-DCMAKE_BUILD_TYPE=${BUILD_TYPE}"
)

if [ -n "${NCNN_DIR}" ]; then
    CMAKE_ARGS+=("-DNCNN_DIR=${NCNN_DIR}")
fi

echo "Configuring..."
cmake "${CMAKE_ARGS[@]}" ..

# Build
echo ""
echo "Building..."
make -j$(nproc)

echo ""
echo "================================="
echo "Build complete!"
echo "Binary: ${BUILD_DIR}/yolo_inference"
echo "================================="

# Print usage
echo ""
echo "Usage examples:"
echo "  # Test model loading:"
echo "  ./yolo_inference --test-model --param ../models/yolov8n_fp16.param --bin ../models/yolov8n_fp16.bin"
echo ""
echo "  # Run benchmark with video:"
echo "  ./yolo_inference --video ../tests/test_video/benchmark.mp4 --param ../models/yolov8n_fp16.param --bin ../models/yolov8n_fp16.bin --frames 1000"
echo ""
echo "  # Run with camera:"
echo "  ./yolo_inference --camera /dev/video0 --param ../models/yolov8n_fp16.param --bin ../models/yolov8n_fp16.bin"
