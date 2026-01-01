#!/bin/bash
#
# Quick Build Script for YOLOv8n Inference System
#
# Usage:
#   ./build.sh          # Standard build with NCNN Vulkan if available
#   ./build.sh clean    # Clean and rebuild
#   ./build.sh debug    # Debug build
#   ./build.sh vulkan   # Force Vulkan NCNN build
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
DEPS_DIR="${SCRIPT_DIR}/deps"

# NCNN installations (prioritize Vulkan version)
NCNN_VULKAN_INSTALL="${DEPS_DIR}/ncnn-vulkan-install"
NCNN_INSTALL="${DEPS_DIR}/ncnn-install"

# Parse arguments
BUILD_TYPE="Release"
CLEAN=0
USE_VULKAN=0

for arg in "$@"; do
    case $arg in
        clean)
            CLEAN=1
            ;;
        debug)
            BUILD_TYPE="Debug"
            ;;
        vulkan)
            USE_VULKAN=1
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: $0 [clean] [debug] [vulkan]"
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

# Check for NCNN (prefer Vulkan version if available)
NCNN_DIR=""
NCNN_TYPE=""

if [ -d "${NCNN_VULKAN_INSTALL}" ]; then
    NCNN_DIR="${NCNN_VULKAN_INSTALL}/lib/cmake/ncnn"
    NCNN_TYPE="Vulkan+INT8"
elif [ -d "${NCNN_INSTALL}" ]; then
    NCNN_DIR="${NCNN_INSTALL}/lib/cmake/ncnn"
    NCNN_TYPE="CPU-only"
fi

echo "================================="
echo "EdgeVision RT Build"
echo "================================="
echo "Build type:   ${BUILD_TYPE}"
echo "NCNN type:    ${NCNN_TYPE}"
echo "Build dir:    ${BUILD_DIR}"
if [ -n "${NCNN_DIR}" ]; then
    echo "NCNN dir:     ${NCNN_DIR}"
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
echo "  # Basic benchmark (FP32):"
echo "  ./run.sh"
echo ""
echo "  # INT8 quantization (2-4x faster!):"
echo "  ./run.sh int8"
echo ""
echo "  # INT8 + Vulkan GPU (fastest!):"
echo "  ./run.sh int8 vulkan"
echo ""
echo "  # With display + class filter:"
echo "  ./run.sh int8 display class person"
echo ""
echo "  # Save video output:"
echo "  ./run.sh int8 video output.mp4"
