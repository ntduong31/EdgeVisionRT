cmake_minimum_required(VERSION 3.16)
project(yolo_inference VERSION 1.0.0 LANGUAGES C CXX)

# ============================================================================
# Build Configuration for Raspberry Pi 5 (Cortex-A76)
# ============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optimization flags for Cortex-A76
set(CORTEX_A76_FLAGS "-march=armv8.2-a+fp16+dotprod -mtune=cortex-a76")

# Release build flags
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${CORTEX_A76_FLAGS} -O3 -ffast-math -flto -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${CORTEX_A76_FLAGS} -O3 -ffast-math -flto -DNDEBUG -fno-rtti -fno-exceptions")

# Debug flags
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} ${CORTEX_A76_FLAGS} -O0 -g -DDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${CORTEX_A76_FLAGS} -O0 -g -DDEBUG")

# Default to Release build
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CXX flags: ${CMAKE_CXX_FLAGS_RELEASE}")

# ============================================================================
# Find Dependencies
# ============================================================================

# NCNN
if(NCNN_DIR)
    find_package(ncnn REQUIRED PATHS ${NCNN_DIR})
else()
    # Try system-wide installation
    find_package(ncnn REQUIRED)
endif()

if(ncnn_FOUND)
    message(STATUS "NCNN found: ${ncnn_DIR}")
else()
    message(FATAL_ERROR "NCNN not found. Please set -DNCNN_DIR=/path/to/ncnn/lib/cmake/ncnn")
endif()

# OpenCV (for video file reading)
# Try find_package first, fallback to manual configuration
find_package(OpenCV QUIET COMPONENTS core imgproc videoio)
if(OpenCV_FOUND)
    message(STATUS "OpenCV found via CMake: ${OpenCV_VERSION}")
else()
    # Manual OpenCV configuration for Debian/Raspbian
    message(STATUS "OpenCV CMake config not found, using manual configuration")
    
    # Find headers
    find_path(OpenCV_INCLUDE_DIRS opencv4/opencv2/opencv.hpp
        PATHS /usr/include /usr/local/include
    )
    if(OpenCV_INCLUDE_DIRS)
        set(OpenCV_INCLUDE_DIRS "${OpenCV_INCLUDE_DIRS}/opencv4")
    else()
        find_path(OpenCV_INCLUDE_DIRS opencv2/opencv.hpp
            PATHS /usr/include /usr/local/include
        )
    endif()
    
    if(NOT OpenCV_INCLUDE_DIRS)
        message(FATAL_ERROR "OpenCV headers not found")
    endif()
    
    # Find libraries
    find_library(OpenCV_CORE_LIB opencv_core PATHS /usr/lib/aarch64-linux-gnu /usr/lib)
    find_library(OpenCV_IMGPROC_LIB opencv_imgproc PATHS /usr/lib/aarch64-linux-gnu /usr/lib)
    find_library(OpenCV_VIDEOIO_LIB opencv_videoio PATHS /usr/lib/aarch64-linux-gnu /usr/lib)
    find_library(OpenCV_HIGHGUI_LIB opencv_highgui PATHS /usr/lib/aarch64-linux-gnu /usr/lib)
    
    if(NOT OpenCV_CORE_LIB OR NOT OpenCV_IMGPROC_LIB OR NOT OpenCV_VIDEOIO_LIB)
        message(FATAL_ERROR "OpenCV libraries not found. Install: libopencv-core-dev libopencv-imgproc-dev libopencv-videoio-dev")
    endif()
    
    set(OpenCV_LIBS ${OpenCV_CORE_LIB} ${OpenCV_IMGPROC_LIB} ${OpenCV_VIDEOIO_LIB})
    
    # Add highgui if available (for display feature)
    # Try standard name first, then versioned name
    find_library(OpenCV_HIGHGUI_LIB 
        NAMES opencv_highgui opencv_highgui.so.406
        PATHS /usr/lib/aarch64-linux-gnu /usr/lib
        NO_DEFAULT_PATH
    )
    if(NOT OpenCV_HIGHGUI_LIB)
        # Direct path fallback
        if(EXISTS "/usr/lib/aarch64-linux-gnu/libopencv_highgui.so.406")
            set(OpenCV_HIGHGUI_LIB "/usr/lib/aarch64-linux-gnu/libopencv_highgui.so.406")
        endif()
    endif()
    
    if(OpenCV_HIGHGUI_LIB)
        list(APPEND OpenCV_LIBS ${OpenCV_HIGHGUI_LIB})
        message(STATUS "OpenCV highgui: ${OpenCV_HIGHGUI_LIB}")
    else()
        message(STATUS "OpenCV highgui not found, display feature may not work")
    endif()
    
    message(STATUS "OpenCV include: ${OpenCV_INCLUDE_DIRS}")
    message(STATUS "OpenCV libs: ${OpenCV_LIBS}")
endif()

# Threads
find_package(Threads REQUIRED)

# ============================================================================
# Source Files
# ============================================================================

set(SOURCES
    src/main.cpp
    src/neon_preprocess.cpp
    src/input_pipeline.cpp
    src/inference_engine.cpp
    src/postprocess.cpp
    src/benchmark.cpp
)

set(HEADERS
    include/common.h
    include/neon_preprocess.h
    include/input_pipeline.h
    include/inference_engine.h
    include/postprocess.h
    include/benchmark.h
)

# ============================================================================
# Build Executable
# ============================================================================

add_executable(yolo_inference ${SOURCES} ${HEADERS})

target_include_directories(yolo_inference PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(yolo_inference PRIVATE
    ncnn
    ${OpenCV_LIBS}
    Threads::Threads
)

# Enable Link-Time Optimization
set_property(TARGET yolo_inference PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)

# ============================================================================
# Install
# ============================================================================

install(TARGETS yolo_inference
    RUNTIME DESTINATION bin
)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/models/
    DESTINATION share/yolo_inference/models
    FILES_MATCHING PATTERN "*.param" PATTERN "*.bin"
)

# ============================================================================
# Summary
# ============================================================================

message(STATUS "")
message(STATUS "YOLOv8n Inference System Configuration:")
message(STATUS "  Build type:      ${CMAKE_BUILD_TYPE}")
message(STATUS "  NCNN:            ${ncnn_DIR}")
message(STATUS "  OpenCV:          ${OpenCV_VERSION}")
message(STATUS "  Install prefix:  ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")
