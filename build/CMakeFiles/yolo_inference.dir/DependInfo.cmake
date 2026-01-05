
# Consider dependencies only in project.
set(CMAKE_DEPENDS_IN_PROJECT_ONLY OFF)

# The set of languages for which implicit dependencies are needed:
set(CMAKE_DEPENDS_LANGUAGES
  "ASM"
  )
# The set of files for implicit dependencies of each language:
set(CMAKE_DEPENDS_CHECK_ASM
  "/home/pi/AI/EdgeVisionRT/src/asm_kernels.S" "/home/pi/AI/EdgeVisionRT/build/CMakeFiles/yolo_inference.dir/src/asm_kernels.S.o"
  )
set(CMAKE_ASM_COMPILER_ID "GNU")

# Preprocessor definitions for this target.
set(CMAKE_TARGET_DEFINITIONS_ASM
  "NCNN_VULKAN=1"
  )

# The include file search paths:
set(CMAKE_ASM_TARGET_INCLUDE_PATH
  "/home/pi/AI/EdgeVisionRT/include"
  "/usr/include/opencv4"
  "/home/pi/AI/EdgeVisionRT/deps/ncnn-vulkan-install/include/ncnn"
  )

# The set of dependency files which are needed:
set(CMAKE_DEPENDS_DEPENDENCY_FILES
  "/home/pi/AI/EdgeVisionRT/src/benchmark.cpp" "CMakeFiles/yolo_inference.dir/src/benchmark.cpp.o" "gcc" "CMakeFiles/yolo_inference.dir/src/benchmark.cpp.o.d"
  "/home/pi/AI/EdgeVisionRT/src/inference_engine.cpp" "CMakeFiles/yolo_inference.dir/src/inference_engine.cpp.o" "gcc" "CMakeFiles/yolo_inference.dir/src/inference_engine.cpp.o.d"
  "/home/pi/AI/EdgeVisionRT/src/input_pipeline.cpp" "CMakeFiles/yolo_inference.dir/src/input_pipeline.cpp.o" "gcc" "CMakeFiles/yolo_inference.dir/src/input_pipeline.cpp.o.d"
  "/home/pi/AI/EdgeVisionRT/src/main.cpp" "CMakeFiles/yolo_inference.dir/src/main.cpp.o" "gcc" "CMakeFiles/yolo_inference.dir/src/main.cpp.o.d"
  "/home/pi/AI/EdgeVisionRT/src/neon_preprocess.cpp" "CMakeFiles/yolo_inference.dir/src/neon_preprocess.cpp.o" "gcc" "CMakeFiles/yolo_inference.dir/src/neon_preprocess.cpp.o.d"
  "/home/pi/AI/EdgeVisionRT/src/postprocess.cpp" "CMakeFiles/yolo_inference.dir/src/postprocess.cpp.o" "gcc" "CMakeFiles/yolo_inference.dir/src/postprocess.cpp.o.d"
  )

# Targets to which this target links.
set(CMAKE_TARGET_LINKED_INFO_FILES
  )

# Fortran module output directory.
set(CMAKE_Fortran_TARGET_MODULE_DIR "")
