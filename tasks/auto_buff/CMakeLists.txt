cmake_minimum_required(VERSION 3.16)
set(OpenVINO_DIR "/opt/intel/openvino_2024.6.0/runtime/cmake")

find_package(OpenVINO REQUIRED)
find_package(Ceres REQUIRED)
find_package(OpenSSL REQUIRED COMPONENTS Crypto)

add_library(auto_buff OBJECT 
    buff_processor.cpp
    buff_type.cpp
    observation_refiner.cpp
    buff_detector.cpp
    rune_reconstructor.cpp
    rune_pose_tools.cpp
    phase_motion_estimator.cpp
    buff_decision_module.cpp
    yolo11_buff_openvino.cpp
    yolo11_buff_model_hash.cpp
    yolo11_buff.cpp
)

target_compile_definitions(auto_buff PUBLIC OPENCV_DISABLE_EIGEN_TENSOR_SUPPORT)
get_target_property(AUTO_BUFF_EIGEN_INCLUDE_DIRS Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
target_include_directories(auto_buff BEFORE PUBLIC ${AUTO_BUFF_EIGEN_INCLUDE_DIRS})

target_link_libraries(auto_buff io openvino::runtime yaml-cpp OpenSSL::Crypto ${CERES_LIBRARIES})

if(SPR_AUTO_BUFF_TENSORRT)
  set(TensorRT_ROOT "$ENV{TensorRT_ROOT}" CACHE PATH "TensorRT SDK root (headers and runtime library)")
  find_path(TensorRT_INCLUDE_DIR NvInfer.h
    HINTS "${TensorRT_ROOT}"
    PATH_SUFFIXES include include/x86_64-linux-gnu include/aarch64-linux-gnu)
  find_library(TensorRT_NVINFER_LIBRARY nvinfer
    HINTS "${TensorRT_ROOT}"
    PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu lib/aarch64-linux-gnu)
  if(NOT TensorRT_INCLUDE_DIR OR NOT TensorRT_NVINFER_LIBRARY)
    message(FATAL_ERROR "TensorRT headers/runtime not found; set -DTensorRT_ROOT=<SDK root>")
  endif()
  if(TensorRT_NVINFER_LIBRARY MATCHES "(^|/)stubs(/|$)")
    message(FATAL_ERROR "TensorRT runtime must be a real library, not a stub: ${TensorRT_NVINFER_LIBRARY}")
  endif()
  set(_trt_version_header "${TensorRT_INCLUDE_DIR}/NvInferVersion.h")
  if(NOT EXISTS "${_trt_version_header}")
    message(FATAL_ERROR "TensorRT SDK is missing NvInferVersion.h")
  endif()
  include(CheckCXXSourceCompiles)
  set(_trt_saved_required_includes "${CMAKE_REQUIRED_INCLUDES}")
  set(CMAKE_REQUIRED_INCLUDES "${TensorRT_INCLUDE_DIR}")
  # Resolve version macros through the compiler: some SDKs alias them to TRT_*_ENTERPRISE.
  # Recheck when reconfiguring so a different TensorRT_ROOT cannot reuse a stale result.
  unset(SPR_AUTO_BUFF_TENSORRT_VERSION_SUPPORTED CACHE)
  check_cxx_source_compiles("
    #include <NvInferVersion.h>
    #if NV_TENSORRT_MAJOR != 10 || NV_TENSORRT_MINOR < 13
    #error auto_buff requires TensorRT 10.13 or newer in the 10.x series
    #endif
    int main() { return 0; }
  " SPR_AUTO_BUFF_TENSORRT_VERSION_SUPPORTED)
  set(CMAKE_REQUIRED_INCLUDES "${_trt_saved_required_includes}")
  if(NOT SPR_AUTO_BUFF_TENSORRT_VERSION_SUPPORTED)
    message(FATAL_ERROR "auto_buff supports TensorRT 10.13+ in the 10.x series; check ${_trt_version_header}")
  endif()

  target_sources(auto_buff PRIVATE yolo11_buff_tensorrt.cpp yolo11_buff_cuda_preprocess.cu)
  target_include_directories(auto_buff PRIVATE "${TensorRT_INCLUDE_DIR}")
  target_compile_definitions(auto_buff PUBLIC SPR_AUTO_BUFF_TENSORRT=1)
  target_link_libraries(auto_buff CUDA::cudart "${TensorRT_NVINFER_LIBRARY}")
  set_target_properties(auto_buff PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
  message(STATUS "auto_buff TensorRT 10.13+ runtime: ${TensorRT_NVINFER_LIBRARY}")
endif()
