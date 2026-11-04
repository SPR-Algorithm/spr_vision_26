cmake_minimum_required(VERSION 3.16)
set(OpenVINO_DIR "/opt/intel/openvino_2024.6.0/runtime/cmake")

find_package(OpenVINO REQUIRED)
find_package(Ceres REQUIRED)

add_library(auto_buff OBJECT 
    buff_processor.cpp
    buff_type.cpp
    observation_refiner.cpp
    buff_detector.cpp
    rune_reconstructor.cpp
    rune_pose_tools.cpp
    phase_motion_estimator.cpp
    buff_decision_module.cpp
    yolo11_buff.cpp
)

target_compile_definitions(auto_buff PUBLIC OPENCV_DISABLE_EIGEN_TENSOR_SUPPORT)
get_target_property(AUTO_BUFF_EIGEN_INCLUDE_DIRS Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
target_include_directories(auto_buff BEFORE PUBLIC ${AUTO_BUFF_EIGEN_INCLUDE_DIRS})

target_link_libraries(auto_buff io openvino::runtime yaml-cpp ${CERES_LIBRARIES})
