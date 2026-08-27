cmake_minimum_required(VERSION 3.16)

find_package(yaml-cpp REQUIRED)

add_subdirectory(serial)

# 创建目标 io
add_library(io STATIC 
    hikrobot/hikrobot.cpp    
    mindvision/mindvision.cpp  
    usbcamera/usbcamera.cpp  
    camera.cpp
    cboard.cpp
    dm_imu/dm_imu.cpp
    gimbal/gimbal.cpp
)

# hikrobot
target_include_directories(io PUBLIC hikrobot/include)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  target_link_directories(io PUBLIC hikrobot/lib/amd64)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  target_link_directories(io PUBLIC hikrobot/lib/arm64)
else()
  message(FATAL_ERROR "Unsupported architecture: ${CMAKE_HOST_SYSTEM_PROCESSOR}!")
endif()

# mindvision
target_include_directories(io PUBLIC mindvision/include)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  target_link_directories(io PUBLIC mindvision/lib/amd64)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  target_link_directories(io PUBLIC mindvision/lib/arm64)
else()
  message(FATAL_ERROR "Unsupported architecture: ${CMAKE_HOST_SYSTEM_PROCESSOR}!")
endif()

target_link_libraries(io MvCameraControl MVSDK usb-1.0 yaml-cpp serial)

################## 检查 ROS 环境 ##################
find_package(ament_cmake QUIET)
find_package(rclcpp QUIET)
find_package(std_msgs QUIET)
find_package(rosidl_typesupport_cpp QUIET)
find_package(rosidl_cmake QUIET)
if(TARGET spr_msgs)
    set(spr_msgs_FOUND TRUE)
else()
    find_package(spr_msgs QUIET)
endif()

# 如果找不到 ROS2 Humble 环境，则跳过 ROS 相关的部分
if(NOT SPR_USE_ROS2 OR NOT ament_cmake_FOUND OR NOT rclcpp_FOUND OR NOT std_msgs_FOUND OR NOT rosidl_typesupport_cpp_FOUND OR NOT spr_msgs_FOUND)
    message(WARNING "ROS2 not found, skipping ROS2 specific code.")
else()
    message(STATUS "ROS2 environment found, compiling ROS2-related code.")
    
    # 添加 ROS2 相关的文件
    target_sources(io PRIVATE ros2/publish2nav.cpp ros2/ros2.cpp ros2/subscribe2nav.cpp)

    # 设置 ROS2 依赖
    include_directories(${std_msgs_INCLUDE_DIRS})
    if(TARGET spr_msgs)
        ament_target_dependencies(io rclcpp std_msgs)
        rosidl_get_typesupport_target(spr_msgs_typesupport_target spr_msgs rosidl_typesupport_cpp)
        rosidl_get_typesupport_target(spr_msgs_fastrtps_typesupport_target spr_msgs rosidl_typesupport_fastrtps_cpp)
        rosidl_get_typesupport_target(spr_msgs_introspection_typesupport_target spr_msgs rosidl_typesupport_introspection_cpp)
        add_dependencies(
            io
            spr_msgs__rosidl_typesupport_fastrtps_cpp
            spr_msgs__rosidl_typesupport_introspection_cpp
        )
        target_link_libraries(
            io
            ${spr_msgs_typesupport_target}
            -Wl,--no-as-needed
            ${spr_msgs_fastrtps_typesupport_target}
            ${spr_msgs_introspection_typesupport_target}
            -Wl,--as-needed
        )
    else()
        ament_target_dependencies(io rclcpp std_msgs spr_msgs)
    endif()
endif()
