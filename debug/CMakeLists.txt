set(DEBUG_SOURCES
  web_debugger.cpp
  param_tuner.cpp
  debug_bus.cpp
  sinks/web_sink.cpp
)

add_library(web_debugger STATIC ${DEBUG_SOURCES})

target_include_directories(web_debugger PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(web_debugger ${OpenCV_LIBS} pthread yaml-cpp)

# 把 static/ 目录安装到构建目录，运行时能找到 index.html
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/static
     DESTINATION ${CMAKE_BINARY_DIR}/debug)
