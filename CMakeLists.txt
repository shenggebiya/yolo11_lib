cmake_minimum_required(VERSION 3.5)
project(YOLODetector)

set(CMAKE_CXX_STANDARD 14)

# ---- 库部分 ----
find_package(OpenCV REQUIRED)

set(ONNXRUNTIME_INCLUDE_DIR "/usr/local/include/onnxruntime")
set(ONNXRUNTIME_LIBRARY     "/usr/local/lib/libonnxruntime.so")

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIR}
)

add_library(yolodetector SHARED src/YOLODetector.cpp)
target_link_libraries(yolodetector
    PRIVATE
      ${OpenCV_LIBS}
      ${ONNXRUNTIME_LIBRARY}
)

# ---- 示例可执行文件 ----
add_executable(demo examples/main.cpp)
# 包含头文件目录
target_include_directories(demo PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)
# 链接 yolodetector、OpenCV、ONNXRuntime
target_link_libraries(demo
    PRIVATE
      yolodetector
      ${OpenCV_LIBS}
      ${ONNXRUNTIME_LIBRARY}
)

 
