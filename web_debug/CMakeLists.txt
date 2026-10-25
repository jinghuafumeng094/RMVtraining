cmake_minimum_required(VERSION 3.16)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(rmv_web_debug LANGUAGES CXX)
endif()

find_package(Threads REQUIRED)
# core: cv::Mat；imgcodecs: cv::imencode 编 JPEG；imgproc: 缩放只在调参时才用
find_package(OpenCV 4 REQUIRED COMPONENTS core imgcodecs imgproc)

# 链接此目标即可获得实现、头文件以及 WEB_DEBUG 宏定义。
add_library(web_debug STATIC
    src/WebLogger.cpp
)

target_compile_features(web_debug PUBLIC cxx_std_17)
target_include_directories(web_debug
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty
        ${OpenCV_INCLUDE_DIRS}
)
target_compile_definitions(web_debug
    PUBLIC WEB_DEBUG
    PRIVATE WEB_DEBUG_STATIC_DIR="${CMAKE_CURRENT_SOURCE_DIR}/static"
)
# OpenCV 用 PUBLIC：WebLogger.hpp 的接口里出现了 cv::Mat。
target_link_libraries(web_debug PUBLIC Threads::Threads ${OpenCV_LIBS})
