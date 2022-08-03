cmake_minimum_required(VERSION 3.5)
project(telpoalgsdk)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)
#set(CMAKE_FIND_ROOT_PATH /usr/arm-linux-gnueabihf)

#set(CMAKE_C_COMPILER "/usr/bin/aarch64-linux-gnu-gcc")
SET(CMAKE_CXX_COMPILER "/usr/bin/arm-linux-gnueabihf-g++")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wl,--allow-shlib-undefined")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wl,--allow-shlib-undefined")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib)

set(CMAKE_CXX_STANDARD 11)

include_directories(
    ${CMAKE_SOURCE_DIR}/3rdparty/librknn_api/include
    ${CMAKE_SOURCE_DIR}/3rdparty/nlohmann
    #${CMAKE_SOURCE_DIR}/3rdparty/opencv/include
    ${CMAKE_SOURCE_DIR}/3rdparty/glog

    ${CMAKE_SOURCE_DIR}/include
)
link_directories(
    ${CMAKE_SOURCE_DIR}/3rdparty/librknn_api/lib
    #${CMAKE_SOURCE_DIR}/3rdparty/opencv/lib
    ${CMAKE_SOURCE_DIR}/lib
)

set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/3rdparty/opencv/share/OpenCV)
find_package(OpenCV REQUIRED)

#compile libtelpoalgsdk.so
list(APPEND libtelpoalgsdk_SOURCE
    ${CMAKE_SOURCE_DIR}/src/telpo_algsdk.cpp
    ${CMAKE_SOURCE_DIR}/src/telpo_applications.cpp
)
add_library(${PROJECT_NAME} SHARED  ${libtelpoalgsdk_SOURCE})
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION 1.0 )
target_include_directories(${PROJECT_NAME}
    PUBLIC ${OpenCV_INCLUDE_DIR}
    PUBLIC ${CMAKE_SOURCE_DIR}/3rdparty/librknn_api/include
    PUBLIC ${CMAKE_SOURCE_DIR}/3rdparty/nlohmann
    PUBLIC ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} librknnrt.so)


#demo_test_image
add_executable(demo_test_image demo_test_image.cpp )
target_link_libraries(demo_test_image  opencv_core;opencv_imgcodecs;opencv_imgproc telpoalgsdk)

message("opencv lib = ${OpenCV_LIBS}" )
#demo_test_rtsp
# add_executable(demo_test_rtsp demo_test_rtsp.cpp )
# target_link_libraries(demo_test_rtsp  ${OpenCV_LIBS}  telpoalgsdk)
