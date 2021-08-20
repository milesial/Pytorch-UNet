
cmake_minimum_required(VERSION 3.15)

project(unet_gpu
    LANGUAGES CXX C CUDA
    VERSION 1.0)

find_package(Msnhnet REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(unet_gpu unet_gpu.cpp)
target_include_directories(unet_gpu PUBLIC ${Msnhnet_INCLUDE_DIR})
target_link_libraries(unet_gpu PUBLIC Msnhnet)

