cmake_minimum_required(VERSION 3.10.0)
message(STATUS "CMake version: ${CMAKE_VERSION}")

# set(CMAKE_SYSTEM_NAME Linux)
# set(CMAKE_SYSTEM_PROCESSOR aarch64)

# set(CMAKE_C_COMPILER /usr/bin/aarch64-linux-gnu-gcc-7)
# set(CMAKE_CXX_COMPILER /usr/bin/aarch64-linux-gnu-g++-7)

# set(CMAKE_FIND_ROOT_PATH /usr/aarch64-linux-gnu)
# set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH)
# set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
# set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM BOTH)

# set(CMAKE_CUDA_FLAGS "-ccbin ${CMAKE_CXX_COMPILER} -Xcompiler -fPIC" CACHE STRING "" FORCE)

# set(CMAKE_COLOR_MAKEFILE ON)
# set(CMAKE_CXX_STANDARD 14)

project(cuda_cross_compile_test LANGUAGES CUDA CXX C)

file(GLOB cpu_source_files "src/*.cc")
file(GLOB gpu_source_files "src/*.cu")

include_directories(include)

# judge the host platform is x86 or aarch64
if(CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "x86_64")
    message(STATUS "Host platform is x86_64")
    include_directories(/usr/local/cuda/include)
    link_directories(/usr/local/cuda/lib64)

    # enable_language(CUDA)
    set(CMAKE_POLICY_DEFAULT_CMP0074 NEW)
    set(CUDAToolkit_ROOT /usr/local/cuda) # it's inside CMAKE_SYSROOT
    set(CMAKE_CUDA_ARCHITECTURES 60)
    set(CUDA_ARCHITECTURES 60)
else()
    message(STATUS "Host platform is aarch64")
    include_directories(/usr/local/cuda-10.2/targets/aarch64-linux/include)
    link_directories(/usr/local/cuda-10.2/targets/aarch64-linux/lib)

    # enable_language(CUDA)
    set(CMAKE_POLICY_DEFAULT_CMP0074 NEW)
    set(CUDAToolkit_ROOT /usr/local/cuda-10.2/targets/aarch64-linux/) # it's inside CMAKE_SYSROOT 
endif()


add_executable(${PROJECT_NAME} ${cpu_source_files} ${gpu_source_files})
target_link_libraries(${PROJECT_NAME} cudadevrt cudart_static rt pthread dl)
