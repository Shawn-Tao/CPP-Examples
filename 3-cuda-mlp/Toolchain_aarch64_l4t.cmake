# Toolchain_aarch64_l4t.cmake

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(target_arch aarch64-linux-gnu)
set(CMAKE_LIBRARY_ARCHITECTURE ${target_arch} CACHE STRING "" FORCE)

# Configure cmake to look for libraries, include directories and
# packages inside the target root prefix.
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
set(CMAKE_FIND_ROOT_PATH "/usr/${target_arch}")

# needed to avoid doing some more strict compiler checks that
# are failing when cross-compiling
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

# specify the toolchain programs
find_program(CMAKE_C_COMPILER ${target_arch}-gcc)
find_program(CMAKE_CXX_COMPILER ${target_arch}-g++)
if(NOT CMAKE_C_COMPILER OR NOT CMAKE_CXX_COMPILER)
    message(FATAL_ERROR "Can't find suitable C/C++ cross compiler for ${target_arch}")
endif()

set(CMAKE_AR ${target_arch}-ar CACHE FILEPATH "" FORCE)
set(CMAKE_RANLIB ${target_arch}-ranlib)
set(CMAKE_LINKER ${target_arch}-ld)

# Not all shared libraries dependencies are installed in host machine.
# Make sure linker doesn't complain.
set(CMAKE_EXE_LINKER_FLAGS_INIT -Wl,--allow-shlib-undefined)

# instruct nvcc to use our cross-compiler
set(CMAKE_CUDA_FLAGS "-ccbin ${CMAKE_CXX_COMPILER} -Xcompiler -fPIC" CACHE STRING "" FORCE)