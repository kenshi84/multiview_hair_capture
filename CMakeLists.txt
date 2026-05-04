# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.18)
project(hair_recon LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA architecture — covers Ampere (A100/RTX 30xx), Ada (RTX 40xx), and
# Hopper (H100). Override with -DCMAKE_CUDA_ARCHITECTURES="80" for a single
# target if you want faster builds.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80;86;89;90)
endif()

# MSVC + CUDA compatibility: automatically allow unsupported host compiler
# when the MSVC version is newer than what the CUDA toolkit officially supports.
# CUDA 12.x supports up to MSVC 19.4x (VS 2022); newer versions need this flag.
if(MSVC AND MSVC_VERSION GREATER_EQUAL 1950)
  if(NOT CMAKE_CUDA_FLAGS MATCHES "allow-unsupported-compiler")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --allow-unsupported-compiler")
    message(STATUS "MSVC ${MSVC_VERSION} detected: adding --allow-unsupported-compiler to CUDA flags")
  endif()
endif()

enable_language(CUDA)

# ---- Dependencies ----
find_package(CUDAToolkit REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)

# ---- Third-party (git submodules) ----
set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

# Check submodules are initialized
if(NOT EXISTS "${THIRD_PARTY_DIR}/nanoflann/include/nanoflann.hpp")
  message(FATAL_ERROR
    "Git submodules not initialized. Run:\n"
    "  git submodule update --init --recursive")
endif()

# ---- Source files ----
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

set(COMMON_SOURCES
  src/common/config.cpp
  src/common/camera.cpp
  src/common/camera_array.cpp
  src/common/logger.cpp
  src/common/ply_io.cpp
  src/common/image_io.cpp
  src/common/profiler.cpp
)

set(MVS_SOURCES
  src/mvs/image_loader.cpp
  src/mvs/patchmatch_mvs.cpp
  src/mvs/depth_fusion.cpp
  src/mvs/patchmatch/engine.cpp
  src/mvs/patchmatch/hierarchical.cpp
)

set(MVS_CUDA_SOURCES
  src/mvs/gabor_orientation.cu
  src/mvs/patchmatch/image_ops.cu
  src/mvs/patchmatch/propagation.cu
)

set(STRAND_SOURCES
  src/strand/meanshift.cpp
  src/strand/strand_tracer.cpp
  src/strand/strand_cleaner.cpp
  src/strand/mesh_generator.cpp
  src/strand/strand_io.cpp
)

set(STRAND_CUDA_SOURCES
  src/strand/meanshift.cu
)

# ---- Build ----
add_executable(hair_recon
  src/main.cpp
  ${COMMON_SOURCES}
  ${MVS_SOURCES}
  ${MVS_CUDA_SOURCES}
  ${STRAND_SOURCES}
  ${STRAND_CUDA_SOURCES}
)

target_include_directories(hair_recon PRIVATE
  ${SRC_DIR}
  ${THIRD_PARTY_DIR}/nanoflann/include   # nanoflann.hpp
  ${THIRD_PARTY_DIR}/tomlplusplus        # toml.hpp (single-header at repo root)
  ${THIRD_PARTY_DIR}/stb                 # stb_image.h etc.
  ${THIRD_PARTY_DIR}/tinyexr             # tinyexr.h
  ${THIRD_PARTY_DIR}/json/single_include # nlohmann/json.hpp (single-header amalgamation)
)

target_link_libraries(hair_recon PRIVATE
  CUDA::cudart
  CUDA::curand
  Eigen3::Eigen
  OpenMP::OpenMP_CXX
)

if(MSVC)
  target_compile_definitions(hair_recon PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

target_compile_options(hair_recon PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    --expt-relaxed-constexpr
    -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored
  >
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/W3>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Wall -Wextra -Wno-unused-parameter>
)

# Suppress warnings in third-party stb implementation
if(MSVC)
  set_source_files_properties(src/common/image_io.cpp PROPERTIES COMPILE_FLAGS "/w")
else()
  set_source_files_properties(src/common/image_io.cpp PROPERTIES COMPILE_FLAGS "-w")
endif()

install(TARGETS hair_recon DESTINATION bin)
