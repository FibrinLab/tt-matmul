cmake_minimum_required(VERSION 3.20)

project(tt_matmul_bench LANGUAGES C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(TT-Metalium REQUIRED)

add_library(blake3 STATIC
    third_party/blake3/blake3.c
    third_party/blake3/blake3_dispatch.c
    third_party/blake3/blake3_portable.c
)

target_include_directories(blake3 PUBLIC third_party/blake3)

target_compile_definitions(blake3 PRIVATE
    BLAKE3_NO_SSE2
    BLAKE3_NO_SSE41
    BLAKE3_NO_AVX2
    BLAKE3_NO_AVX512
    BLAKE3_USE_NEON=0
)

add_executable(tt_matmul_bench
    src/main.cpp
)

target_link_libraries(tt_matmul_bench PRIVATE
    TT::Metalium
    blake3
)
