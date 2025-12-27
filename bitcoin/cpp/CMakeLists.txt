cmake_minimum_required(VERSION 3.16)
project(hqt_sct_trading VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# Aggressive optimizations for nanosecond trading
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -flto -ffast-math -funroll-loops -DNDEBUG")

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# ============================================
# HQT ARBITRAGE DETECTOR (Cross-exchange arbitrage)
# ============================================
add_executable(hqt_arbitrage
    src/hqt_arbitrage.cpp
)

target_compile_definitions(hqt_arbitrage PRIVATE HQT_STANDALONE)

target_link_libraries(hqt_arbitrage PRIVATE
    pthread
)

# ============================================
# SCT WILSON CI (Statistical Certainty Trading)
# ============================================
add_executable(sct_wilson
    src/sct_wilson.cpp
)

target_compile_definitions(sct_wilson PRIVATE SCT_STANDALONE)

target_link_libraries(sct_wilson PRIVATE
    pthread
)

# ============================================
# SHARED LIBRARY (For Python FFI integration)
# ============================================
add_library(hqt_sct SHARED
    src/hqt_arbitrage.cpp
    src/sct_wilson.cpp
)

target_link_libraries(hqt_sct PRIVATE
    pthread
)

set_target_properties(hqt_sct PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# Install
install(TARGETS hqt_arbitrage sct_wilson hqt_sct DESTINATION bin)
