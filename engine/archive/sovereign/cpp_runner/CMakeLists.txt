cmake_minimum_required(VERSION 3.16)
project(blockchain_runner VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# Aggressive optimizations for HFT
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -flto -ffast-math -funroll-loops -DNDEBUG")

# Find required packages
find_package(PkgConfig REQUIRED)
pkg_check_modules(ZMQ REQUIRED libzmq)
pkg_check_modules(SQLITE3 REQUIRED sqlite3)

# ============================================
# BLOCKCHAIN RUNNER (uses mmap - INSTANT startup)
# ============================================
add_executable(blockchain_runner
    src/main.cpp
    src/utxo_cache.cpp
    src/flow_detector.cpp
    src/tx_decoder.cpp
)

target_include_directories(blockchain_runner PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${ZMQ_INCLUDE_DIRS}
    ${SQLITE3_INCLUDE_DIRS}
)

# Note: Still need SQLite for UTXO cache, but address loading is via mmap
target_link_libraries(blockchain_runner PRIVATE
    ${ZMQ_LIBRARIES}
    ${SQLITE3_LIBRARIES}
    pthread
)

# ============================================
# ADDRESS COMPILER (SQLite -> Binary, run ONCE)
# ============================================
add_executable(address_compiler
    src/address_compiler.cpp
)

target_include_directories(address_compiler PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${SQLITE3_INCLUDE_DIRS}
)

target_link_libraries(address_compiler PRIVATE
    ${SQLITE3_LIBRARIES}
)

# Install
install(TARGETS blockchain_runner address_compiler DESTINATION bin)
