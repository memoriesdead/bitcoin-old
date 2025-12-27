#!/bin/bash
# Build HQT and SCT C++ modules

set -e

echo "========================================="
echo "Building HQT + SCT C++ Trading Modules"
echo "========================================="

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo ""
echo ">>> Configuring..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
echo ""
echo ">>> Building..."
make -j$(nproc)

echo ""
echo "========================================="
echo "BUILD COMPLETE"
echo "========================================="
echo ""
echo "Executables:"
echo "  ./build/hqt_arbitrage  - HQT Arbitrage Detector"
echo "  ./build/sct_wilson     - SCT Wilson CI Calculator"
echo "  ./build/libhqt_sct.so  - Shared library for Python FFI"
echo ""
