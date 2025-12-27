#!/bin/bash
# Build script for C++ Blockchain Runner

set -e

echo "========================================="
echo "Building C++ Blockchain Runner"
echo "========================================="

# Install dependencies if needed
if ! pkg-config --exists libzmq; then
    echo "Installing libzmq..."
    apt-get update && apt-get install -y libzmq3-dev
fi

if ! pkg-config --exists sqlite3; then
    echo "Installing sqlite3..."
    apt-get update && apt-get install -y libsqlite3-dev
fi

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo "Configuring..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build with all cores
echo "Building..."
make -j$(nproc)

echo ""
echo "========================================="
echo "Build complete!"
echo "Binary: $(pwd)/blockchain_runner"
echo "========================================="

# Show binary size
ls -lh blockchain_runner
