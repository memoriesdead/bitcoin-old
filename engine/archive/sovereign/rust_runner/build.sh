#!/bin/bash
# Build script for Nanosecond Blockchain Runner
# Run on VPS: ./build.sh

set -e

echo "========================================"
echo "NANOSECOND BLOCKCHAIN RUNNER - Build"
echo "========================================"

# Install Rust if not present
if ! command -v cargo &> /dev/null; then
    echo "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source $HOME/.cargo/env
fi

# Build in release mode (max optimization)
echo "Building with optimizations..."
cargo build --release

echo ""
echo "Build complete!"
echo "Binary: target/release/blockchain_runner"
echo ""
echo "To run:"
echo "  ./target/release/blockchain_runner"
echo ""
echo "Expected latency: 6-12 microseconds (vs 250-500us Python)"
