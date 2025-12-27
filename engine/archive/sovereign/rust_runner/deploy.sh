#!/bin/bash
# DEPLOY NANOSECOND BLOCKCHAIN RUNNER TO VPS
#
# Run locally: ./deploy.sh
#
# This script:
# 1. Uploads Rust source to VPS
# 2. Builds on VPS (compiles for Linux)
# 3. Starts the runner

set -e

VPS="root@31.97.211.217"
REMOTE_DIR="/root/sovereign/rust_runner"

echo "========================================"
echo "DEPLOYING NANOSECOND BLOCKCHAIN RUNNER"
echo "========================================"
echo ""

# Upload source files
echo "[1/4] Uploading source files..."
ssh $VPS "mkdir -p $REMOTE_DIR/src"
scp Cargo.toml $VPS:$REMOTE_DIR/
scp src/main.rs $VPS:$REMOTE_DIR/src/
scp build.sh $VPS:$REMOTE_DIR/
scp signal_bridge.py $VPS:$REMOTE_DIR/

echo ""
echo "[2/4] Installing Rust on VPS (if needed)..."
ssh $VPS "which cargo || (curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && source ~/.cargo/env)"

echo ""
echo "[3/4] Building on VPS..."
ssh $VPS "cd $REMOTE_DIR && source ~/.cargo/env && cargo build --release 2>&1 | tail -20"

echo ""
echo "[4/4] Testing binary..."
ssh $VPS "ls -la $REMOTE_DIR/target/release/blockchain_runner"

echo ""
echo "========================================"
echo "DEPLOYMENT COMPLETE"
echo "========================================"
echo ""
echo "To run the Rust blockchain runner:"
echo "  ssh $VPS"
echo "  cd $REMOTE_DIR"
echo "  ./target/release/blockchain_runner"
echo ""
echo "Or with paper trading:"
echo "  ./target/release/blockchain_runner | python3 signal_bridge.py --stdin"
echo ""
echo "Expected latency: 6-12 microseconds (vs 250-500us Python)"
echo "========================================"
