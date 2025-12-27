#!/bin/bash
# =============================================================================
# C++ MASTER PIPELINE LAUNCHER
# =============================================================================
# Starts the C++ blockchain runner as the primary signal generator.
# This is the main entry point for the trading system.
#
# Usage:
#   ./start_cpp_pipeline.sh              # Start in foreground
#   ./start_cpp_pipeline.sh --tmux       # Start in tmux session
#   ./start_cpp_pipeline.sh --paper      # Start with paper trading
#
# =============================================================================

set -e

# Configuration
CPP_BINARY="/root/sovereign/cpp_runner/build/blockchain_runner"
ADDRESS_DB="/root/sovereign/walletexplorer_addresses.db"
UTXO_DB="/root/sovereign/exchange_utxos.db"
ZMQ_ENDPOINT="tcp://127.0.0.1:28332"
LOG_FILE="/root/sovereign/cpp_pipeline.log"
TMUX_SESSION="cpp_pipeline"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Parse arguments
USE_TMUX=false
PAPER_MODE=false
PYTHON_BRIDGE=false

for arg in "$@"; do
    case $arg in
        --tmux)
            USE_TMUX=true
            shift
            ;;
        --paper)
            PAPER_MODE=true
            shift
            ;;
        --bridge)
            PYTHON_BRIDGE=true
            shift
            ;;
        *)
            ;;
    esac
done

echo -e "${GREEN}=======================================${NC}"
echo -e "${GREEN}C++ MASTER PIPELINE${NC}"
echo -e "${GREEN}=======================================${NC}"

# Check if binary exists
if [ ! -f "$CPP_BINARY" ]; then
    echo -e "${RED}ERROR: C++ binary not found at $CPP_BINARY${NC}"
    echo "Please build the C++ runner first:"
    echo "  cd /root/sovereign/cpp_runner && ./build.sh"
    exit 1
fi

# Check if address database exists
if [ ! -f "$ADDRESS_DB" ]; then
    echo -e "${RED}ERROR: Address database not found at $ADDRESS_DB${NC}"
    exit 1
fi

# Kill any existing processes
echo -e "${YELLOW}Stopping existing processes...${NC}"
pkill -9 blockchain_runner 2>/dev/null || true
tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true
sleep 1

# Build command
if [ "$PYTHON_BRIDGE" = true ]; then
    # Use Python bridge (handles trading integration)
    CMD="python3 /root/sovereign/blockchain/cpp_master_pipeline.py"
    if [ "$PAPER_MODE" = true ]; then
        CMD="$CMD --paper"
    fi
else
    # Direct C++ runner with line buffering
    CMD="stdbuf -oL $CPP_BINARY --db $ADDRESS_DB --utxo $UTXO_DB --zmq $ZMQ_ENDPOINT"
fi

echo -e "${GREEN}Starting C++ pipeline...${NC}"
echo "Binary:     $CPP_BINARY"
echo "Address DB: $ADDRESS_DB"
echo "UTXO DB:    $UTXO_DB"
echo "ZMQ:        $ZMQ_ENDPOINT"
echo ""

if [ "$USE_TMUX" = true ]; then
    # Start in tmux session
    echo -e "${GREEN}Starting in tmux session: $TMUX_SESSION${NC}"
    tmux new-session -d -s "$TMUX_SESSION" "$CMD 2>&1 | tee $LOG_FILE"
    echo ""
    echo "Pipeline started in background."
    echo "  Attach: tmux attach -t $TMUX_SESSION"
    echo "  Log:    tail -f $LOG_FILE"
    echo ""

    # Wait a moment and show initial output
    sleep 3
    echo -e "${YELLOW}Initial output:${NC}"
    tail -20 "$LOG_FILE" 2>/dev/null || echo "(waiting for output...)"
else
    # Run in foreground
    echo -e "${GREEN}Running in foreground (Ctrl+C to stop)${NC}"
    echo ""
    $CMD 2>&1 | tee "$LOG_FILE"
fi
