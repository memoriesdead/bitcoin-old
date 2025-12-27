#!/bin/bash
# Deploy Blockchain Live Trading Services
# Run on VPS: bash deploy-blockchain-trading.sh

set -e

echo "=========================================="
echo "BLOCKCHAIN LIVE TRADING DEPLOYMENT"
echo "=========================================="
echo "THE REAL EDGE: 10-60 seconds ahead of market"
echo "INFLOW = SHORT, OUTFLOW = LONG"
echo "=========================================="

# Check Bitcoin Core is running
if ! systemctl is-active --quiet bitcoind; then
    echo "[ERROR] bitcoind is not running!"
    echo "Start with: systemctl start bitcoind"
    exit 1
fi

# Check ZMQ is enabled
if ! bitcoin-cli getzmqnotifications 2>/dev/null | grep -q rawtx; then
    echo "[WARNING] ZMQ may not be configured in bitcoin.conf"
    echo "Add these lines to /root/.bitcoin/bitcoin.conf:"
    echo "  zmqpubrawtx=tcp://127.0.0.1:28332"
    echo "  zmqpubrawblock=tcp://127.0.0.1:28333"
fi

# Install services
echo ""
echo "[1/5] Installing systemd services..."
cp /root/livetrading/engine/sovereign/config/blockchain-*.service /etc/systemd/system/
systemctl daemon-reload

# Enable services
echo "[2/5] Enabling services..."
for exchange in binance kraken coinbase bitstamp gemini; do
    systemctl enable blockchain-${exchange}.service
    echo "  Enabled: blockchain-${exchange}"
done

# Create data directory
echo "[3/5] Creating data directory..."
mkdir -p /root/livetrading/data

# Start services
echo "[4/5] Starting services..."
for exchange in binance kraken coinbase bitstamp gemini; do
    systemctl start blockchain-${exchange}.service
    echo "  Started: blockchain-${exchange}"
done

# Show status
echo "[5/5] Checking status..."
echo ""
echo "=========================================="
echo "SERVICE STATUS"
echo "=========================================="
for exchange in binance kraken coinbase bitstamp gemini; do
    status=$(systemctl is-active blockchain-${exchange}.service)
    echo "  blockchain-${exchange}: $status"
done

echo ""
echo "=========================================="
echo "DEPLOYMENT COMPLETE"
echo "=========================================="
echo ""
echo "Commands:"
echo "  View logs:  journalctl -u blockchain-binance -f"
echo "  Stop all:   systemctl stop blockchain-*.service"
echo "  Start all:  systemctl start blockchain-*.service"
echo ""
echo "Each service runs with:"
echo "  - \$100 capital"
echo "  - 0.25 Kelly fraction"
echo "  - Real exchange fees"
echo "  - 7.6M exchange addresses for flow detection"
echo ""
echo "Target: 50.75% win rate after fees"
echo "=========================================="
