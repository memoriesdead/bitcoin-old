#!/bin/bash
# ==============================================================================
# HYPERLIQUID EXECUTOR DEPLOYMENT FOR HOSTINGER VPS
# ==============================================================================
#
# This script sets up the Hyperliquid executor on your Hostinger VPS.
# Run: bash hyperliquid_deploy.sh
#
# Your VPS: 31.97.211.217
# ==============================================================================

set -e

echo "=============================================="
echo "HYPERLIQUID EXECUTOR DEPLOYMENT"
echo "=============================================="
echo ""

# Configuration
WORK_DIR="/root/sovereign"
VENV_DIR="$WORK_DIR/venv"

# Create working directory
mkdir -p $WORK_DIR
cd $WORK_DIR

echo "[1/6] Installing system dependencies..."
apt update
apt install -y python3 python3-pip python3-venv tmux curl wget git

echo ""
echo "[2/6] Creating Python virtual environment..."
python3 -m venv $VENV_DIR
source $VENV_DIR/bin/activate

echo ""
echo "[3/6] Installing Python packages..."
pip install --upgrade pip
pip install \
    hyperliquid-python-sdk \
    eth-account \
    websockets \
    requests \
    python-dotenv

echo ""
echo "[4/6] Creating configuration..."
cat > $WORK_DIR/.env << 'ENVFILE'
# Hyperliquid Configuration
# ==========================
# IMPORTANT: Replace with your actual private key!
#
# To get your private key:
# 1. Go to https://app.hyperliquid.xyz
# 2. Connect your wallet
# 3. Export the private key from your wallet (MetaMask, etc.)
#
# NEVER share this key with anyone!

HYPERLIQUID_PRIVATE_KEY=0x_YOUR_PRIVATE_KEY_HERE

# Execution mode: paper, testnet, live
EXECUTION_MODE=paper

# Trading parameters
SYMBOL=BTC
LEVERAGE=10
MAX_POSITION_USD=100

# Risk management
STOP_LOSS_PCT=0.003
TAKE_PROFIT_PCT=0.006
MAX_DAILY_TRADES=50
MAX_DAILY_LOSS_PCT=0.10

# Signal thresholds
MIN_CONFIDENCE=0.6
STRONG_SIGNAL_THRESHOLD=0.8
ENVFILE

chmod 600 $WORK_DIR/.env

echo ""
echo "[5/6] Creating main runner script..."
cat > $WORK_DIR/run_executor.py << 'PYFILE'
#!/usr/bin/env python3
"""
Sovereign Hyperliquid Executor Runner
=====================================

Connects the blockchain signal engine to Hyperliquid execution.
"""

import os
import sys
import time
import json
import signal
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add paths
sys.path.insert(0, '/root/sovereign')

# Load environment
load_dotenv('/root/sovereign/.env')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/root/sovereign/executor.log')
    ]
)
logger = logging.getLogger(__name__)

# Import executor (will be copied from livetrading)
try:
    from hyperliquid_executor import (
        HyperliquidExecutor,
        HyperliquidConfig,
        ExecutionMode,
        SignalBridge
    )
except ImportError:
    logger.error("hyperliquid_executor.py not found. Copy it from Windows.")
    sys.exit(1)


class SovereignRunner:
    """Main runner that connects signals to execution."""

    def __init__(self):
        self.running = True
        self.executor = None
        self.bridge = None
        self.setup_signal_handlers()

    def setup_signal_handlers(self):
        """Handle graceful shutdown."""
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

    def _shutdown(self, signum, frame):
        logger.info("Shutdown signal received...")
        self.running = False

    def load_config(self) -> HyperliquidConfig:
        """Load configuration from environment."""
        mode_str = os.getenv('EXECUTION_MODE', 'paper').lower()
        mode_map = {
            'paper': ExecutionMode.PAPER,
            'testnet': ExecutionMode.TESTNET,
            'live': ExecutionMode.LIVE
        }

        return HyperliquidConfig(
            private_key=os.getenv('HYPERLIQUID_PRIVATE_KEY', '0x' + '0' * 64),
            mode=mode_map.get(mode_str, ExecutionMode.PAPER),
            symbol=os.getenv('SYMBOL', 'BTC'),
            leverage=int(os.getenv('LEVERAGE', 10)),
            max_position_usd=float(os.getenv('MAX_POSITION_USD', 100)),
            stop_loss_pct=float(os.getenv('STOP_LOSS_PCT', 0.003)),
            take_profit_pct=float(os.getenv('TAKE_PROFIT_PCT', 0.006)),
            max_daily_trades=int(os.getenv('MAX_DAILY_TRADES', 50)),
            max_daily_loss_pct=float(os.getenv('MAX_DAILY_LOSS_PCT', 0.10)),
            min_confidence=float(os.getenv('MIN_CONFIDENCE', 0.6)),
            strong_signal_threshold=float(os.getenv('STRONG_SIGNAL_THRESHOLD', 0.8)),
        )

    def start(self):
        """Start the executor."""
        logger.info("=" * 60)
        logger.info("SOVEREIGN HYPERLIQUID EXECUTOR")
        logger.info("=" * 60)

        # Load config
        config = self.load_config()
        logger.info(f"Mode: {config.mode.value}")
        logger.info(f"Symbol: {config.symbol}")
        logger.info(f"Leverage: {config.leverage}x")
        logger.info(f"Max Position: ${config.max_position_usd}")

        # Initialize executor
        self.executor = HyperliquidExecutor(config)
        self.bridge = SignalBridge(self.executor)

        # Show initial state
        stats = self.executor.get_stats()
        logger.info(f"Account: {stats['account']}")

        # Main loop - listen for signals
        logger.info("")
        logger.info("Executor ready. Waiting for signals...")
        logger.info("Signal file: /root/sovereign/signal.json")
        logger.info("")

        signal_file = Path('/root/sovereign/signal.json')
        last_signal_time = 0

        while self.running:
            try:
                # Check for signal file
                if signal_file.exists():
                    mtime = signal_file.stat().st_mtime
                    if mtime > last_signal_time:
                        # New signal
                        signal_data = json.loads(signal_file.read_text())
                        last_signal_time = mtime

                        logger.info(f"Signal received: {signal_data}")

                        # Process signal
                        result = self.bridge.process_signal(signal_data)
                        if result:
                            logger.info(f"Execution result: {result}")

                # Sleep before next check
                time.sleep(1)

            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(5)

        logger.info("Executor shutdown complete.")


def main():
    runner = SovereignRunner()
    runner.start()


if __name__ == "__main__":
    main()
PYFILE

chmod +x $WORK_DIR/run_executor.py

echo ""
echo "[6/6] Creating systemd service..."
cat > /etc/systemd/system/sovereign-executor.service << 'SVCFILE'
[Unit]
Description=Sovereign Hyperliquid Executor
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/sovereign
Environment=PATH=/root/sovereign/venv/bin:/usr/bin
ExecStart=/root/sovereign/venv/bin/python /root/sovereign/run_executor.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
SVCFILE

systemctl daemon-reload

echo ""
echo "=============================================="
echo "DEPLOYMENT COMPLETE!"
echo "=============================================="
echo ""
echo "NEXT STEPS:"
echo ""
echo "1. Copy the executor file from Windows:"
echo "   scp C:\\Users\\kevin\\livetrading\\engine\\sovereign\\execution\\hyperliquid_executor.py root@31.97.211.217:/root/sovereign/"
echo ""
echo "2. Edit configuration:"
echo "   nano /root/sovereign/.env"
echo "   # Add your private key and set mode to 'live'"
echo ""
echo "3. Test in paper mode:"
echo "   cd /root/sovereign"
echo "   source venv/bin/activate"
echo "   python run_executor.py"
echo ""
echo "4. Run with systemd (production):"
echo "   systemctl enable sovereign-executor"
echo "   systemctl start sovereign-executor"
echo "   journalctl -u sovereign-executor -f"
echo ""
echo "5. Send signals by writing to /root/sovereign/signal.json:"
echo '   echo '"'"'{"direction": 1, "confidence": 0.8}'"'"' > /root/sovereign/signal.json'
echo ""
echo "=============================================="
