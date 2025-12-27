#!/bin/bash
# ==============================================================================
# SIGNAL VALIDATION DEPLOYMENT FOR HOSTINGER VPS
# ==============================================================================
#
# RenTech Approach: Prove the edge before trading
#
# This deploys the data collection infrastructure to your Hostinger VPS.
# Run: bash deploy_validation.sh
#
# Your VPS: 31.97.211.217
# ==============================================================================

set -e

echo "=============================================="
echo "SIGNAL VALIDATION DEPLOYMENT"
echo "=============================================="
echo ""
echo "RenTech Principle: Collect data → Prove edge → Then trade"
echo ""

# Configuration
WORK_DIR="/root/validation"
VENV_DIR="$WORK_DIR/venv"
DATA_DIR="$WORK_DIR/data"

# Create directories
mkdir -p $WORK_DIR
mkdir -p $DATA_DIR
cd $WORK_DIR

echo "[1/5] Installing system dependencies..."
apt update
apt install -y python3 python3-pip python3-venv tmux curl wget

echo ""
echo "[2/5] Creating Python virtual environment..."
python3 -m venv $VENV_DIR
source $VENV_DIR/bin/activate

echo ""
echo "[3/5] Installing Python packages..."
pip install --upgrade pip
pip install \
    numpy \
    requests \
    websocket-client \
    python-dotenv

echo ""
echo "[4/5] Creating run script..."
cat > $WORK_DIR/run_collection.py << 'PYFILE'
#!/usr/bin/env python3
"""
Signal Validation Data Collector
=================================

Collects blockchain signals + price data for edge validation.
Run 24/7 for 2-4 weeks before trading.
"""

import os
import sys
import time
import json
import signal
import sqlite3
import threading
import requests
import logging
from pathlib import Path
from datetime import datetime

try:
    import websocket
    HAS_WEBSOCKET = True
except ImportError:
    HAS_WEBSOCKET = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/root/validation/collection.log')
    ]
)
logger = logging.getLogger(__name__)

# Configuration
DATA_DIR = Path("/root/validation/data")
SIGNALS_DB = DATA_DIR / "signals.db"
PRICES_DB = DATA_DIR / "prices.db"


class PriceCollector:
    """Collects BTC price every second."""

    def __init__(self):
        self.running = False
        self.current_price = 0.0
        self.prices_logged = 0
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(PRICES_DB)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS prices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                timestamp_ms INTEGER NOT NULL,
                price REAL NOT NULL,
                bid REAL,
                ask REAL,
                source TEXT
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_price_ts ON prices(timestamp)")
        conn.commit()
        conn.close()

    def log_price(self, price, bid=0, ask=0, source="binance"):
        now = time.time()
        now_ms = int(now * 1000)

        conn = sqlite3.connect(PRICES_DB)
        c = conn.cursor()
        c.execute(
            "INSERT INTO prices (timestamp, timestamp_ms, price, bid, ask, source) VALUES (?,?,?,?,?,?)",
            (now, now_ms, price, bid, ask, source)
        )
        conn.commit()
        conn.close()

        self.current_price = price
        self.prices_logged += 1

    def start_websocket(self):
        if not HAS_WEBSOCKET:
            logger.error("websocket-client not installed")
            return

        self.running = True
        thread = threading.Thread(target=self._ws_loop, daemon=True)
        thread.start()

    def _ws_loop(self):
        url = "wss://stream.binance.com:9443/ws/btcusdt@bookTicker"

        while self.running:
            try:
                ws = websocket.WebSocketApp(
                    url,
                    on_message=self._on_message,
                    on_error=lambda ws, e: logger.error(f"WS error: {e}"),
                )
                ws.run_forever()
                if self.running:
                    time.sleep(5)
            except Exception as e:
                logger.error(f"WS exception: {e}")
                time.sleep(5)

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
            bid = float(data.get('b', 0))
            ask = float(data.get('a', 0))
            price = (bid + ask) / 2
            self.log_price(price, bid, ask)
        except:
            pass

    def stop(self):
        self.running = False


class SignalCollector:
    """Collects blockchain signals."""

    def __init__(self):
        self.signals_logged = 0
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(SIGNALS_DB)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                timestamp_ms INTEGER NOT NULL,
                direction INTEGER NOT NULL,
                confidence REAL NOT NULL,
                should_trade INTEGER,
                inflow_btc REAL,
                outflow_btc REAL,
                net_flow REAL,
                price_at_signal REAL,
                source TEXT
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_signal_ts ON signals(timestamp)")
        conn.commit()
        conn.close()

    def log_signal(self, signal, price=0):
        now = time.time()
        now_ms = int(now * 1000)

        conn = sqlite3.connect(SIGNALS_DB)
        c = conn.cursor()
        c.execute("""
            INSERT INTO signals (
                timestamp, timestamp_ms, direction, confidence, should_trade,
                inflow_btc, outflow_btc, net_flow, price_at_signal, source
            ) VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (
            now, now_ms,
            signal.get('direction', 0),
            signal.get('confidence', 0),
            int(signal.get('should_trade', False)),
            signal.get('inflow_btc', 0),
            signal.get('outflow_btc', 0),
            signal.get('net_flow', 0),
            price,
            signal.get('source', 'blockchain')
        ))
        conn.commit()
        conn.close()

        self.signals_logged += 1
        return self.signals_logged


class ValidationCollector:
    """Main collector orchestrator."""

    def __init__(self):
        self.price_collector = PriceCollector()
        self.signal_collector = SignalCollector()
        self.running = False
        self.start_time = None

        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

    def _shutdown(self, signum, frame):
        logger.info("Shutdown signal received...")
        self.running = False

    def start(self, duration_hours=None):
        logger.info("=" * 60)
        logger.info("SIGNAL VALIDATION DATA COLLECTION")
        logger.info("=" * 60)
        logger.info(f"Duration: {duration_hours or 'unlimited'} hours")
        logger.info(f"Data dir: {DATA_DIR}")
        logger.info("")

        self.running = True
        self.start_time = time.time()

        # Start price collection
        self.price_collector.start_websocket()
        logger.info("Price collection started (Binance WebSocket)")

        # Signal file monitoring
        signal_file = Path("/root/validation/signal.json")
        last_signal_mtime = 0

        logger.info("")
        logger.info("Waiting for signals...")
        logger.info(f"Write signals to: {signal_file}")
        logger.info("")

        end_time = None
        if duration_hours:
            end_time = self.start_time + (duration_hours * 3600)

        last_status = 0

        while self.running:
            now = time.time()

            # Check duration
            if end_time and now >= end_time:
                logger.info("Duration reached")
                break

            # Check for new signal
            if signal_file.exists():
                mtime = signal_file.stat().st_mtime
                if mtime > last_signal_mtime:
                    last_signal_mtime = mtime
                    try:
                        signal_data = json.loads(signal_file.read_text())
                        price = self.price_collector.current_price
                        self.signal_collector.log_signal(signal_data, price)
                        logger.info(f"Signal logged: dir={signal_data.get('direction')}, "
                                   f"conf={signal_data.get('confidence', 0):.2f}")
                    except Exception as e:
                        logger.error(f"Signal parse error: {e}")

            # Status every 60 seconds
            if now - last_status >= 60:
                self._print_status()
                last_status = now

            time.sleep(0.5)

        self._print_final()

    def _print_status(self):
        runtime = (time.time() - self.start_time) / 3600
        logger.info(f"Runtime: {runtime:.2f}h | "
                   f"Signals: {self.signal_collector.signals_logged} | "
                   f"Prices: {self.price_collector.prices_logged:,} | "
                   f"BTC: ${self.price_collector.current_price:,.2f}")

    def _print_final(self):
        runtime = (time.time() - self.start_time) / 3600
        logger.info("")
        logger.info("=" * 60)
        logger.info("COLLECTION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Runtime: {runtime:.2f} hours")
        logger.info(f"Signals collected: {self.signal_collector.signals_logged}")
        logger.info(f"Prices collected: {self.price_collector.prices_logged:,}")
        logger.info("")
        logger.info("Data files:")
        logger.info(f"  {SIGNALS_DB}")
        logger.info(f"  {PRICES_DB}")
        logger.info("")
        logger.info("Next: Transfer data and run analysis")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=None,
                       help="Duration in hours")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    collector = ValidationCollector()
    collector.start(duration_hours=args.duration)


if __name__ == "__main__":
    main()
PYFILE

chmod +x $WORK_DIR/run_collection.py

echo ""
echo "[5/5] Creating systemd service..."
cat > /etc/systemd/system/signal-validation.service << 'SVCFILE'
[Unit]
Description=Signal Validation Data Collector
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/validation
Environment=PATH=/root/validation/venv/bin:/usr/bin
ExecStart=/root/validation/venv/bin/python /root/validation/run_collection.py
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
echo "QUICK START:"
echo ""
echo "1. Test collection (5 minutes):"
echo "   cd /root/validation"
echo "   source venv/bin/activate"
echo "   python run_collection.py --duration 0.083"
echo ""
echo "2. Run for 1 week (recommended):"
echo "   systemctl enable signal-validation"
echo "   systemctl start signal-validation"
echo "   journalctl -u signal-validation -f"
echo ""
echo "3. Send signals from your blockchain engine:"
echo '   echo '"'"'{"direction": 1, "confidence": 0.65, "should_trade": true}'"'"' > /root/validation/signal.json'
echo ""
echo "4. After collection, transfer data back to Windows:"
echo "   scp root@31.97.211.217:/root/validation/data/*.db C:\\Users\\kevin\\livetrading\\data\\validation\\"
echo ""
echo "5. Run analysis on Windows:"
echo "   python -m engine.sovereign.validation.analysis.calculate_edge"
echo ""
echo "=============================================="
echo "TIMELINE:"
echo "  Week 1-2: Collect data"
echo "  Week 3:   Analyze edge"
echo "  Week 4:   Paper trade (if edge confirmed)"
echo "  Week 5+:  Live trade (if paper confirms)"
echo "=============================================="
