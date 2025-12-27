#!/bin/bash
# ==============================================================================
# API-BASED SIGNAL VALIDATION - NO BITCOIN CORE REQUIRED
# ==============================================================================
#
# RenTech Approach: Prove the edge before trading
#
# This uses mempool.space API for blockchain signals.
# NO 500GB Bitcoin Core node needed!
#
# Run: bash deploy_api_validation.sh
#
# Your VPS: 31.97.211.217
# ==============================================================================

set -e

echo "=============================================="
echo "API-BASED SIGNAL VALIDATION DEPLOYMENT"
echo "=============================================="
echo ""
echo "RenTech Principle: Collect data -> Prove edge -> Then trade"
echo ""
echo "Using mempool.space API - NO Bitcoin Core required!"
echo ""

# Configuration
WORK_DIR="/root/validation"
VENV_DIR="$WORK_DIR/venv"
DATA_DIR="$WORK_DIR/data"
EXCHANGES_JSON="$WORK_DIR/exchanges.json"

# Create directories
mkdir -p $WORK_DIR
mkdir -p $DATA_DIR
cd $WORK_DIR

echo "[1/6] Installing system dependencies..."
apt update
apt install -y python3 python3-pip python3-venv tmux curl wget

echo ""
echo "[2/6] Creating Python virtual environment..."
python3 -m venv $VENV_DIR
source $VENV_DIR/bin/activate

echo ""
echo "[3/6] Installing Python packages..."
pip install --upgrade pip
pip install \
    numpy \
    requests \
    websocket-client \
    python-dotenv

echo ""
echo "[4/6] Downloading exchange addresses..."
# Download exchanges.json if not exists
if [ ! -f "$EXCHANGES_JSON" ]; then
    echo "Creating exchange addresses file..."
    cat > $EXCHANGES_JSON << 'EXCHANGES'
{
  "binance": {
    "name": "Binance",
    "addresses": [
      "34xp4vRoCGJym3xR7yCVPFHoCNxv4Twseo",
      "3JZq4atUahhuA9rLhXLMhhTo133J9rF97j",
      "bc1qgdjqv0av3q56jvd82tkdjpy7gdp9ut8tlqmgrpmv24sq90ecnvqqjwvw97",
      "1NDyJtNTjmwk5xPNhjgAMu4HDHigtobu1s",
      "3M219KR5vEneNb47ewrPfWyb5jQ2DjxRP6"
    ]
  },
  "coinbase": {
    "name": "Coinbase",
    "addresses": [
      "bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh",
      "3Kzh9qAqVWQhEsfQz7zEQL1EuSx5tyNLNS",
      "395xWkEKb1bxMGEVpRRKcr7D2RqWJbLRjv",
      "1FzWLkAahHooV3kzYJSJAn4PrR2kGF7BFC",
      "bc1q0p5wm5uy6z2k6p5pz5dlq5q56lw6qrcfv85vc9"
    ]
  },
  "kraken": {
    "name": "Kraken",
    "addresses": [
      "bc1qmxjefnuy06v345v6vhwpwt05dztztmx4g3y7wp",
      "3AfMqpmZWq7GLbYkJDZJhHwq3b7hGUTpVW",
      "bc1q8cpjf6xsz9ww7y4z3r4x3e2tzu4wfn7z5jzcyp",
      "3DVJfEsDTPkGDvqPCLC41X85L1B1DQWDyh"
    ]
  },
  "bitstamp": {
    "name": "Bitstamp",
    "addresses": [
      "3P3QsMVK89JBNqZQv5zMAKG8FK3kJM4rjt",
      "3BiLTf2hNYKEXuVnKqJAiE7XJUE6zGQqMy",
      "bc1qssd9gfyqjvdxfxm7wzzj6e4vz5f2h7wfnhvqza"
    ]
  },
  "gemini": {
    "name": "Gemini",
    "addresses": [
      "bc1qsxvfmv5v99ye6k4hxzp4adfz9k3w3t7lk8r8qu",
      "3NjHh7YHaWrqkfKEyPbqNTqvKgVcVFCfHS"
    ]
  },
  "okx": {
    "name": "OKX",
    "addresses": [
      "bc1qnvkqg5wmwkpdhq2efhqamq0g6nh9xqluwwgmft",
      "3LQUu4v9z6KNch71j7kbj8GPeAGUo1FW6a"
    ]
  },
  "bybit": {
    "name": "Bybit",
    "addresses": [
      "bc1qjysjfd9t9aspttpjqzv68k0ydpe7pvyd5vlyn37868473lell5tqkz456m"
    ]
  },
  "bitfinex": {
    "name": "Bitfinex",
    "addresses": [
      "3D2oetdNuZUqQHPJmcMDDHYoqkyNVsFk9r",
      "bc1qgdjqv0av3q56jvd82tkdjpy7gdp9ut8tlqmgrpmv24sq90ecnvqqjwvw97"
    ]
  }
}
EXCHANGES
    echo "Created exchanges.json with seed addresses"
fi

echo ""
echo "[5/6] Creating signal collector script..."
cat > $WORK_DIR/run_api_validation.py << 'PYFILE'
#!/usr/bin/env python3
"""
API-Based Signal Validation Collector
=====================================

Collects blockchain signals via mempool.space API + price data.
NO BITCOIN CORE REQUIRED!

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
from datetime import datetime, timezone
from typing import Dict, Set, Optional

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
        logging.FileHandler('/root/validation/api_collection.log')
    ]
)
logger = logging.getLogger(__name__)

# Configuration
DATA_DIR = Path("/root/validation/data")
SIGNALS_DB = DATA_DIR / "signals.db"
PRICES_DB = DATA_DIR / "prices.db"
EXCHANGES_JSON = Path("/root/validation/exchanges.json")

MEMPOOL_WS_URL = "wss://mempool.space/api/v1/ws"
MEMPOOL_API_URL = "https://mempool.space/api"


def load_exchange_addresses() -> tuple:
    """Load exchange addresses from JSON."""
    addr_to_exchange = {}
    addr_set = set()

    if EXCHANGES_JSON.exists():
        try:
            data = json.loads(EXCHANGES_JSON.read_text())
            for exchange_id, info in data.items():
                for addr in info.get('addresses', []):
                    addr_to_exchange[addr] = exchange_id
                    addr_set.add(addr)
            logger.info(f"Loaded {len(addr_to_exchange)} exchange addresses from JSON")
        except Exception as e:
            logger.error(f"Error loading exchanges.json: {e}")

    return addr_to_exchange, addr_set


class PriceCollector:
    """Collects BTC price via Binance WebSocket."""

    def __init__(self):
        self.running = False
        self.current_price = 0.0
        self.current_bid = 0.0
        self.current_ask = 0.0
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
        self.current_bid = bid
        self.current_ask = ask
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
                    on_error=lambda ws, e: logger.error(f"Price WS error: {e}"),
                )
                ws.run_forever()
                if self.running:
                    time.sleep(5)
            except Exception as e:
                logger.error(f"Price WS exception: {e}")
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


class BlockchainSignalCollector:
    """Collects blockchain signals via mempool.space API."""

    def __init__(self, price_collector: PriceCollector):
        self.price_collector = price_collector
        self.running = False

        # Load addresses
        self.addr_to_exchange, self.addr_set = load_exchange_addresses()

        # Stats
        self.signals_logged = 0
        self.txs_processed = 0
        self.total_inflow_btc = 0.0
        self.total_outflow_btc = 0.0

        self._init_db()
        self._lock = threading.Lock()

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

    def log_signal(self, direction: int, confidence: float, inflow: float, outflow: float):
        now = time.time()
        now_ms = int(now * 1000)
        net = outflow - inflow
        price = self.price_collector.current_price

        conn = sqlite3.connect(SIGNALS_DB)
        c = conn.cursor()
        c.execute("""
            INSERT INTO signals (
                timestamp, timestamp_ms, direction, confidence, should_trade,
                inflow_btc, outflow_btc, net_flow, price_at_signal, source
            ) VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (
            now, now_ms, direction, confidence, int(direction != 0),
            inflow, outflow, net, price, 'mempool_api'
        ))
        conn.commit()
        conn.close()

        self.signals_logged += 1
        return self.signals_logged

    def start(self):
        """Start blockchain signal collection."""
        self.running = True

        # Start WebSocket thread
        ws_thread = threading.Thread(target=self._ws_loop, daemon=True)
        ws_thread.start()

        # Start signal emission thread (every 60 seconds)
        signal_thread = threading.Thread(target=self._signal_loop, daemon=True)
        signal_thread.start()

        logger.info("Blockchain signal collection started (mempool.space API)")

    def _ws_loop(self):
        """WebSocket connection to mempool.space."""
        while self.running:
            try:
                ws = websocket.WebSocketApp(
                    MEMPOOL_WS_URL,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=lambda ws, e: logger.error(f"Mempool WS error: {e}"),
                )
                ws.run_forever(ping_interval=30)

                if self.running:
                    logger.warning("Mempool WS disconnected, reconnecting...")
                    time.sleep(5)
            except Exception as e:
                logger.error(f"Mempool WS exception: {e}")
                time.sleep(5)

    def _on_open(self, ws):
        """Subscribe to blocks."""
        logger.info("Connected to mempool.space WebSocket")
        ws.send(json.dumps({"action": "want", "data": ["blocks", "mempool-blocks"]}))

    def _on_message(self, ws, message):
        """Handle mempool.space messages."""
        try:
            data = json.loads(message)

            # New block - fetch transactions
            if 'block' in data:
                block_hash = data['block'].get('id', '')
                if block_hash:
                    threading.Thread(
                        target=self._process_block,
                        args=(block_hash,),
                        daemon=True
                    ).start()

        except Exception as e:
            pass

    def _process_block(self, block_hash: str):
        """Fetch and process block transactions."""
        try:
            resp = requests.get(
                f"{MEMPOOL_API_URL}/block/{block_hash}/txs/0",
                timeout=15
            )
            if resp.status_code != 200:
                return

            txs = resp.json()
            block_inflow = 0.0
            block_outflow = 0.0

            for tx in txs:
                inflow, outflow = self._process_tx(tx)
                block_inflow += inflow
                block_outflow += outflow

            with self._lock:
                self.txs_processed += len(txs)
                self.total_inflow_btc += block_inflow
                self.total_outflow_btc += block_outflow

            if block_inflow > 1 or block_outflow > 1:
                logger.info(f"Block flows: +{block_outflow:.2f} BTC out, -{block_inflow:.2f} BTC in")

        except Exception as e:
            logger.error(f"Block process error: {e}")

    def _process_tx(self, tx: Dict) -> tuple:
        """Process transaction, return (inflow, outflow)."""
        inflow = 0.0
        outflow = 0.0

        # Check inputs (outflow from exchange)
        for inp in tx.get('vin', []):
            prevout = inp.get('prevout', {})
            addr = prevout.get('scriptpubkey_address', '')
            value = prevout.get('value', 0) / 1e8

            if addr in self.addr_set and value > 0:
                outflow += value

        # Check outputs (inflow to exchange)
        for out in tx.get('vout', []):
            addr = out.get('scriptpubkey_address', '')
            value = out.get('value', 0) / 1e8

            if addr in self.addr_set and value > 0:
                inflow += value

        return inflow, outflow

    def _signal_loop(self):
        """Emit signals periodically."""
        while self.running:
            time.sleep(60)  # Emit every 60 seconds

            with self._lock:
                net = self.total_outflow_btc - self.total_inflow_btc

                if net > 1.0:
                    direction = 1  # LONG
                    confidence = min(0.8, 0.5 + net / 100)
                elif net < -1.0:
                    direction = -1  # SHORT
                    confidence = min(0.8, 0.5 + abs(net) / 100)
                else:
                    direction = 0
                    confidence = 0.0

                inflow = self.total_inflow_btc
                outflow = self.total_outflow_btc

                # Reset counters
                self.total_inflow_btc = 0.0
                self.total_outflow_btc = 0.0

            # Log signal
            if direction != 0:
                self.log_signal(direction, confidence, inflow, outflow)
                dir_str = "LONG" if direction == 1 else "SHORT"
                logger.info(f"SIGNAL: {dir_str} | Conf: {confidence:.2f} | Net: {outflow - inflow:.2f} BTC")

    def stop(self):
        self.running = False


class APIValidationCollector:
    """Main collector orchestrator."""

    def __init__(self):
        self.price_collector = PriceCollector()
        self.signal_collector = BlockchainSignalCollector(self.price_collector)
        self.running = False
        self.start_time = None

        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

    def _shutdown(self, signum, frame):
        logger.info("Shutdown signal received...")
        self.running = False

    def start(self, duration_hours=None):
        logger.info("=" * 60)
        logger.info("API-BASED SIGNAL VALIDATION")
        logger.info("NO BITCOIN CORE REQUIRED")
        logger.info("=" * 60)
        logger.info(f"Duration: {duration_hours or 'unlimited'} hours")
        logger.info(f"Data dir: {DATA_DIR}")
        logger.info("")

        self.running = True
        self.start_time = time.time()

        # Start collectors
        self.price_collector.start_websocket()
        logger.info("Price collection started (Binance WebSocket)")

        self.signal_collector.start()
        logger.info("Blockchain signals started (mempool.space API)")

        end_time = None
        if duration_hours:
            end_time = self.start_time + (duration_hours * 3600)

        last_status = 0

        while self.running:
            now = time.time()

            if end_time and now >= end_time:
                logger.info("Duration reached")
                break

            if now - last_status >= 60:
                self._print_status()
                last_status = now

            time.sleep(1)

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

    collector = APIValidationCollector()
    collector.start(duration_hours=args.duration)


if __name__ == "__main__":
    main()
PYFILE

chmod +x $WORK_DIR/run_api_validation.py

echo ""
echo "[6/6] Creating systemd service..."
cat > /etc/systemd/system/api-validation.service << 'SVCFILE'
[Unit]
Description=API-Based Signal Validation Collector
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/validation
Environment=PATH=/root/validation/venv/bin:/usr/bin
ExecStart=/root/validation/venv/bin/python /root/validation/run_api_validation.py
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
echo "   python run_api_validation.py --duration 0.083"
echo ""
echo "2. Run for 1 week (recommended):"
echo "   systemctl enable api-validation"
echo "   systemctl start api-validation"
echo "   journalctl -u api-validation -f"
echo ""
echo "3. Check status:"
echo "   systemctl status api-validation"
echo ""
echo "4. After collection, transfer data to Windows:"
echo "   scp root@31.97.211.217:/root/validation/data/*.db C:\\Users\\kevin\\livetrading\\data\\validation\\"
echo ""
echo "5. Run analysis on Windows:"
echo "   python -m engine.sovereign.validation.analysis.calculate_edge"
echo ""
echo "=============================================="
echo "TIMELINE:"
echo "  Week 1-2: Collect data (this script)"
echo "  Week 3:   Analyze edge (calculate_edge.py)"
echo "  Week 4:   Paper trade (if edge confirmed)"
echo "  Week 5+:  Live trade (if paper confirms)"
echo "=============================================="
echo ""
echo "NO BITCOIN CORE REQUIRED!"
echo "This uses mempool.space FREE API for blockchain data."
echo "=============================================="
