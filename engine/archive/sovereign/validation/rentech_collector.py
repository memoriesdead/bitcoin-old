#!/usr/bin/env python3
"""
RenTech-Style Bitcoin Metric Collector
=======================================

The RenTech approach: Don't assume what matters. Collect EVERYTHING.
Let the data tell you what predicts price.

We have 10-60 seconds of information advantage. This collector extracts
every possible metric from raw transactions to find statistical edges.
"""

import os
import sys
import time
import json
import signal
import sqlite3
import threading
import struct
import hashlib
from pathlib import Path
from datetime import datetime
from collections import deque
from typing import Dict, Optional
import logging
import statistics

try:
    import zmq
    HAS_ZMQ = True
except ImportError:
    HAS_ZMQ = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

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
        logging.FileHandler('/root/validation/rentech.log')
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path("/root/validation/data")
METRICS_DB = DATA_DIR / "metrics.db"
ZMQ_RAWTX = "tcp://127.0.0.1:28332"

# Metric aggregation interval (seconds)
AGGREGATION_INTERVAL = 1.0

# =============================================================================
# TRANSACTION DECODER
# =============================================================================

def decode_varint(data: bytes, offset: int) -> tuple:
    val = data[offset]
    if val < 0xfd:
        return val, offset + 1
    elif val == 0xfd:
        return struct.unpack_from('<H', data, offset + 1)[0], offset + 3
    elif val == 0xfe:
        return struct.unpack_from('<I', data, offset + 1)[0], offset + 5
    else:
        return struct.unpack_from('<Q', data, offset + 1)[0], offset + 9


def decode_transaction(raw: bytes) -> Optional[Dict]:
    """Decode raw transaction for metrics extraction."""
    try:
        offset = 0
        version = struct.unpack_from('<I', raw, offset)[0]
        offset += 4

        # Check for SegWit
        is_segwit = False
        is_taproot = False
        if raw[offset] == 0x00 and raw[offset + 1] == 0x01:
            is_segwit = True
            offset += 2

        # Inputs
        in_count, offset = decode_varint(raw, offset)
        total_input_script_len = 0

        for _ in range(in_count):
            offset += 36  # prev txid + vout
            script_len, offset = decode_varint(raw, offset)
            total_input_script_len += script_len
            offset += script_len + 4  # script + sequence

        # Outputs
        out_count, offset = decode_varint(raw, offset)
        total_value = 0
        output_types = {'p2pkh': 0, 'p2sh': 0, 'p2wpkh': 0, 'p2wsh': 0, 'p2tr': 0, 'other': 0}

        for _ in range(out_count):
            value = struct.unpack_from('<Q', raw, offset)[0]
            total_value += value
            offset += 8

            script_len, offset = decode_varint(raw, offset)
            script = raw[offset:offset + script_len]
            offset += script_len

            # Classify output type
            if len(script) == 25 and script[0] == 0x76 and script[1] == 0xa9:
                output_types['p2pkh'] += 1
            elif len(script) == 23 and script[0] == 0xa9:
                output_types['p2sh'] += 1
            elif len(script) == 22 and script[0] == 0x00 and script[1] == 0x14:
                output_types['p2wpkh'] += 1
            elif len(script) == 34 and script[0] == 0x00 and script[1] == 0x20:
                output_types['p2wsh'] += 1
            elif len(script) == 34 and script[0] == 0x51 and script[1] == 0x20:
                output_types['p2tr'] += 1
                is_taproot = True
            else:
                output_types['other'] += 1

        # Calculate virtual size for fee estimation
        # Simplified: actual vsize calculation is more complex
        base_size = len(raw)
        vsize = base_size  # Simplified

        return {
            'size': len(raw),
            'vsize': vsize,
            'version': version,
            'is_segwit': is_segwit,
            'is_taproot': is_taproot,
            'input_count': in_count,
            'output_count': out_count,
            'total_value_sat': total_value,
            'total_value_btc': total_value / 1e8,
            'output_types': output_types,
        }

    except Exception as e:
        return None


# =============================================================================
# METRIC AGGREGATOR
# =============================================================================

class MetricAggregator:
    """Aggregates transaction metrics over time windows."""

    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

        # Rolling windows for z-score calculation
        self.tx_count_history = deque(maxlen=60)  # Last 60 seconds
        self.volume_history = deque(maxlen=60)
        self.whale_history = deque(maxlen=60)

    def reset(self):
        """Reset current aggregation window."""
        self.tx_count = 0
        self.total_volume_btc = 0.0
        self.total_size = 0
        self.total_vsize = 0
        self.input_count = 0
        self.output_count = 0

        # Transaction size buckets
        self.tx_small = 0      # < 0.1 BTC
        self.tx_medium = 0     # 0.1 - 1 BTC
        self.tx_large = 0      # 1 - 10 BTC
        self.tx_whale = 0      # 10 - 100 BTC
        self.tx_mega = 0       # > 100 BTC

        # Output type counts
        self.segwit_outputs = 0
        self.taproot_outputs = 0
        self.legacy_outputs = 0
        self.total_outputs = 0

        # Individual transaction values for stats
        self.tx_values = []

    def add_transaction(self, tx: Dict):
        """Add a decoded transaction to the current window."""
        with self._lock:
            self.tx_count += 1
            btc = tx['total_value_btc']
            self.total_volume_btc += btc
            self.total_size += tx['size']
            self.total_vsize += tx['vsize']
            self.input_count += tx['input_count']
            self.output_count += tx['output_count']

            # Size classification
            if btc < 0.1:
                self.tx_small += 1
            elif btc < 1:
                self.tx_medium += 1
            elif btc < 10:
                self.tx_large += 1
            elif btc < 100:
                self.tx_whale += 1
            else:
                self.tx_mega += 1

            # Output types
            ot = tx['output_types']
            self.segwit_outputs += ot['p2wpkh'] + ot['p2wsh']
            self.taproot_outputs += ot['p2tr']
            self.legacy_outputs += ot['p2pkh'] + ot['p2sh']
            self.total_outputs += sum(ot.values())

            self.tx_values.append(btc)

    def get_metrics(self) -> Dict:
        """Get aggregated metrics for the current window."""
        with self._lock:
            # Consolidation ratio (inputs / outputs)
            consolidation_ratio = self.input_count / max(1, self.output_count)

            # Average transaction size
            avg_tx_btc = self.total_volume_btc / max(1, self.tx_count)

            # Median transaction size
            median_tx_btc = statistics.median(self.tx_values) if self.tx_values else 0

            # Output type ratios
            segwit_ratio = self.segwit_outputs / max(1, self.total_outputs)
            taproot_ratio = self.taproot_outputs / max(1, self.total_outputs)
            legacy_ratio = self.legacy_outputs / max(1, self.total_outputs)

            # Update rolling history
            self.tx_count_history.append(self.tx_count)
            self.volume_history.append(self.total_volume_btc)
            self.whale_history.append(self.tx_whale + self.tx_mega)

            # Calculate z-scores (how unusual is this second?)
            tx_count_zscore = self._zscore(self.tx_count, self.tx_count_history)
            volume_zscore = self._zscore(self.total_volume_btc, self.volume_history)
            whale_zscore = self._zscore(self.tx_whale + self.tx_mega, self.whale_history)

            metrics = {
                # Raw counts
                'tx_count': self.tx_count,
                'total_volume_btc': self.total_volume_btc,
                'total_size_bytes': self.total_size,
                'input_count': self.input_count,
                'output_count': self.output_count,

                # Size distribution
                'tx_small': self.tx_small,
                'tx_medium': self.tx_medium,
                'tx_large': self.tx_large,
                'tx_whale': self.tx_whale,
                'tx_mega': self.tx_mega,

                # Derived metrics
                'consolidation_ratio': consolidation_ratio,
                'avg_tx_btc': avg_tx_btc,
                'median_tx_btc': median_tx_btc,

                # Output type ratios
                'segwit_ratio': segwit_ratio,
                'taproot_ratio': taproot_ratio,
                'legacy_ratio': legacy_ratio,

                # Z-scores (anomaly detection)
                'tx_count_zscore': tx_count_zscore,
                'volume_zscore': volume_zscore,
                'whale_zscore': whale_zscore,
            }

            return metrics

    def _zscore(self, value: float, history: deque) -> float:
        """Calculate z-score relative to recent history."""
        if len(history) < 10:
            return 0.0
        mean = statistics.mean(history)
        std = statistics.stdev(history) if len(history) > 1 else 1
        if std == 0:
            return 0.0
        return (value - mean) / std


# =============================================================================
# PRICE COLLECTOR
# =============================================================================

class PriceCollector:
    """Collect price data with history for return calculations."""

    def __init__(self):
        self.running = False
        self.current_price = 0.0
        self.current_bid = 0.0
        self.current_ask = 0.0
        self._lock = threading.Lock()

        # Price history for returns
        self.price_history = deque(maxlen=3600)  # Last hour
        self.last_update = 0

    def get_price_data(self) -> Dict:
        """Get current price and derived metrics."""
        with self._lock:
            now = time.time()

            # Calculate returns
            return_1m = self._calculate_return(60)
            return_5m = self._calculate_return(300)
            return_15m = self._calculate_return(900)

            spread = self.current_ask - self.current_bid if self.current_bid > 0 else 0
            spread_bps = (spread / self.current_price * 10000) if self.current_price > 0 else 0

            return {
                'price': self.current_price,
                'bid': self.current_bid,
                'ask': self.current_ask,
                'spread': spread,
                'spread_bps': spread_bps,
                'return_1m': return_1m,
                'return_5m': return_5m,
                'return_15m': return_15m,
            }

    def _calculate_return(self, seconds_ago: int) -> float:
        """Calculate return over the given period."""
        if not self.price_history or self.current_price == 0:
            return 0.0

        target_time = time.time() - seconds_ago
        old_price = None

        for ts, price in self.price_history:
            if ts <= target_time:
                old_price = price
                break

        if old_price and old_price > 0:
            return (self.current_price - old_price) / old_price
        return 0.0

    def update_price(self, price: float, bid: float = 0, ask: float = 0):
        """Update current price."""
        with self._lock:
            now = time.time()
            self.current_price = price
            self.current_bid = bid or price
            self.current_ask = ask or price
            self.price_history.appendleft((now, price))
            self.last_update = now

    def start(self):
        """Start price collection thread."""
        self.running = True
        thread = threading.Thread(target=self._ws_loop, daemon=True)
        thread.start()

    def _ws_loop(self):
        """WebSocket price feed loop."""
        while self.running:
            try:
                ws = websocket.WebSocketApp(
                    "wss://ws.kraken.com",
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=lambda ws, e: logger.error(f"WS error: {e}"),
                )
                ws.run_forever()
            except Exception as e:
                logger.error(f"WS failed: {e}")

            if self.running:
                time.sleep(5)

    def _on_open(self, ws):
        subscribe = {
            "event": "subscribe",
            "pair": ["XBT/USD"],
            "subscription": {"name": "ticker"}
        }
        ws.send(json.dumps(subscribe))
        logger.info("Price feed connected")

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
            if isinstance(data, list) and len(data) >= 2:
                ticker = data[1]
                if isinstance(ticker, dict) and 'c' in ticker:
                    price = float(ticker['c'][0])
                    bid = float(ticker['b'][0]) if 'b' in ticker else price
                    ask = float(ticker['a'][0]) if 'a' in ticker else price
                    self.update_price(price, bid, ask)
        except:
            pass

    def stop(self):
        self.running = False


# =============================================================================
# DATABASE
# =============================================================================

class MetricsDatabase:
    """SQLite database for metrics storage."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # Main metrics table - one row per second
        c.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                timestamp_ms INTEGER NOT NULL,

                -- Transaction metrics
                tx_count INTEGER,
                total_volume_btc REAL,
                total_size_bytes INTEGER,
                input_count INTEGER,
                output_count INTEGER,

                -- Size distribution
                tx_small INTEGER,
                tx_medium INTEGER,
                tx_large INTEGER,
                tx_whale INTEGER,
                tx_mega INTEGER,

                -- Derived metrics
                consolidation_ratio REAL,
                avg_tx_btc REAL,
                median_tx_btc REAL,
                segwit_ratio REAL,
                taproot_ratio REAL,
                legacy_ratio REAL,

                -- Z-scores
                tx_count_zscore REAL,
                volume_zscore REAL,
                whale_zscore REAL,

                -- Price metrics
                price REAL,
                bid REAL,
                ask REAL,
                spread_bps REAL,
                return_1m REAL,
                return_5m REAL,
                return_15m REAL
            )
        """)

        c.execute("CREATE INDEX IF NOT EXISTS idx_metrics_ts ON metrics(timestamp)")
        conn.commit()
        conn.close()

    def insert_metrics(self, tx_metrics: Dict, price_metrics: Dict):
        """Insert a metrics row."""
        now = time.time()
        now_ms = int(now * 1000)

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute("""
            INSERT INTO metrics (
                timestamp, timestamp_ms,
                tx_count, total_volume_btc, total_size_bytes, input_count, output_count,
                tx_small, tx_medium, tx_large, tx_whale, tx_mega,
                consolidation_ratio, avg_tx_btc, median_tx_btc,
                segwit_ratio, taproot_ratio, legacy_ratio,
                tx_count_zscore, volume_zscore, whale_zscore,
                price, bid, ask, spread_bps, return_1m, return_5m, return_15m
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            now, now_ms,
            tx_metrics['tx_count'],
            tx_metrics['total_volume_btc'],
            tx_metrics['total_size_bytes'],
            tx_metrics['input_count'],
            tx_metrics['output_count'],
            tx_metrics['tx_small'],
            tx_metrics['tx_medium'],
            tx_metrics['tx_large'],
            tx_metrics['tx_whale'],
            tx_metrics['tx_mega'],
            tx_metrics['consolidation_ratio'],
            tx_metrics['avg_tx_btc'],
            tx_metrics['median_tx_btc'],
            tx_metrics['segwit_ratio'],
            tx_metrics['taproot_ratio'],
            tx_metrics['legacy_ratio'],
            tx_metrics['tx_count_zscore'],
            tx_metrics['volume_zscore'],
            tx_metrics['whale_zscore'],
            price_metrics['price'],
            price_metrics['bid'],
            price_metrics['ask'],
            price_metrics['spread_bps'],
            price_metrics['return_1m'],
            price_metrics['return_5m'],
            price_metrics['return_15m'],
        ))

        conn.commit()
        conn.close()

    def get_row_count(self) -> int:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM metrics")
        count = c.fetchone()[0]
        conn.close()
        return count


# =============================================================================
# ZMQ COLLECTOR
# =============================================================================

class ZMQMetricCollector:
    """Collects metrics from Bitcoin Core ZMQ."""

    def __init__(self, aggregator: MetricAggregator):
        self.aggregator = aggregator
        self.running = False
        self.txs_processed = 0
        self._lock = threading.Lock()

    def start(self):
        self.running = True
        thread = threading.Thread(target=self._zmq_loop, daemon=True)
        thread.start()
        logger.info(f"ZMQ connected to {ZMQ_RAWTX}")

    def _zmq_loop(self):
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.setsockopt_string(zmq.SUBSCRIBE, 'rawtx')
        socket.connect(ZMQ_RAWTX)

        while self.running:
            try:
                msg = socket.recv_multipart()
                if len(msg) >= 2 and msg[0].decode('utf-8', errors='ignore') == 'rawtx':
                    raw_tx = msg[1]
                    tx = decode_transaction(raw_tx)
                    if tx:
                        self.aggregator.add_transaction(tx)
                        with self._lock:
                            self.txs_processed += 1
            except zmq.ZMQError as e:
                if self.running:
                    logger.error(f"ZMQ error: {e}")
                    time.sleep(1)

    def get_tx_count(self) -> int:
        with self._lock:
            return self.txs_processed

    def stop(self):
        self.running = False


# =============================================================================
# MAIN COLLECTOR
# =============================================================================

class RenTechCollector:
    """Main orchestrator for RenTech-style data collection."""

    def __init__(self):
        self.aggregator = MetricAggregator()
        self.price_collector = PriceCollector()
        self.zmq_collector = ZMQMetricCollector(self.aggregator)
        self.db = MetricsDatabase(METRICS_DB)

        self.running = False
        self.start_time = None

        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

    def _shutdown(self, signum, frame):
        logger.info("Shutdown signal received...")
        self.running = False

    def start(self, duration_hours: float = None):
        logger.info("=" * 60)
        logger.info("RENTECH-STYLE METRIC COLLECTION")
        logger.info("Collect everything. Let data reveal the edge.")
        logger.info("=" * 60)
        logger.info(f"Duration: {duration_hours or 'unlimited'} hours")
        logger.info(f"Database: {METRICS_DB}")
        logger.info("")

        self.running = True
        self.start_time = time.time()

        # Start collectors
        self.price_collector.start()
        self.zmq_collector.start()

        logger.info("LIVE - Collecting blockchain metrics")
        logger.info("")

        end_time = None
        if duration_hours:
            end_time = self.start_time + (duration_hours * 3600)

        last_aggregate = time.time()
        last_status = time.time()

        while self.running:
            now = time.time()

            if end_time and now >= end_time:
                logger.info("Duration reached")
                break

            # Aggregate and store metrics every second
            if now - last_aggregate >= AGGREGATION_INTERVAL:
                tx_metrics = self.aggregator.get_metrics()
                price_metrics = self.price_collector.get_price_data()

                self.db.insert_metrics(tx_metrics, price_metrics)
                self.aggregator.reset()
                last_aggregate = now

            # Status every 60 seconds
            if now - last_status >= 60:
                self._print_status()
                last_status = now

            time.sleep(0.1)

        self._print_final()

    def _print_status(self):
        runtime = (time.time() - self.start_time) / 3600
        rows = self.db.get_row_count()
        txs = self.zmq_collector.get_tx_count()
        price = self.price_collector.current_price

        logger.info(f"[{runtime:.2f}h] TXs: {txs:,} | Rows: {rows:,} | BTC: ${price:,.0f}")

    def _print_final(self):
        runtime = (time.time() - self.start_time) / 3600
        rows = self.db.get_row_count()
        txs = self.zmq_collector.get_tx_count()

        logger.info("")
        logger.info("=" * 60)
        logger.info("COLLECTION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Runtime: {runtime:.2f} hours")
        logger.info(f"Transactions processed: {txs:,}")
        logger.info(f"Metric rows stored: {rows:,}")
        logger.info(f"Database: {METRICS_DB}")
        logger.info("")
        logger.info("Next: Run analysis to find predictive features")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=None)
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    collector = RenTechCollector()
    collector.start(duration_hours=args.duration)


if __name__ == "__main__":
    main()
