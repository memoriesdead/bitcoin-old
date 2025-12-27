#!/usr/bin/env python3
"""
CORRELATION COLLECTOR PIPELINE
==============================
This pipeline does ONE thing: Collect data to build the math.

NO TRADING. ONLY DATA COLLECTION.

For every blockchain flow we detect:
1. Record the flow (exchange, direction, amount)
2. Record price at T=0
3. Check price at T+1m, T+5m, T+15m, T+30m, T+60m
4. Store all data in correlation database

After collecting enough data (1000+ events per exchange):
- Calculate correlations
- Build deterministic formulas
- THEN we can trade

Usage:
    python3 correlation_collector_pipeline.py

Run for 24-48 hours to collect enough data.
"""

import subprocess
import sys
import os
import re
import time
import signal
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Dict, Optional
from dataclasses import dataclass

# Add paths
sys.path.insert(0, '/root/sovereign')
sys.path.insert(0, '/root/sovereign/blockchain')

# Import correlation engine
from deterministic_correlation import (
    DeterministicCorrelationDB,
    CorrelationCollector,
    CorrelationConfig
)

# Import price feed
try:
    import ccxt
    HAS_CCXT = True
except ImportError:
    HAS_CCXT = False
    print("Warning: ccxt not installed. Install with: pip install ccxt")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class CollectorConfig:
    """Configuration for data collection."""
    # C++ binary path
    cpp_binary: str = "/root/sovereign/cpp_runner/build/blockchain_runner"

    # Database paths
    address_db: str = "/root/sovereign/walletexplorer_addresses.db"
    utxo_db: str = "/root/sovereign/exchange_utxos.db"
    correlation_db: str = "/root/sovereign/deterministic_correlation.db"

    # ZMQ endpoint
    zmq_endpoint: str = "tcp://127.0.0.1:28332"

    # Minimum flow to track (reduce noise)
    min_flow_btc: float = 0.5

    # Print stats every N seconds
    stats_interval: int = 60


# =============================================================================
# PRICE FETCHER (Using CCXT)
# =============================================================================

class PriceFetcher:
    """Fetches current BTC price from multiple exchanges."""

    def __init__(self):
        self.exchanges = {}
        self.prices = {}
        self.lock = threading.Lock()
        self._init_exchanges()
        self._start_price_thread()

    def _init_exchanges(self):
        """Initialize CCXT exchange connections."""
        if not HAS_CCXT:
            return

        exchange_ids = [
            'binance', 'coinbase', 'kraken', 'bitstamp', 'gemini',
            'bitfinex', 'okx', 'bybit', 'huobi', 'kucoin', 'gateio'
        ]

        for ex_id in exchange_ids:
            try:
                exchange_class = getattr(ccxt, ex_id)
                self.exchanges[ex_id] = exchange_class({'enableRateLimit': True})
                print(f"  [OK] {ex_id}")
            except Exception as e:
                print(f"  [FAIL] {ex_id}: {e}")

    def _start_price_thread(self):
        """Start background price update thread."""
        def update_loop():
            while True:
                for ex_id, exchange in self.exchanges.items():
                    try:
                        ticker = exchange.fetch_ticker('BTC/USDT')
                        with self.lock:
                            self.prices[ex_id] = ticker['last']
                    except:
                        try:
                            ticker = exchange.fetch_ticker('BTC/USD')
                            with self.lock:
                                self.prices[ex_id] = ticker['last']
                        except:
                            pass
                time.sleep(5)

        thread = threading.Thread(target=update_loop, daemon=True)
        thread.start()

    def get_price(self, exchange: str) -> Optional[float]:
        """Get current price for exchange."""
        with self.lock:
            # Try exact match
            if exchange in self.prices:
                return self.prices[exchange]

            # Try aliases
            aliases = {
                'huobi': 'htx',
                'htx': 'huobi',
                'gate.io': 'gateio',
                'crypto.com': 'cryptocom'
            }

            if exchange in aliases:
                alt = aliases[exchange]
                if alt in self.prices:
                    return self.prices[alt]

            # Return any price (market is connected)
            if self.prices:
                return list(self.prices.values())[0]

            return None


# =============================================================================
# SIGNAL PARSER
# =============================================================================

def parse_cpp_signal(line: str) -> Optional[Dict]:
    """
    Parse a signal line from C++ runner output.

    Format: [LONG/SHORT] exchange1, exchange2 | In: X.XX | Out: Y.YY | Net: +/-Z.ZZ | Latency: Nns
    """
    # Pattern with ANSI codes
    pattern = r'\[(LONG|SHORT)\]\x1b\[0m\s+(.+?)\s*\|\s*In:\s*([\d.]+)\s*\|\s*Out:\s*([\d.]+)\s*\|\s*Net:\s*([+-]?[\d.]+)\s*\|\s*Latency:\s*(\d+)ns'

    # Try without ANSI codes
    if '\x1b' not in line:
        pattern = r'\[(LONG|SHORT)\]\s+(.+?)\s*\|\s*In:\s*([\d.]+)\s*\|\s*Out:\s*([\d.]+)\s*\|\s*Net:\s*([+-]?[\d.]+)\s*\|\s*Latency:\s*(\d+)ns'

    match = re.search(pattern, line)
    if not match:
        return None

    direction = match.group(1)
    exchanges = [e.strip() for e in match.group(2).split(',')]
    inflow = float(match.group(3))
    outflow = float(match.group(4))
    net = float(match.group(5))
    latency = int(match.group(6))

    # Determine flow direction based on net
    if net < 0:
        flow_direction = 'inflow'
        amount = abs(net)
    else:
        flow_direction = 'outflow'
        amount = abs(net)

    return {
        'signal': direction,
        'exchanges': exchanges,
        'inflow': inflow,
        'outflow': outflow,
        'net': net,
        'flow_direction': flow_direction,
        'amount': amount,
        'latency_ns': latency
    }


# =============================================================================
# MAIN COLLECTOR
# =============================================================================

class DataCollectorPipeline:
    """Main data collection pipeline."""

    def __init__(self, config: CollectorConfig = None):
        self.config = config or CollectorConfig()
        self.running = False

        # Initialize correlation database
        corr_config = CorrelationConfig(
            db_path=self.config.correlation_db,
            min_flow_btc=self.config.min_flow_btc
        )
        self.corr_db = DeterministicCorrelationDB(corr_config)

        # Initialize price fetcher
        print("\nInitializing price feeds...")
        self.price_fetcher = PriceFetcher()

        # Initialize correlation collector
        self.collector = CorrelationCollector(
            db=self.corr_db,
            price_fetcher=self.price_fetcher.get_price
        )

        # Statistics
        self.flows_detected = 0
        self.flows_recorded = 0
        self.start_time = None

    def start(self):
        """Start the data collection pipeline."""
        self.running = True
        self.start_time = time.time()

        # Start background collector
        self.collector.start()

        # Print header
        print()
        print("=" * 70)
        print("CORRELATION DATA COLLECTOR")
        print("=" * 70)
        print("Mode: DATA COLLECTION ONLY (No Trading)")
        print(f"Database: {self.config.correlation_db}")
        print(f"Min flow: {self.config.min_flow_btc} BTC")
        print("=" * 70)
        print()
        print("Waiting for C++ signals...")
        print()

        # Start stats thread
        stats_thread = threading.Thread(target=self._stats_loop, daemon=True)
        stats_thread.start()

        # Run C++ pipeline
        self._run_cpp_pipeline()

    def _run_cpp_pipeline(self):
        """Run C++ blockchain runner and parse output."""
        cmd = [
            self.config.cpp_binary,
            "--address-db", self.config.address_db,
            "--utxo-db", self.config.utxo_db,
            "--zmq", self.config.zmq_endpoint
        ]

        print(f"Running: {' '.join(cmd)}")
        print()

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            for line in process.stdout:
                if not self.running:
                    break

                line = line.rstrip()

                # Parse signal
                signal = parse_cpp_signal(line)

                if signal:
                    self._handle_signal(signal)
                else:
                    # Print non-signal output
                    print(line)

        except KeyboardInterrupt:
            print("\nShutting down...")
            self.running = False
        except Exception as e:
            print(f"Error: {e}")
            self.running = False

    def _handle_signal(self, signal: Dict):
        """Handle a detected flow signal."""
        self.flows_detected += 1

        # Record flow for each exchange
        for exchange in signal['exchanges']:
            self.collector.on_flow(
                exchange=exchange,
                direction=signal['flow_direction'],
                amount_btc=signal['amount']
            )
            self.flows_recorded += 1

        # Print signal
        ex_str = ', '.join(signal['exchanges'])
        direction = signal['flow_direction'].upper()
        print(f"[{direction}] {ex_str} | {signal['amount']:.2f} BTC | "
              f"Latency: {signal['latency_ns']}ns")

    def _stats_loop(self):
        """Print statistics periodically."""
        while self.running:
            time.sleep(self.config.stats_interval)

            if not self.running:
                break

            elapsed = time.time() - self.start_time
            hours = elapsed / 3600

            print()
            print("-" * 50)
            print(f"DATA COLLECTION STATS ({hours:.1f} hours)")
            print("-" * 50)
            print(f"  Flows detected: {self.flows_detected}")
            print(f"  Flows recorded: {self.flows_recorded}")
            print()

            # Get database stats
            report = self.corr_db.get_statistics_report()
            for line in report.split('\n')[:20]:
                print(line)
            print("-" * 50)
            print()

    def stop(self):
        """Stop the pipeline."""
        self.running = False
        self.collector.stop()
        self.corr_db.close()


# =============================================================================
# SIGNAL HANDLERS
# =============================================================================

def signal_handler(sig, frame):
    """Handle Ctrl+C."""
    print("\nReceived shutdown signal...")
    sys.exit(0)


# =============================================================================
# MAIN
# =============================================================================

def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("=" * 70)
    print("DETERMINISTIC CORRELATION DATA COLLECTOR")
    print("=" * 70)
    print()
    print("This script ONLY collects data. NO trading.")
    print("Run for 24-48 hours to collect enough data for correlations.")
    print()

    pipeline = DataCollectorPipeline()

    try:
        pipeline.start()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        pipeline.stop()
        print()
        print("Final statistics:")
        print(pipeline.corr_db.get_statistics_report())


if __name__ == "__main__":
    main()
