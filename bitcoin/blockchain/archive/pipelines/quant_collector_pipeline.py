#!/usr/bin/env python3
"""
QUANT COLLECTOR PIPELINE
========================

Integrated pipeline that:
1. Collects blockchain flow data
2. Applies gold-standard quant formulas (Kyle, Almgren-Chriss, VPIN)
3. Records predictions vs actual outcomes
4. Continuously calibrates the models

This is the data collection phase. Run for 24-48 hours to:
- Collect enough samples per exchange
- Calibrate the price impact coefficients
- Validate the mathematical models

AFTER calibration, you get EXACT deterministic formulas:
    ΔP = λ × NetFlow × Price

Where λ is calibrated per exchange from real data.
"""

import subprocess
import sys
import os
import re
import time
import signal
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, List
from dataclasses import dataclass

# Add paths
sys.path.insert(0, '/root/sovereign')
sys.path.insert(0, '/root/sovereign/blockchain')

# Import quant models
from quant_price_impact import (
    DeterministicPricePredictor,
    PriceImpactConfig
)

# Import price feeds
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
class QuantCollectorConfig:
    """Configuration for quant data collection."""

    # C++ binary
    cpp_binary: str = "/root/sovereign/cpp_runner/build/blockchain_runner"

    # Database paths
    address_db: str = "/root/sovereign/walletexplorer_addresses.db"
    utxo_db: str = "/root/sovereign/exchange_utxos.db"
    quant_db: str = "/root/sovereign/quant_price_impact.db"

    # ZMQ endpoint
    zmq_endpoint: str = "tcp://127.0.0.1:28332"

    # Thresholds
    min_flow_btc: float = 1.0  # Only track >= 1 BTC flows

    # Price check intervals
    price_intervals: List[int] = None  # [60, 300, 900, 1800] seconds

    # Stats interval
    stats_interval: int = 120  # Print stats every 2 minutes

    # Calibration interval
    calibration_interval: int = 3600  # Recalibrate every hour

    def __post_init__(self):
        if self.price_intervals is None:
            self.price_intervals = [60, 300, 900, 1800]


# =============================================================================
# PRICE FETCHER
# =============================================================================

class MultiExchangePriceFetcher:
    """Fetches BTC prices from multiple exchanges."""

    def __init__(self):
        self.exchanges: Dict[str, any] = {}
        self.prices: Dict[str, float] = {}
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

        print("Connecting to exchanges...")
        for ex_id in exchange_ids:
            try:
                exchange_class = getattr(ccxt, ex_id)
                self.exchanges[ex_id] = exchange_class({
                    'enableRateLimit': True,
                    'timeout': 10000
                })
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
        """Get current price, with fallback to any available."""
        with self.lock:
            # Try exact match
            if exchange in self.prices:
                return self.prices[exchange]

            # Try common aliases
            aliases = {
                'huobi': 'htx', 'htx': 'huobi',
                'gate.io': 'gateio', 'crypto.com': 'cryptocom'
            }
            if exchange in aliases and aliases[exchange] in self.prices:
                return self.prices[aliases[exchange]]

            # Return any price as fallback
            if self.prices:
                return list(self.prices.values())[0]

            return None

    def get_all_prices(self) -> Dict[str, float]:
        """Get all current prices."""
        with self.lock:
            return dict(self.prices)


# =============================================================================
# SIGNAL PARSER
# =============================================================================

def parse_cpp_signal(line: str) -> Optional[Dict]:
    """
    Parse a signal line from C++ runner output.

    Format: [LONG/SHORT] exchange1, exchange2 | In: X.XX | Out: Y.YY | Net: +/-Z.ZZ | Latency: Nns
    """
    # Try with ANSI codes
    if '\x1b' in line:
        pattern = r'\[(LONG|SHORT)\]\x1b\[0m\s+(.+?)\s*\|\s*In:\s*([\d.]+)\s*\|\s*Out:\s*([\d.]+)\s*\|\s*Net:\s*([+-]?[\d.]+)\s*\|\s*Latency:\s*(\d+)ns'
    else:
        pattern = r'\[(LONG|SHORT)\]\s+(.+?)\s*\|\s*In:\s*([\d.]+)\s*\|\s*Out:\s*([\d.]+)\s*\|\s*Net:\s*([+-]?[\d.]+)\s*\|\s*Latency:\s*(\d+)ns'

    match = re.search(pattern, line)
    if not match:
        return None

    direction_signal = match.group(1)
    exchanges = [e.strip() for e in match.group(2).split(',')]
    inflow = float(match.group(3))
    outflow = float(match.group(4))
    net = float(match.group(5))
    latency = int(match.group(6))

    # Flow direction from net
    if net < 0:
        flow_direction = 'inflow'
        amount = abs(net)
    else:
        flow_direction = 'outflow'
        amount = abs(net)

    return {
        'signal': direction_signal,
        'exchanges': exchanges,
        'inflow': inflow,
        'outflow': outflow,
        'net': net,
        'flow_direction': flow_direction,
        'amount': amount,
        'latency_ns': latency
    }


# =============================================================================
# PRICE OBSERVER (Background thread to check prices at intervals)
# =============================================================================

@dataclass
class PendingObservation:
    """Pending price observation."""
    prediction_id: int
    exchange: str
    flow_time: datetime
    check_times: Dict[int, datetime]  # interval -> check time
    completed: Dict[int, bool]  # interval -> done


class PriceObserver:
    """Background thread that checks prices at scheduled intervals."""

    def __init__(self, predictor: DeterministicPricePredictor,
                 price_fetcher: MultiExchangePriceFetcher,
                 intervals: List[int]):
        self.predictor = predictor
        self.price_fetcher = price_fetcher
        self.intervals = intervals  # [60, 300, 900, 1800]
        self.pending: List[PendingObservation] = []
        self.lock = threading.Lock()
        self.running = False

    def schedule(self, prediction_id: int, exchange: str):
        """Schedule price observations for a prediction."""
        now = datetime.now(timezone.utc)
        obs = PendingObservation(
            prediction_id=prediction_id,
            exchange=exchange,
            flow_time=now,
            check_times={i: now + timedelta(seconds=i) for i in self.intervals},
            completed={i: False for i in self.intervals}
        )
        with self.lock:
            self.pending.append(obs)

    def start(self):
        """Start the observer thread."""
        self.running = True
        thread = threading.Thread(target=self._observation_loop, daemon=True)
        thread.start()

    def stop(self):
        """Stop the observer."""
        self.running = False

    def _observation_loop(self):
        """Main observation loop."""
        while self.running:
            now = datetime.now(timezone.utc)

            with self.lock:
                for obs in self.pending:
                    for interval, check_time in obs.check_times.items():
                        if not obs.completed[interval] and now >= check_time:
                            # Time to check price
                            price = self.price_fetcher.get_price(obs.exchange)
                            if price:
                                # Record this observation
                                key = f'{interval // 60}m'
                                prices = {key: price}
                                self.predictor.record_outcome(obs.prediction_id, prices)
                                obs.completed[interval] = True

                # Clean up completed observations
                self.pending = [
                    obs for obs in self.pending
                    if not all(obs.completed.values())
                ]

            time.sleep(1)

    def get_pending_count(self) -> int:
        """Get count of pending observations."""
        with self.lock:
            return len(self.pending)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

class QuantCollectorPipeline:
    """Main quant data collection pipeline."""

    def __init__(self, config: QuantCollectorConfig = None):
        self.config = config or QuantCollectorConfig()
        self.running = False

        # Initialize quant predictor
        impact_config = PriceImpactConfig(
            db_path=self.config.quant_db,
            min_flow_btc=self.config.min_flow_btc
        )
        self.predictor = DeterministicPricePredictor(impact_config)

        # Initialize price fetcher
        print("\nInitializing price feeds...")
        self.price_fetcher = MultiExchangePriceFetcher()

        # Wait for initial prices
        print("Waiting for initial price data...")
        time.sleep(5)
        print(f"Got prices for {len(self.price_fetcher.prices)} exchanges")

        # Initialize price observer
        self.observer = PriceObserver(
            self.predictor,
            self.price_fetcher,
            self.config.price_intervals
        )

        # Statistics
        self.flows_detected = 0
        self.predictions_made = 0
        self.start_time = None

    def start(self):
        """Start the pipeline."""
        self.running = True
        self.start_time = time.time()

        # Start price observer
        self.observer.start()

        # Print header
        print()
        print("=" * 70)
        print("QUANT PRICE IMPACT COLLECTOR")
        print("=" * 70)
        print("Mode: DATA COLLECTION + PREDICTION")
        print()
        print("Models active:")
        print("  - Kyle's Lambda (price impact coefficient)")
        print("  - Almgren-Chriss (permanent/temporary impact)")
        print("  - VPIN (order flow toxicity)")
        print("  - Order Flow Imbalance")
        print()
        print(f"Database: {self.config.quant_db}")
        print(f"Min flow: {self.config.min_flow_btc} BTC")
        print("=" * 70)
        print()

        # Start stats thread
        stats_thread = threading.Thread(target=self._stats_loop, daemon=True)
        stats_thread.start()

        # Start calibration thread
        calibration_thread = threading.Thread(target=self._calibration_loop, daemon=True)
        calibration_thread.start()

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

        print(f"Starting C++ runner: {self.config.cpp_binary}")
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
                    # Print other output
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

        for exchange in signal['exchanges']:
            # Get current price
            price = self.price_fetcher.get_price(exchange)
            if not price:
                continue

            # Update predictor with current price
            self.predictor.update_price(exchange, price)

            # Generate prediction
            result = self.predictor.on_flow(
                exchange=exchange,
                direction=signal['flow_direction'],
                amount_btc=signal['amount'],
                current_price=price
            )

            if result.get('status') == 'skipped':
                continue

            self.predictions_made += 1

            # Schedule price observations
            self.observer.schedule(result['id'], exchange)

            # Print prediction
            direction = signal['flow_direction'].upper()
            pred_dir = result['expected_direction']
            pred_amt = result['predictions']['combined']

            print(f"[{direction}] {exchange} | {signal['amount']:.2f} BTC | "
                  f"Price: ${price:,.0f}")
            print(f"         → Predicted: {pred_dir} ${abs(pred_amt):.2f} | "
                  f"VPIN: {result['predictions']['vpin']:.3f}")

    def _stats_loop(self):
        """Print statistics periodically."""
        while self.running:
            time.sleep(self.config.stats_interval)

            if not self.running:
                break

            elapsed = time.time() - self.start_time
            hours = elapsed / 3600

            print()
            print("=" * 60)
            print(f"QUANT COLLECTOR STATS ({hours:.1f} hours)")
            print("=" * 60)
            print(f"Flows detected: {self.flows_detected}")
            print(f"Predictions made: {self.predictions_made}")
            print(f"Pending observations: {self.observer.get_pending_count()}")
            print()

            # Get model statistics
            stats = self.predictor.get_statistics()
            print(stats)

            # Print current prices
            prices = self.price_fetcher.get_all_prices()
            if prices:
                avg_price = sum(prices.values()) / len(prices)
                print(f"Current BTC: ${avg_price:,.0f} (avg of {len(prices)} exchanges)")

            print("=" * 60)
            print()

    def _calibration_loop(self):
        """Recalibrate models periodically."""
        while self.running:
            time.sleep(self.config.calibration_interval)

            if not self.running:
                break

            print()
            print("[CALIBRATION] Recalibrating models from collected data...")
            self.predictor.calibrate_from_data()
            print("[CALIBRATION] Complete")
            print()

    def stop(self):
        """Stop the pipeline."""
        self.running = False
        self.observer.stop()
        self.predictor.close()


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
    print("GOLD-STANDARD QUANT PRICE IMPACT COLLECTOR")
    print("=" * 70)
    print()
    print("Academic models implemented:")
    print("  1. Kyle's Lambda (1985) - Price impact coefficient")
    print("  2. Almgren-Chriss (2001) - Permanent/temporary impact")
    print("  3. VPIN (2012) - Order flow toxicity")
    print("  4. Order Flow Imbalance (2014)")
    print()
    print("Run for 24-48 hours to collect data and calibrate.")
    print("After calibration: EXACT deterministic price formulas.")
    print()

    pipeline = QuantCollectorPipeline()

    try:
        pipeline.start()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pipeline.stop()

        print()
        print("=" * 60)
        print("FINAL STATISTICS")
        print("=" * 60)
        print(pipeline.predictor.get_statistics())

        # Print calibrated formulas
        for exchange in ['binance', 'coinbase', 'kraken']:
            print(pipeline.predictor.get_formula(exchange))


if __name__ == "__main__":
    main()
