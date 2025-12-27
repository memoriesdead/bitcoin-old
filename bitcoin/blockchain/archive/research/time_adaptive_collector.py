#!/usr/bin/env python3
"""
TIME-ADAPTIVE COLLECTOR PIPELINE
=================================

Collects blockchain flow data and observes price at MULTIPLE time points
to calibrate the time-dependent decay model.

The key insight: What works at 1 second may not work at 5 minutes.

OBSERVATION SCHEDULE:
---------------------
For each flow detected, we observe price at:
  t = [1, 5, 10, 30, 60, 120, 300, 600, 1800] seconds

This gives us the decay curve which we fit to:
  ΔP(t) = γ × F + η × F × t^(-β)

Once calibrated, we can predict impact at ANY time horizon.
"""

import subprocess
import sys
import os
import re
import time
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, List
from dataclasses import dataclass, field

sys.path.insert(0, '/root/sovereign/blockchain')

try:
    import ccxt
    HAS_CCXT = True
except ImportError:
    HAS_CCXT = False

# Direct import (avoid package-level imports)
from time_adaptive_impact import TimeAdaptiveImpactModel, TimeAdaptiveConfig


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class CollectorConfig:
    """Configuration for the collector."""
    cpp_binary: str = "/root/sovereign/cpp_runner/build/blockchain_runner"
    address_db: str = "/root/sovereign/walletexplorer_addresses.db"
    utxo_db: str = "/root/sovereign/exchange_utxos.db"
    zmq_endpoint: str = "tcp://127.0.0.1:28332"

    min_flow_btc: float = 10.0  # Proven threshold

    # Observation time points
    time_points: List[int] = field(default_factory=lambda: [
        1, 5, 10, 30, 60, 120, 300, 600, 1800
    ])

    # Calibration interval (seconds)
    calibration_interval: int = 1800  # Every 30 minutes

    # Stats interval
    stats_interval: int = 120  # Every 2 minutes


# =============================================================================
# PRICE FETCHER
# =============================================================================

class PriceFetcher:
    """Multi-exchange price fetcher."""

    def __init__(self):
        self.exchanges = {}
        self.prices = {}
        self.lock = threading.Lock()
        self._init()

    def _init(self):
        if not HAS_CCXT:
            print("WARNING: ccxt not installed")
            return

        exchange_ids = [
            'binance', 'coinbase', 'kraken', 'bitstamp', 'gemini',
            'bitfinex', 'okx', 'bybit', 'huobi', 'kucoin'
        ]

        print("Connecting to exchanges...")
        for ex_id in exchange_ids:
            try:
                self.exchanges[ex_id] = getattr(ccxt, ex_id)({'enableRateLimit': True})
                print(f"  [OK] {ex_id}")
            except Exception as e:
                print(f"  [FAIL] {ex_id}")

        # Start update thread
        def update():
            while True:
                for ex_id, ex in self.exchanges.items():
                    try:
                        ticker = ex.fetch_ticker('BTC/USDT')
                        with self.lock:
                            self.prices[ex_id] = ticker['last']
                    except:
                        try:
                            ticker = ex.fetch_ticker('BTC/USD')
                            with self.lock:
                                self.prices[ex_id] = ticker['last']
                        except:
                            pass
                time.sleep(2)

        threading.Thread(target=update, daemon=True).start()

    def get_price(self, exchange: str = None) -> Optional[float]:
        with self.lock:
            if exchange and exchange in self.prices:
                return self.prices[exchange]
            if self.prices:
                return list(self.prices.values())[0]
            return None


# =============================================================================
# PENDING OBSERVATION SCHEDULER
# =============================================================================

@dataclass
class PendingObservation:
    """Tracks pending price observations for a flow."""
    obs_id: int
    exchange: str
    start_time: float
    time_points: List[int]
    completed: set = field(default_factory=set)


class ObservationScheduler:
    """Schedules and executes price observations at multiple time points."""

    def __init__(self, model: TimeAdaptiveImpactModel,
                 price_fetcher: PriceFetcher,
                 time_points: List[int]):
        self.model = model
        self.price_fetcher = price_fetcher
        self.time_points = time_points
        self.pending: List[PendingObservation] = []
        self.lock = threading.Lock()
        self.running = False

    def schedule(self, obs_id: int, exchange: str):
        """Schedule observations for a new flow."""
        obs = PendingObservation(
            obs_id=obs_id,
            exchange=exchange,
            start_time=time.time(),
            time_points=self.time_points.copy()
        )
        with self.lock:
            self.pending.append(obs)

    def start(self):
        """Start the observation thread."""
        self.running = True
        threading.Thread(target=self._observation_loop, daemon=True).start()

    def stop(self):
        self.running = False

    def _observation_loop(self):
        """Main observation loop - checks every 100ms."""
        while self.running:
            time.sleep(0.1)
            now = time.time()

            with self.lock:
                to_remove = []

                for obs in self.pending:
                    elapsed = now - obs.start_time

                    for t in obs.time_points:
                        if t not in obs.completed and elapsed >= t:
                            # Time to observe!
                            price = self.price_fetcher.get_price(obs.exchange)
                            if price:
                                self.model.record_price_observation(obs.obs_id, t, price)
                                obs.completed.add(t)

                                # Log observation
                                label = f"{t}s" if t < 60 else f"{t//60}m"
                                print(f"    [OBS] {obs.exchange} t={label}: ${price:,.2f}")

                    # Remove if all observations complete
                    if obs.completed >= set(obs.time_points):
                        to_remove.append(obs)

                for obs in to_remove:
                    self.pending.remove(obs)

    def get_pending_count(self) -> int:
        with self.lock:
            return len(self.pending)


# =============================================================================
# MAIN COLLECTOR
# =============================================================================

class TimeAdaptiveCollector:
    """Main collector pipeline."""

    def __init__(self, config: CollectorConfig = None):
        self.config = config or CollectorConfig()
        self.running = False

        # Initialize model
        model_config = TimeAdaptiveConfig(
            db_path="/root/sovereign/time_adaptive_impact.db",
            time_points=self.config.time_points
        )
        self.model = TimeAdaptiveImpactModel(model_config)

        # Initialize price fetcher
        print("\nInitializing price feeds...")
        self.price_fetcher = PriceFetcher()
        time.sleep(3)

        # Initialize scheduler
        self.scheduler = ObservationScheduler(
            self.model,
            self.price_fetcher,
            self.config.time_points
        )

        # Statistics
        self.flows_detected = 0
        self.observations_scheduled = 0
        self.start_time = None

    def start(self):
        """Start the collector."""
        self.running = True
        self.start_time = time.time()

        # Start scheduler
        self.scheduler.start()

        # Print header
        print()
        print("=" * 70)
        print("TIME-ADAPTIVE PRICE IMPACT COLLECTOR")
        print("=" * 70)
        print()
        print("MATHEMATICAL MODEL: ΔP(t) = γF + ηF × t^(-β)")
        print()
        print("  γ = Permanent impact (information content)")
        print("  η = Temporary impact (liquidity displacement)")
        print("  β = Decay exponent (typically ~0.5)")
        print()
        print("OBSERVATION SCHEDULE:")
        print(f"  Time points: {self.config.time_points}")
        print()
        print("After collecting data, we calibrate to find γ, η, β per exchange.")
        print("Then we can predict impact at ANY time horizon.")
        print("=" * 70)
        print()

        # Start stats thread
        threading.Thread(target=self._stats_loop, daemon=True).start()

        # Start calibration thread
        threading.Thread(target=self._calibration_loop, daemon=True).start()

        # Run C++ pipeline
        self._run_pipeline()

    def _run_pipeline(self):
        """Run C++ blockchain runner."""
        cmd = [
            self.config.cpp_binary,
            "--address-db", self.config.address_db,
            "--utxo-db", self.config.utxo_db,
            "--zmq", self.config.zmq_endpoint
        ]

        print(f"Starting C++ runner...")
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
                signal = self._parse_signal(line)

                if signal:
                    self._handle_signal(signal)
                elif 'Loaded' in line or 'Connected' in line or 'NANOSECOND' in line:
                    print(line)

        except KeyboardInterrupt:
            print("\nShutting down...")
            self.running = False
        except Exception as e:
            print(f"Error: {e}")
            self.running = False

    def _parse_signal(self, line: str) -> Optional[Dict]:
        """Parse C++ signal line."""
        if '\\x1b' in line:
            pattern = r'\[(LONG|SHORT)\]\\x1b\[0m\s+(.+?)\s*\|\s*In:\s*([\d.]+)\s*\|\s*Out:\s*([\d.]+)\s*\|\s*Net:\s*([+-]?[\d.]+)'
        else:
            pattern = r'\[(LONG|SHORT)\]\s+(.+?)\s*\|\s*In:\s*([\d.]+)\s*\|\s*Out:\s*([\d.]+)\s*\|\s*Net:\s*([+-]?[\d.]+)'

        match = re.search(pattern, line)
        if not match:
            return None

        net = float(match.group(5))
        amount = abs(net)

        if amount < self.config.min_flow_btc:
            return None

        return {
            'exchanges': [e.strip() for e in match.group(2).split(',')],
            'direction': 'inflow' if net < 0 else 'outflow',
            'amount': amount
        }

    def _handle_signal(self, signal: Dict):
        """Handle a detected flow signal."""
        self.flows_detected += 1

        for exchange in signal['exchanges']:
            price = self.price_fetcher.get_price(exchange)
            if not price:
                price = self.price_fetcher.get_price()
            if not price:
                continue

            # Record flow and get observation ID
            obs_id = self.model.record_flow(
                exchange=exchange,
                direction=signal['direction'],
                amount_btc=signal['amount'],
                price=price
            )

            # Schedule observations
            self.scheduler.schedule(obs_id, exchange)
            self.observations_scheduled += 1

            # Print
            dir_str = signal['direction'].upper()
            print()
            print(f"[{self.flows_detected}] {dir_str} {exchange} | {signal['amount']:.2f} BTC | ${price:,.2f}")
            print(f"    Scheduled {len(self.config.time_points)} observations...")

            # Show multi-timeframe prediction
            pred = self.model.predict_multi_timeframe(
                exchange=exchange,
                direction=signal['direction'],
                amount_btc=signal['amount'],
                current_price=price
            )

            print("    PREDICTIONS:")
            for label, p in pred['predictions'].items():
                print(f"      t={label:4}: {p['expected_direction']} ${abs(p['predicted_delta']):,.2f}")

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
            print(f"TIME-ADAPTIVE COLLECTOR STATS ({hours:.1f} hours)")
            print("=" * 60)
            print(f"Flows detected: {self.flows_detected}")
            print(f"Observations scheduled: {self.observations_scheduled}")
            print(f"Pending observations: {self.scheduler.get_pending_count()}")
            print()
            print(self.model.get_statistics())
            print("=" * 60)
            print()

    def _calibration_loop(self):
        """Recalibrate models periodically."""
        while self.running:
            time.sleep(self.config.calibration_interval)

            if not self.running:
                break

            print()
            print("[CALIBRATION] Fitting decay curves from collected data...")

            # Calibrate for all exchanges
            result = self.model.calibrate()

            if result.get('status') == 'calibrated':
                print(f"[CALIBRATION] SUCCESS: γ={result['gamma']:.6f}, η={result['eta']:.6f}, β={result['beta']:.2f}")
                print(f"              R² = {result['r_squared']:.3f} from {result['samples']} samples")
            else:
                print(f"[CALIBRATION] {result.get('status', 'unknown')}: {result}")

            print()

    def stop(self):
        """Stop the collector."""
        self.running = False
        self.scheduler.stop()
        self.model.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    import signal as sig

    def handler(s, f):
        print("\nStopping...")
        sys.exit(0)

    sig.signal(sig.SIGINT, handler)
    sig.signal(sig.SIGTERM, handler)

    print("=" * 70)
    print("TIME-ADAPTIVE PRICE IMPACT COLLECTOR")
    print("=" * 70)
    print()
    print("This collector observes price at MULTIPLE time points after each flow:")
    print("  t = [1, 5, 10, 30, 60, 120, 300, 600, 1800] seconds")
    print()
    print("From this data, we fit the decay model:")
    print("  ΔP(t) = γF + ηF × t^(-β)")
    print()
    print("This gives us formulas that work at ANY time horizon.")
    print()
    print("Press Ctrl+C to stop.")
    print()

    collector = TimeAdaptiveCollector()

    try:
        collector.start()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        collector.stop()

        # Print final calibration
        print()
        print("=" * 70)
        print("FINAL CALIBRATION")
        print("=" * 70)

        collector.model = TimeAdaptiveImpactModel()
        result = collector.model.calibrate()

        if result.get('status') == 'calibrated':
            print(f"\nCALIBRATED FORMULA (all exchanges):")
            print(f"  ΔP(t) = γF + ηF × t^(-β)")
            print(f"  γ = {result['gamma']:.6f} (permanent impact)")
            print(f"  η = {result['eta']:.6f} (temporary impact)")
            print(f"  β = {result['beta']:.2f} (decay exponent)")
            print(f"  R² = {result['r_squared']:.3f}")
            print()
            print("EXAMPLE PREDICTIONS (100 BTC inflow at $100,000):")
            for t, coef in zip(result['time_points'], result['mean_coefs']):
                delta = -coef * 100 * 100000
                label = f"{t}s" if t < 60 else f"{t//60}m"
                print(f"  t={label:5}: ΔP = ${delta:+,.2f}")
        else:
            print(f"Status: {result}")


if __name__ == "__main__":
    main()
