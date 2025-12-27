#!/usr/bin/env python3
"""
QUANT LIVE TEST
===============

Integrated test that:
1. Detects blockchain flows in real-time
2. Makes predictions using gold-standard quant formulas
3. Checks actual price after 60 seconds
4. Reports accuracy immediately

Run for 10-30 minutes to see if predictions are accurate.
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
from collections import deque

sys.path.insert(0, '/root/sovereign')
sys.path.insert(0, '/root/sovereign/blockchain')

try:
    import ccxt
    HAS_CCXT = True
except ImportError:
    HAS_CCXT = False

from quant_price_impact import (
    DeterministicPricePredictor,
    PriceImpactConfig
)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TestConfig:
    cpp_binary: str = "/root/sovereign/cpp_runner/build/blockchain_runner"
    address_db: str = "/root/sovereign/walletexplorer_addresses.db"
    utxo_db: str = "/root/sovereign/exchange_utxos.db"
    zmq_endpoint: str = "tcp://127.0.0.1:28332"
    min_flow_btc: float = 10.0  # Proven threshold from CLAUDE.md
    verify_delay_seconds: int = 300  # 5 minutes - proven window
    min_price_move_pct: float = 0.0005  # 0.05% noise filter


# =============================================================================
# PRICE FETCHER
# =============================================================================

class PriceFetcher:
    def __init__(self):
        self.exchanges = {}
        self.prices = {}
        self.lock = threading.Lock()
        self._init()

    def _init(self):
        if not HAS_CCXT:
            return
        for ex_id in ['binance', 'coinbase', 'kraken', 'bitstamp', 'bitfinex', 'okx', 'bybit']:
            try:
                self.exchanges[ex_id] = getattr(ccxt, ex_id)({'enableRateLimit': True})
            except:
                pass

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
# PENDING VERIFICATION
# =============================================================================

@dataclass
class PendingVerification:
    id: int
    exchange: str
    direction: str
    amount_btc: float
    price_at_signal: float
    predicted_delta: float
    predicted_direction: str
    signal_time: datetime
    verify_time: datetime


# =============================================================================
# LIVE TESTER
# =============================================================================

class QuantLiveTester:
    def __init__(self, config: TestConfig = None):
        self.config = config or TestConfig()
        self.running = False

        # Initialize predictor
        impact_config = PriceImpactConfig(
            db_path="/tmp/quant_test.db",
            min_flow_btc=self.config.min_flow_btc
        )
        self.predictor = DeterministicPricePredictor(impact_config)

        # Price fetcher
        print("Initializing price feeds...")
        self.price_fetcher = PriceFetcher()
        time.sleep(3)

        # Pending verifications
        self.pending: List[PendingVerification] = []
        self.lock = threading.Lock()

        # Results
        self.total_predictions = 0
        self.correct_direction = 0
        self.total_error = 0.0
        self.results: List[Dict] = []

    def start(self):
        self.running = True

        # Start verification thread
        threading.Thread(target=self._verification_loop, daemon=True).start()

        # Print header
        print()
        print("=" * 70)
        print("QUANT LIVE TEST - PREDICTION vs REALITY")
        print("=" * 70)
        print(f"Verify delay: {self.config.verify_delay_seconds} seconds")
        print(f"Min flow: {self.config.min_flow_btc} BTC")
        print()
        print("Waiting for blockchain signals...")
        print()

        # Run C++ pipeline
        self._run_pipeline()

    def _run_pipeline(self):
        cmd = [
            self.config.cpp_binary,
            "--address-db", self.config.address_db,
            "--utxo-db", self.config.utxo_db,
            "--zmq", self.config.zmq_endpoint
        ]

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
            self.running = False
            self._print_final_results()
        except Exception as e:
            print(f"Error: {e}")
            self.running = False

    def _parse_signal(self, line: str) -> Optional[Dict]:
        if '\x1b' in line:
            pattern = r'\[(LONG|SHORT)\]\x1b\[0m\s+(.+?)\s*\|\s*In:\s*([\d.]+)\s*\|\s*Out:\s*([\d.]+)\s*\|\s*Net:\s*([+-]?[\d.]+)'
        else:
            pattern = r'\[(LONG|SHORT)\]\s+(.+?)\s*\|\s*In:\s*([\d.]+)\s*\|\s*Out:\s*([\d.]+)\s*\|\s*Net:\s*([+-]?[\d.]+)'

        match = re.search(pattern, line)
        if not match:
            return None

        net = float(match.group(5))
        return {
            'exchanges': [e.strip() for e in match.group(2).split(',')],
            'direction': 'inflow' if net < 0 else 'outflow',
            'amount': abs(net)
        }

    def _handle_signal(self, signal: Dict):
        for exchange in signal['exchanges']:
            price = self.price_fetcher.get_price(exchange)
            if not price:
                price = self.price_fetcher.get_price()
            if not price:
                continue

            # Update predictor
            self.predictor.update_price(exchange, price)

            # Get prediction
            result = self.predictor.on_flow(
                exchange=exchange,
                direction=signal['direction'],
                amount_btc=signal['amount'],
                current_price=price
            )

            if result.get('status') == 'skipped':
                continue

            self.total_predictions += 1

            # Schedule verification
            now = datetime.now(timezone.utc)
            pending = PendingVerification(
                id=result['id'],
                exchange=exchange,
                direction=signal['direction'],
                amount_btc=signal['amount'],
                price_at_signal=price,
                predicted_delta=result['predictions']['combined'],
                predicted_direction=result['expected_direction'],
                signal_time=now,
                verify_time=now + timedelta(seconds=self.config.verify_delay_seconds)
            )

            with self.lock:
                self.pending.append(pending)

            # Print prediction
            dir_str = signal['direction'].upper()
            pred_dir = result['expected_direction']
            pred_amt = abs(result['predictions']['combined'])

            print(f"[{self.total_predictions}] {dir_str} {exchange} | {signal['amount']:.2f} BTC")
            print(f"    Price: ${price:,.2f} | Predict: {pred_dir} ${pred_amt:.2f}")
            print(f"    Verifying in {self.config.verify_delay_seconds}s...")
            print()

    def _verification_loop(self):
        while self.running:
            time.sleep(1)
            now = datetime.now(timezone.utc)

            with self.lock:
                ready = [p for p in self.pending if now >= p.verify_time]
                self.pending = [p for p in self.pending if now < p.verify_time]

            for p in ready:
                self._verify(p)

    def _verify(self, p: PendingVerification):
        # Get current price
        price_now = self.price_fetcher.get_price(p.exchange)
        if not price_now:
            price_now = self.price_fetcher.get_price()
        if not price_now:
            print(f"[SKIP] Could not get price for verification")
            return

        # Calculate actual delta
        actual_delta = price_now - p.price_at_signal
        actual_direction = "UP" if actual_delta > 0 else "DOWN"

        # Check if prediction was correct
        direction_correct = (p.predicted_direction == actual_direction)
        prediction_error = abs(p.predicted_delta - actual_delta)

        if direction_correct:
            self.correct_direction += 1

        self.total_error += prediction_error

        # Calculate accuracy so far
        accuracy = (self.correct_direction / len(self.results) * 100) if self.results else 0

        # Store result
        self.results.append({
            'exchange': p.exchange,
            'direction': p.direction,
            'amount': p.amount_btc,
            'price_at_signal': p.price_at_signal,
            'price_after': price_now,
            'predicted_delta': p.predicted_delta,
            'actual_delta': actual_delta,
            'predicted_direction': p.predicted_direction,
            'actual_direction': actual_direction,
            'direction_correct': direction_correct,
            'error': prediction_error
        })

        # Calculate running stats
        n = len(self.results)
        accuracy = self.correct_direction / n * 100
        avg_error = self.total_error / n

        # Print result
        status = "CORRECT" if direction_correct else "WRONG"
        status_color = "\033[92m" if direction_correct else "\033[91m"

        print()
        print("=" * 60)
        print(f"VERIFICATION #{n}: {p.exchange.upper()} {p.direction.upper()}")
        print("-" * 60)
        print(f"  Flow: {p.amount_btc:.2f} BTC")
        print(f"  Price at signal: ${p.price_at_signal:,.2f}")
        print(f"  Price after 60s: ${price_now:,.2f}")
        print(f"  Actual move: ${actual_delta:+.2f} ({actual_direction})")
        print(f"  Predicted:   ${p.predicted_delta:+.2f} ({p.predicted_direction})")
        print(f"  Error: ${prediction_error:.2f}")
        print()
        print(f"  {status_color}>>> {status} <<<\033[0m")
        print()
        print(f"  RUNNING STATS: {self.correct_direction}/{n} = {accuracy:.1f}% accuracy")
        print(f"  Avg error: ${avg_error:.2f}")
        print("=" * 60)
        print()

    def _print_final_results(self):
        print()
        print("=" * 70)
        print("FINAL RESULTS")
        print("=" * 70)

        if not self.results:
            print("No verified predictions yet.")
            return

        n = len(self.results)
        accuracy = self.correct_direction / n * 100
        avg_error = self.total_error / n

        print(f"Total predictions: {self.total_predictions}")
        print(f"Verified: {n}")
        print(f"Correct direction: {self.correct_direction}/{n} = {accuracy:.1f}%")
        print(f"Average error: ${avg_error:.2f}")
        print()

        # By exchange
        print("BY EXCHANGE:")
        by_exchange = {}
        for r in self.results:
            ex = r['exchange']
            if ex not in by_exchange:
                by_exchange[ex] = {'correct': 0, 'total': 0, 'error': 0}
            by_exchange[ex]['total'] += 1
            by_exchange[ex]['error'] += r['error']
            if r['direction_correct']:
                by_exchange[ex]['correct'] += 1

        for ex, stats in by_exchange.items():
            acc = stats['correct'] / stats['total'] * 100
            avg_err = stats['error'] / stats['total']
            print(f"  {ex:12} {stats['correct']}/{stats['total']} = {acc:.1f}% | Avg err: ${avg_err:.2f}")

        print()
        print("=" * 70)

    def stop(self):
        self.running = False
        self.predictor.close()


def main():
    import signal as sig

    def handler(s, f):
        print("\nStopping...")
        sys.exit(0)

    sig.signal(sig.SIGINT, handler)
    sig.signal(sig.SIGTERM, handler)

    print("=" * 70)
    print("QUANT FORMULA LIVE TEST")
    print("=" * 70)
    print()
    print("This test will:")
    print("  1. Detect blockchain flows")
    print("  2. Make price predictions using quant formulas")
    print("  3. Verify predictions after 60 seconds")
    print("  4. Report accuracy in real-time")
    print()
    print("Press Ctrl+C to stop and see final results.")
    print()

    tester = QuantLiveTester()

    try:
        tester.start()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        tester.stop()


if __name__ == "__main__":
    main()
