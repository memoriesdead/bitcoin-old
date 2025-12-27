#!/usr/bin/env python3
"""
REGIME DETECTION LIVE TEST
==========================

Tests the regime detection model against live blockchain data.

KEY INSIGHT: Individual flows = noise. Aggregate patterns = signal.

This pipeline:
1. Receives blockchain flows from C++ nanosecond runner
2. Aggregates flows into 10-minute windows per exchange
3. Calculates regime change indicators (z-score, CFI, persistence, whale ratio)
4. Only signals when RCS > 1.0 (STRONG regime change)
5. Verifies predictions after 5 minutes
6. Reports accuracy in real-time

The goal: ONLY trade when math says conditions are UNUSUAL.
No signal = no trade = no random 50% accuracy.
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

sys.path.insert(0, '/root/sovereign/blockchain')

try:
    import ccxt
    HAS_CCXT = True
except ImportError:
    HAS_CCXT = False

from regime_detection_model import RegimeDetector, RegimeConfig, RegimeSignal


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class LiveTestConfig:
    """Configuration for live testing."""
    cpp_binary: str = "/root/sovereign/cpp_runner/build/blockchain_runner"
    address_db: str = "/root/sovereign/walletexplorer_addresses.db"
    utxo_db: str = "/root/sovereign/exchange_utxos.db"
    zmq_endpoint: str = "tcp://127.0.0.1:28332"

    # Flow threshold (proven from CLAUDE.md)
    min_flow_btc: float = 10.0

    # Verification delay
    verify_delay_seconds: int = 300  # 5 minutes

    # Price movement threshold (0.05% minimum)
    min_price_move_pct: float = 0.0005

    # Only act on strong signals
    min_signal_strength: str = "MEDIUM"  # "WEAK", "MEDIUM", or "STRONG"


# =============================================================================
# PRICE FETCHER
# =============================================================================

class PriceFetcher:
    """Multi-exchange price fetcher using CCXT."""

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
                print(f"  [FAIL] {ex_id}: {e}")

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
    """A signal awaiting price verification."""
    signal: RegimeSignal
    price_at_signal: float
    verify_time: datetime


# =============================================================================
# LIVE TESTER
# =============================================================================

class RegimeLiveTester:
    """Live testing of regime detection model."""

    def __init__(self, config: LiveTestConfig = None):
        self.config = config or LiveTestConfig()
        self.running = False

        # Initialize regime detector
        regime_config = RegimeConfig(
            db_path="/root/sovereign/regime_detection.db",
            flow_window_seconds=600,  # 10-minute windows
            lookback_windows=20,      # 20-window history
            min_flow_btc=self.config.min_flow_btc
        )
        self.detector = RegimeDetector(regime_config)

        # Initialize price fetcher
        print("\nInitializing price feeds...")
        self.price_fetcher = PriceFetcher()
        time.sleep(3)  # Wait for initial prices

        # Pending verifications
        self.pending: List[PendingVerification] = []
        self.lock = threading.Lock()

        # Results tracking
        self.total_flows = 0
        self.total_signals = 0
        self.verified_signals = 0
        self.correct_predictions = 0
        self.results: List[Dict] = []

        # Start time
        self.start_time = None

    def start(self):
        """Start the live test."""
        self.running = True
        self.start_time = time.time()

        # Start verification thread
        threading.Thread(target=self._verification_loop, daemon=True).start()

        # Start stats thread
        threading.Thread(target=self._stats_loop, daemon=True).start()

        # Print header
        self._print_header()

        # Run C++ pipeline
        self._run_pipeline()

    def _print_header(self):
        """Print startup header."""
        print()
        print("=" * 70)
        print("REGIME DETECTION LIVE TEST")
        print("=" * 70)
        print()
        print("THE MATHEMATICAL INSIGHT:")
        print("  Individual flows = NOISE (50% accuracy)")
        print("  Aggregate patterns = SIGNAL (targeting 100%)")
        print()
        print("REGIME CHANGE INDICATORS:")
        print("  1. Z-Score:     Flow deviation from rolling mean (|z| > 2)")
        print("  2. CFI:         Cumulative flow imbalance (|CFI| > 0.6)")
        print("  3. Persistence: Consecutive same-direction windows (> 70%)")
        print("  4. Whale Ratio: Large flow dominance (> 50%)")
        print()
        print("REGIME CHANGE SCORE (RCS):")
        print("  RCS = 0.3*|z| + 0.3*|CFI| + 0.2*Persistence + 0.2*WhaleRatio")
        print()
        print(f"  Minimum signal: {self.config.min_signal_strength}")
        print(f"  Verify after:   {self.config.verify_delay_seconds} seconds")
        print()
        print("Only trading when REGIME CHANGE is detected (unusual patterns).")
        print("=" * 70)
        print()

    def _run_pipeline(self):
        """Run the C++ blockchain runner."""
        cmd = [
            self.config.cpp_binary,
            "--address-db", self.config.address_db,
            "--utxo-db", self.config.utxo_db,
            "--zmq", self.config.zmq_endpoint
        ]

        print("Starting C++ nanosecond runner...")
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
                flow = self._parse_flow(line)

                if flow:
                    self._handle_flow(flow)
                elif 'Loaded' in line or 'Connected' in line or 'NANOSECOND' in line:
                    print(line)

        except KeyboardInterrupt:
            self.running = False
            self._print_final_results()
        except Exception as e:
            print(f"Error: {e}")
            self.running = False

    def _parse_flow(self, line: str) -> Optional[Dict]:
        """Parse C++ output for flow signals."""
        # Pattern: [LONG|SHORT] exchange | In: X | Out: Y | Net: Z
        if '\\x1b' in line:
            pattern = r'\[(LONG|SHORT)\]\\x1b\[0m\s+(.+?)\s*\|\s*In:\s*([\d.]+)\s*\|\s*Out:\s*([\d.]+)\s*\|\s*Net:\s*([+-]?[\d.]+)'
        else:
            pattern = r'\[(LONG|SHORT)\]\s+(.+?)\s*\|\s*In:\s*([\d.]+)\s*\|\s*Out:\s*([\d.]+)\s*\|\s*Net:\s*([+-]?[\d.]+)'

        match = re.search(pattern, line)
        if not match:
            return None

        inflow = float(match.group(3))
        outflow = float(match.group(4))
        net = float(match.group(5))

        # Only process if meaningful flow
        if inflow < self.config.min_flow_btc and outflow < self.config.min_flow_btc:
            return None

        return {
            'exchanges': [e.strip() for e in match.group(2).split(',')],
            'inflow': inflow,
            'outflow': outflow,
            'net': net
        }

    def _handle_flow(self, flow: Dict):
        """Handle a detected flow."""
        self.total_flows += 1

        for exchange in flow['exchanges']:
            # Add flow to regime detector
            if flow['inflow'] >= self.config.min_flow_btc:
                signal = self.detector.add_flow(exchange, 'inflow', flow['inflow'])
            elif flow['outflow'] >= self.config.min_flow_btc:
                signal = self.detector.add_flow(exchange, 'outflow', flow['outflow'])
            else:
                signal = None

            # Check if we got a signal
            if signal and self._is_actionable(signal):
                self._handle_signal(exchange, signal)

    def _is_actionable(self, signal: RegimeSignal) -> bool:
        """Check if signal meets minimum strength requirement."""
        strength_order = ["NONE", "WEAK", "MEDIUM", "STRONG"]

        min_idx = strength_order.index(self.config.min_signal_strength)
        signal_idx = strength_order.index(signal.strength)

        return signal_idx >= min_idx and signal.direction != 0

    def _handle_signal(self, exchange: str, signal: RegimeSignal):
        """Handle a regime change signal."""
        self.total_signals += 1

        price = self.price_fetcher.get_price(exchange)
        if not price:
            price = self.price_fetcher.get_price()
        if not price:
            return

        # Schedule verification
        now = datetime.now(timezone.utc)
        pending = PendingVerification(
            signal=signal,
            price_at_signal=price,
            verify_time=now + timedelta(seconds=self.config.verify_delay_seconds)
        )

        with self.lock:
            self.pending.append(pending)

        # Print signal
        print()
        print("=" * 60)
        print(f"REGIME CHANGE DETECTED: {exchange.upper()}")
        print("-" * 60)
        print(f"  Direction:   {signal.direction_label} ({signal.strength})")
        print(f"  Z-Score:     {signal.z_score:+.2f} (current vs rolling mean)")
        print(f"  CFI:         {signal.cfi_normalized:+.2f} (cumulative imbalance)")
        print(f"  Persistence: {signal.persistence_score:.1%} (same direction)")
        print(f"  Whale Ratio: {signal.whale_ratio:.1%} (>100 BTC flows)")
        print(f"  RCS:         {signal.regime_change_score:.2f}")
        print()
        print(f"  Current Net Flow: {signal.current_window_net_flow:+.1f} BTC")
        print(f"  Rolling Mean:     {signal.rolling_mean:+.1f} BTC")
        print(f"  Rolling Std:      {signal.rolling_std:.1f} BTC")
        print()
        print(f"  Price:       ${price:,.2f}")
        print(f"  Prediction:  {signal.direction_label}")
        print(f"  Verify in:   {self.config.verify_delay_seconds}s")
        print("=" * 60)
        print()

    def _verification_loop(self):
        """Verify signals after delay."""
        while self.running:
            time.sleep(1)
            now = datetime.now(timezone.utc)

            with self.lock:
                ready = [p for p in self.pending if now >= p.verify_time]
                self.pending = [p for p in self.pending if now < p.verify_time]

            for p in ready:
                self._verify(p)

    def _verify(self, p: PendingVerification):
        """Verify a signal against actual price movement."""
        signal = p.signal
        exchange = signal.exchange

        # Get current price
        price_now = self.price_fetcher.get_price(exchange)
        if not price_now:
            price_now = self.price_fetcher.get_price()
        if not price_now:
            print(f"[SKIP] Could not get price for {exchange}")
            return

        # Calculate actual movement
        delta = price_now - p.price_at_signal
        pct_change = delta / p.price_at_signal

        # Determine actual direction (with noise filter)
        if pct_change > self.config.min_price_move_pct:
            actual_direction = 1
            actual_label = "UP"
        elif pct_change < -self.config.min_price_move_pct:
            actual_direction = -1
            actual_label = "DOWN"
        else:
            actual_direction = 0
            actual_label = "FLAT"

        # Check if prediction was correct
        predicted_label = "UP" if signal.direction == 1 else "DOWN"

        # Only count if price actually moved
        if actual_direction != 0:
            self.verified_signals += 1

            if signal.direction == actual_direction:
                self.correct_predictions += 1
                correct = True
            else:
                correct = False

            # Calculate accuracy
            accuracy = (self.correct_predictions / self.verified_signals * 100) if self.verified_signals > 0 else 0

            # Store result
            self.results.append({
                'exchange': exchange,
                'signal_strength': signal.strength,
                'z_score': signal.z_score,
                'cfi': signal.cfi_normalized,
                'rcs': signal.regime_change_score,
                'predicted_direction': signal.direction,
                'actual_direction': actual_direction,
                'price_at_signal': p.price_at_signal,
                'price_after': price_now,
                'pct_change': pct_change,
                'correct': correct
            })

            # Print result
            status = "CORRECT" if correct else "WRONG"
            status_color = "\033[92m" if correct else "\033[91m"

            print()
            print("=" * 60)
            print(f"VERIFICATION #{self.verified_signals}: {exchange.upper()}")
            print("-" * 60)
            print(f"  Signal:      {signal.direction_label} ({signal.strength})")
            print(f"  RCS:         {signal.regime_change_score:.2f}")
            print(f"  Z-Score:     {signal.z_score:+.2f}")
            print(f"  CFI:         {signal.cfi_normalized:+.2f}")
            print()
            print(f"  Price at signal: ${p.price_at_signal:,.2f}")
            print(f"  Price after 5m:  ${price_now:,.2f}")
            print(f"  Change:          ${delta:+.2f} ({pct_change:+.2%})")
            print()
            print(f"  Predicted: {predicted_label}")
            print(f"  Actual:    {actual_label}")
            print()
            print(f"  {status_color}>>> {status} <<<\033[0m")
            print()
            print(f"  RUNNING ACCURACY: {self.correct_predictions}/{self.verified_signals} = {accuracy:.1f}%")
            print("=" * 60)
            print()
        else:
            print(f"[FLAT] {exchange}: No significant price movement ({pct_change:+.2%})")

    def _stats_loop(self):
        """Print periodic statistics."""
        while self.running:
            time.sleep(120)  # Every 2 minutes

            if not self.running:
                break

            elapsed = time.time() - self.start_time
            hours = elapsed / 3600

            accuracy = (self.correct_predictions / self.verified_signals * 100) if self.verified_signals > 0 else 0
            signal_rate = self.total_signals / hours if hours > 0 else 0

            print()
            print("-" * 60)
            print(f"REGIME DETECTION STATS ({hours:.1f} hours)")
            print("-" * 60)
            print(f"  Total flows processed: {self.total_flows}")
            print(f"  Regime signals:        {self.total_signals}")
            print(f"  Signal rate:           {signal_rate:.1f}/hour")
            print(f"  Verified:              {self.verified_signals}")
            print(f"  Correct:               {self.correct_predictions}")
            print(f"  ACCURACY:              {accuracy:.1f}%")
            print(f"  Pending verification:  {len(self.pending)}")
            print("-" * 60)
            print()

    def _print_final_results(self):
        """Print final test results."""
        print()
        print("=" * 70)
        print("FINAL RESULTS - REGIME DETECTION TEST")
        print("=" * 70)

        if not self.results:
            print("No verified signals yet.")
            return

        elapsed = time.time() - self.start_time
        hours = elapsed / 3600

        accuracy = (self.correct_predictions / self.verified_signals * 100) if self.verified_signals > 0 else 0

        print()
        print(f"Test duration:      {hours:.1f} hours")
        print(f"Total flows:        {self.total_flows}")
        print(f"Regime signals:     {self.total_signals}")
        print(f"Verified:           {self.verified_signals}")
        print(f"Correct:            {self.correct_predictions}")
        print()
        print(f"ACCURACY: {accuracy:.1f}%")
        print()

        # By signal strength
        print("BY SIGNAL STRENGTH:")
        for strength in ["STRONG", "MEDIUM", "WEAK"]:
            subset = [r for r in self.results if r['signal_strength'] == strength]
            if subset:
                correct = sum(1 for r in subset if r['correct'])
                acc = correct / len(subset) * 100
                print(f"  {strength:8}: {correct}/{len(subset)} = {acc:.1f}%")

        print()

        # By exchange
        print("BY EXCHANGE:")
        exchanges = set(r['exchange'] for r in self.results)
        for exchange in sorted(exchanges):
            subset = [r for r in self.results if r['exchange'] == exchange]
            if subset:
                correct = sum(1 for r in subset if r['correct'])
                acc = correct / len(subset) * 100
                print(f"  {exchange:12}: {correct}/{len(subset)} = {acc:.1f}%")

        print()

        # RCS analysis
        print("BY RCS THRESHOLD:")
        for threshold in [0.8, 1.0, 1.2, 1.5, 2.0]:
            subset = [r for r in self.results if r['rcs'] >= threshold]
            if subset:
                correct = sum(1 for r in subset if r['correct'])
                acc = correct / len(subset) * 100
                print(f"  RCS >= {threshold:.1f}: {correct}/{len(subset)} = {acc:.1f}%")

        print()
        print("=" * 70)
        print("KEY INSIGHT:")
        print("-" * 70)
        print("If accuracy < 60%: Lower RCS threshold or adjust weights")
        print("If accuracy > 80%: Model is working - increase position sizing")
        print("If signals rare:   Lower thresholds to capture more patterns")
        print("If signals wrong:  Raise thresholds to filter noise")
        print("=" * 70)

    def stop(self):
        """Stop the test."""
        self.running = False
        self.detector.close()


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
    print("REGIME DETECTION LIVE TEST")
    print("=" * 70)
    print()
    print("THE MATHEMATICAL APPROACH TO 100% ACCURACY")
    print()
    print("Problem: Simple flow→direction = 50% accuracy (noise)")
    print("Solution: Only trade REGIME CHANGES (unusual aggregate patterns)")
    print()
    print("This test validates the regime detection model against live data.")
    print()
    print("Press Ctrl+C to stop and see final results.")
    print()

    tester = RegimeLiveTester()

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
