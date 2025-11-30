#!/usr/bin/env python3
"""
BLOCKCHAIN → PRICE CALIBRATION
===============================
Track blockchain signals and compare to REAL price movements.
Learn which signals predict price direction.

Goal: Find the math that makes blockchain data predict exchange price.
"""
import time
import json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import deque
import numpy as np


@dataclass
class SignalSnapshot:
    """Snapshot of blockchain signals at a point in time."""
    timestamp: float
    real_price: float

    # Blockchain metrics
    tx_velocity: float = 0.0      # Transactions per second
    fee_fast: float = 0.0         # Fast fee (sat/vB)
    fee_medium: float = 0.0       # Medium fee
    mempool_size: int = 0         # Pending transactions
    mempool_btc: float = 0.0      # BTC value in mempool
    whale_count: int = 0          # Large transactions (>100 BTC)
    block_interval: float = 0.0   # Seconds since last block

    # Formula signals
    formula_signal: float = 0.0   # Aggregated signal (-1 to 1)
    formula_confidence: float = 0.0
    bullish_count: int = 0
    bearish_count: int = 0
    neutral_count: int = 0

    # Price outcome (filled later)
    price_1s: Optional[float] = None   # Price 1 second later
    price_5s: Optional[float] = None   # Price 5 seconds later
    price_10s: Optional[float] = None  # Price 10 seconds later
    price_30s: Optional[float] = None  # Price 30 seconds later


class PriceCalibrator:
    """
    Calibrates blockchain signals to real price movements.

    Tracks:
    1. Blockchain state at time T
    2. Price at T, T+1s, T+5s, T+10s, T+30s
    3. Which signals correctly predicted direction
    """

    def __init__(self, history_size: int = 10000):
        self.snapshots: deque = deque(maxlen=history_size)
        self.pending_snapshots: List[SignalSnapshot] = []

        # Correlation tracking
        self.signal_accuracy: Dict[str, Dict] = {
            'tx_velocity_high': {'correct': 0, 'total': 0},
            'tx_velocity_low': {'correct': 0, 'total': 0},
            'fee_spike': {'correct': 0, 'total': 0},
            'fee_drop': {'correct': 0, 'total': 0},
            'mempool_growing': {'correct': 0, 'total': 0},
            'mempool_shrinking': {'correct': 0, 'total': 0},
            'whale_activity': {'correct': 0, 'total': 0},
            'formula_bullish': {'correct': 0, 'total': 0},
            'formula_bearish': {'correct': 0, 'total': 0},
        }

        # Running averages for baseline
        self.avg_tx_velocity = 5.0
        self.avg_fee = 10.0
        self.avg_mempool = 50000

        # Learned weights (start equal, adjust based on accuracy)
        self.signal_weights = {
            'tx_velocity': 1.0,
            'fee_pressure': 1.0,
            'mempool': 1.0,
            'whale': 1.0,
            'formula': 1.0,
        }

    def record_snapshot(self, snapshot: SignalSnapshot):
        """Record a new snapshot for later analysis."""
        self.pending_snapshots.append(snapshot)

        # Update running averages
        self.avg_tx_velocity = 0.95 * self.avg_tx_velocity + 0.05 * snapshot.tx_velocity
        self.avg_fee = 0.95 * self.avg_fee + 0.05 * snapshot.fee_fast
        self.avg_mempool = 0.95 * self.avg_mempool + 0.05 * snapshot.mempool_size

    def update_outcomes(self, current_price: float, current_time: float):
        """Update pending snapshots with price outcomes."""
        completed = []

        for snap in self.pending_snapshots:
            elapsed = current_time - snap.timestamp

            # Fill in price outcomes as time passes
            if elapsed >= 1.0 and snap.price_1s is None:
                snap.price_1s = current_price
            if elapsed >= 5.0 and snap.price_5s is None:
                snap.price_5s = current_price
            if elapsed >= 10.0 and snap.price_10s is None:
                snap.price_10s = current_price
            if elapsed >= 30.0 and snap.price_30s is None:
                snap.price_30s = current_price
                # Snapshot complete - analyze it
                self._analyze_snapshot(snap)
                completed.append(snap)
                self.snapshots.append(snap)

        # Remove completed snapshots from pending
        for snap in completed:
            self.pending_snapshots.remove(snap)

    def _analyze_snapshot(self, snap: SignalSnapshot):
        """Analyze a completed snapshot to learn signal accuracy."""
        if snap.price_10s is None:
            return

        # Price moved up or down?
        price_change = (snap.price_10s - snap.real_price) / snap.real_price
        went_up = price_change > 0.0001  # >0.01% = up
        went_down = price_change < -0.0001

        # Check each signal type

        # TX Velocity
        if snap.tx_velocity > self.avg_tx_velocity * 1.2:  # 20% above avg
            self.signal_accuracy['tx_velocity_high']['total'] += 1
            if went_up:  # High velocity often = bullish
                self.signal_accuracy['tx_velocity_high']['correct'] += 1
        elif snap.tx_velocity < self.avg_tx_velocity * 0.8:
            self.signal_accuracy['tx_velocity_low']['total'] += 1
            if went_down:
                self.signal_accuracy['tx_velocity_low']['correct'] += 1

        # Fee pressure
        if snap.fee_fast > self.avg_fee * 1.5:  # Fee spike
            self.signal_accuracy['fee_spike']['total'] += 1
            if went_up:  # High fees = demand = bullish
                self.signal_accuracy['fee_spike']['correct'] += 1
        elif snap.fee_fast < self.avg_fee * 0.7:
            self.signal_accuracy['fee_drop']['total'] += 1
            if went_down:
                self.signal_accuracy['fee_drop']['correct'] += 1

        # Whale activity
        if snap.whale_count > 0:
            self.signal_accuracy['whale_activity']['total'] += 1
            # Whale moves often precede volatility - check if we predicted direction
            if (snap.formula_signal > 0 and went_up) or (snap.formula_signal < 0 and went_down):
                self.signal_accuracy['whale_activity']['correct'] += 1

        # Formula signals
        if snap.formula_signal > 0.1:
            self.signal_accuracy['formula_bullish']['total'] += 1
            if went_up:
                self.signal_accuracy['formula_bullish']['correct'] += 1
        elif snap.formula_signal < -0.1:
            self.signal_accuracy['formula_bearish']['total'] += 1
            if went_down:
                self.signal_accuracy['formula_bearish']['correct'] += 1

    def get_calibrated_signal(self, snapshot: SignalSnapshot) -> Tuple[float, float]:
        """
        Get calibrated signal based on learned accuracy weights.

        Returns: (signal -1 to 1, confidence 0 to 1)
        """
        signals = []
        weights = []

        # TX Velocity signal
        if snapshot.tx_velocity > self.avg_tx_velocity * 1.2:
            acc = self._get_accuracy('tx_velocity_high')
            if acc > 0.5:
                signals.append(1.0)
                weights.append(acc * self.signal_weights['tx_velocity'])
        elif snapshot.tx_velocity < self.avg_tx_velocity * 0.8:
            acc = self._get_accuracy('tx_velocity_low')
            if acc > 0.5:
                signals.append(-1.0)
                weights.append(acc * self.signal_weights['tx_velocity'])

        # Fee pressure signal
        if snapshot.fee_fast > self.avg_fee * 1.5:
            acc = self._get_accuracy('fee_spike')
            if acc > 0.5:
                signals.append(1.0)
                weights.append(acc * self.signal_weights['fee_pressure'])
        elif snapshot.fee_fast < self.avg_fee * 0.7:
            acc = self._get_accuracy('fee_drop')
            if acc > 0.5:
                signals.append(-1.0)
                weights.append(acc * self.signal_weights['fee_pressure'])

        # Formula signal (always include)
        if abs(snapshot.formula_signal) > 0.01:
            if snapshot.formula_signal > 0:
                acc = self._get_accuracy('formula_bullish')
            else:
                acc = self._get_accuracy('formula_bearish')
            signals.append(snapshot.formula_signal)
            weights.append(max(acc, 0.5) * self.signal_weights['formula'])

        if not signals:
            return 0.0, 0.0

        # Weighted average
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0, 0.0

        weighted_signal = sum(s * w for s, w in zip(signals, weights)) / total_weight
        confidence = min(total_weight / len(signals), 1.0)

        return float(np.clip(weighted_signal, -1, 1)), float(confidence)

    def _get_accuracy(self, signal_name: str) -> float:
        """Get accuracy rate for a signal type."""
        data = self.signal_accuracy.get(signal_name, {'correct': 0, 'total': 0})
        if data['total'] < 5:  # Need minimum samples
            return 0.5  # Assume 50% until we have data
        return data['correct'] / data['total']

    def get_stats(self) -> Dict:
        """Get calibration statistics."""
        stats = {
            'total_snapshots': len(self.snapshots),
            'pending_snapshots': len(self.pending_snapshots),
            'signal_accuracy': {},
            'avg_tx_velocity': self.avg_tx_velocity,
            'avg_fee': self.avg_fee,
            'avg_mempool': self.avg_mempool,
        }

        for name, data in self.signal_accuracy.items():
            if data['total'] > 0:
                stats['signal_accuracy'][name] = {
                    'accuracy': data['correct'] / data['total'] * 100,
                    'samples': data['total']
                }

        return stats

    def print_calibration_report(self):
        """Print current calibration status."""
        print("\n" + "=" * 60)
        print("BLOCKCHAIN → PRICE CALIBRATION REPORT")
        print("=" * 60)
        print(f"Total snapshots analyzed: {len(self.snapshots)}")
        print(f"Pending snapshots: {len(self.pending_snapshots)}")
        print()
        print("SIGNAL ACCURACY (predicting 10s price movement):")
        print("-" * 60)

        for name, data in self.signal_accuracy.items():
            if data['total'] >= 5:
                acc = data['correct'] / data['total'] * 100
                edge = acc - 50  # Edge over random
                print(f"  {name:20} {acc:5.1f}% ({data['total']:4} samples) Edge: {edge:+.1f}%")
            elif data['total'] > 0:
                print(f"  {name:20} (need more samples: {data['total']}/5)")

        print()
        print("BASELINE AVERAGES:")
        print(f"  TX Velocity: {self.avg_tx_velocity:.2f}/sec")
        print(f"  Fee (fast):  {self.avg_fee:.1f} sat/vB")
        print(f"  Mempool:     {self.avg_mempool:,.0f} txs")
        print("=" * 60)
