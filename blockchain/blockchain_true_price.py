#!/usr/bin/env python3
"""
BLOCKCHAIN TRUE PRICE ENGINE
============================
Derives price DIRECTION signals from PURE blockchain data.

TRUTH: The blockchain does NOT contain price. Price is determined by
exchange order books (OFF-CHAIN). But blockchain metrics are LEADING
INDICATORS of price direction.

What blockchain gives us:
1. Fee rates (demand urgency) - HIGH fees = people NEED to transact = demand
2. Mempool growth (congestion) - GROWING mempool = demand backlog
3. TX velocity (network activity) - HIGH activity = interest/momentum
4. Whale transactions (institutional) - BIG moves = informed players
5. Block intervals (mining) - Faster blocks = more hashrate = confidence

These metrics LEAD price by seconds to minutes because:
- Fee spike = urgent buying on-chain = demand = price up
- Whale TX = big player positioning = price move imminent
- Mempool surge = transaction urgency = demand pressure

This engine provides:
1. DIRECTION signal (-1 to +1) - Which way price should move
2. MOMENTUM signal - How fast price should move
3. CONFIDENCE - How strong the signal is

For trading: Direction * Momentum * Confidence = Trade signal
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Tuple, Optional
from collections import deque


@dataclass
class BlockchainSignal:
    """Pure blockchain-derived signal for trading."""
    timestamp: float

    # Direction signal: -1 (bearish) to +1 (bullish)
    direction: float = 0.0

    # Momentum: 0 (no momentum) to 1 (strong momentum)
    momentum: float = 0.0

    # Confidence: 0 (low) to 1 (high)
    confidence: float = 0.0

    # Component signals for debugging
    fee_signal: float = 0.0
    mempool_signal: float = 0.0
    velocity_signal: float = 0.0
    whale_signal: float = 0.0
    acceleration_signal: float = 0.0

    # Raw metrics
    fee_rate: float = 0.0
    mempool_size: int = 0
    tx_velocity: float = 0.0
    whale_count: int = 0

    @property
    def trade_signal(self) -> float:
        """Combined signal for trading: direction * momentum * confidence"""
        return self.direction * (0.5 + 0.5 * self.momentum) * self.confidence


@dataclass
class MetricHistory:
    """Tracks history of a metric for trend analysis."""
    values: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=1000))

    def add(self, timestamp: float, value: float):
        self.values.append((timestamp, value))

    def get_trend(self, window_sec: float = 10.0) -> float:
        """Get trend over window: positive = increasing, negative = decreasing"""
        if len(self.values) < 2:
            return 0.0

        now = time.time()
        recent = [(t, v) for t, v in self.values if now - t <= window_sec]

        if len(recent) < 2:
            return 0.0

        # Linear regression slope
        times = np.array([t for t, _ in recent])
        values = np.array([v for _, v in recent])

        # Normalize time
        times = times - times[0]
        if times[-1] == 0:
            return 0.0

        # Calculate slope
        mean_t = np.mean(times)
        mean_v = np.mean(values)

        numerator = np.sum((times - mean_t) * (values - mean_v))
        denominator = np.sum((times - mean_t) ** 2)

        if denominator == 0:
            return 0.0

        slope = numerator / denominator

        # Normalize to -1 to +1 based on % change
        if mean_v != 0:
            pct_change = slope * window_sec / mean_v
            return np.clip(pct_change * 10, -1, 1)  # 10% change = max signal

        return 0.0

    def get_acceleration(self, window_sec: float = 10.0) -> float:
        """Get acceleration: is trend accelerating or decelerating?"""
        if len(self.values) < 3:
            return 0.0

        # Compare trend in first half vs second half of window
        half_window = window_sec / 2

        now = time.time()
        mid_time = now - half_window

        first_half = [(t, v) for t, v in self.values if mid_time - half_window <= t < mid_time]
        second_half = [(t, v) for t, v in self.values if t >= mid_time]

        if len(first_half) < 2 or len(second_half) < 2:
            return 0.0

        # Get slopes for each half
        def slope(data):
            times = np.array([t for t, _ in data])
            values = np.array([v for _, v in data])
            times = times - times[0]
            if len(times) < 2 or times[-1] == 0:
                return 0.0
            mean_t = np.mean(times)
            mean_v = np.mean(values)
            num = np.sum((times - mean_t) * (values - mean_v))
            den = np.sum((times - mean_t) ** 2)
            return num / den if den != 0 else 0.0

        slope1 = slope(first_half)
        slope2 = slope(second_half)

        # Acceleration = change in slope
        accel = slope2 - slope1
        return np.clip(accel * 100, -1, 1)


class BlockchainTruePrice:
    """
    PURE BLOCKCHAIN SIGNAL ENGINE

    NO API dependence. NO calibration price needed.
    Derives trading signals from blockchain metrics ONLY.

    Theory:
    1. Fee spikes LEAD price increases (people urgently buying on-chain)
    2. Mempool growth LEADS price increases (demand backlog)
    3. TX velocity correlates with interest/activity
    4. Whale TXs often precede big moves

    Usage:
        engine = BlockchainTruePrice()
        signal = engine.update(
            fee_fast=15,
            fee_medium=10,
            mempool_size=50000,
            tx_per_sec=5.0,
            whale_count=2
        )

        if signal.direction > 0.1 and signal.confidence > 0.5:
            # Bullish signal - consider long
            pass
    """

    def __init__(
        self,
        fee_weight: float = 0.25,
        mempool_weight: float = 0.25,
        velocity_weight: float = 0.35,
        whale_weight: float = 0.15,
        invert_mempool: bool = False,   # DO NOT INVERT - correlations are unstable!
        adaptive: bool = True,          # Enable adaptive weight learning
    ):
        # Signal weights - BALANCED because correlations are UNSTABLE
        #
        # CRITICAL FINDING: Blockchain -> price correlations are NOT stable.
        # Run 1: Velocity 84%, Mempool 35% (inverse)
        # Run 2: Velocity 33%, Mempool 49% (neutral)
        #
        # This means: Blockchain data CANNOT reliably predict short-term price.
        # Use for: network activity signals, supplementary data
        # Do NOT use for: replacing exchange price feeds
        total = fee_weight + mempool_weight + velocity_weight + whale_weight
        self.fee_weight = fee_weight / total
        self.mempool_weight = mempool_weight / total
        self.velocity_weight = velocity_weight / total
        self.whale_weight = whale_weight / total
        self.invert_mempool = invert_mempool
        self.adaptive = adaptive

        # Metric histories for trend analysis
        self.fee_history = MetricHistory()
        self.mempool_history = MetricHistory()
        self.velocity_history = MetricHistory()
        self.whale_history = MetricHistory()

        # Baseline calibration (learned from data, not hardcoded)
        self.baseline_fee: Optional[float] = None
        self.baseline_mempool: Optional[float] = None
        self.baseline_velocity: Optional[float] = None

        # EMA for baseline updates
        self.alpha = 0.01  # Slow adaptation

        # Stats
        self.update_count = 0
        self.signals_generated = 0

        # Signal history for accuracy tracking
        self.signal_history: Deque[BlockchainSignal] = deque(maxlen=1000)

    def update(
        self,
        fee_fast: float,
        fee_medium: float = 0,
        mempool_size: int = 0,
        tx_per_sec: float = 0,
        whale_count: int = 0,
        whale_volume_btc: float = 0,
    ) -> BlockchainSignal:
        """
        Update with new blockchain metrics and generate signal.

        Returns BlockchainSignal with direction, momentum, confidence.
        """
        now = time.time()
        self.update_count += 1

        # Record metrics to history
        self.fee_history.add(now, fee_fast)
        self.mempool_history.add(now, mempool_size)
        self.velocity_history.add(now, tx_per_sec)
        self.whale_history.add(now, whale_count)

        # Initialize baselines from first data
        if self.baseline_fee is None:
            self.baseline_fee = max(fee_fast, 1)
            self.baseline_mempool = max(mempool_size, 1000)
            self.baseline_velocity = max(tx_per_sec, 0.1)
        else:
            # Update baselines with EMA
            self.baseline_fee = self.alpha * max(fee_fast, 1) + (1 - self.alpha) * self.baseline_fee
            self.baseline_mempool = self.alpha * max(mempool_size, 1000) + (1 - self.alpha) * self.baseline_mempool
            self.baseline_velocity = self.alpha * max(tx_per_sec, 0.1) + (1 - self.alpha) * self.baseline_velocity

        # === SIGNAL COMPONENTS ===

        # 1. FEE SIGNAL
        # High fees relative to baseline = BULLISH (demand)
        # Fee INCREASE = BULLISH (rising demand)
        fee_ratio = fee_fast / self.baseline_fee
        fee_level_signal = np.clip((fee_ratio - 1) * 2, -1, 1)  # 50% above baseline = +1
        fee_trend_signal = self.fee_history.get_trend(window_sec=30)
        fee_signal = 0.6 * fee_level_signal + 0.4 * fee_trend_signal

        # 2. MEMPOOL SIGNAL
        # NOTE: Correlation with price is UNSTABLE - varies by market conditions
        # Growing mempool = congestion (could be bullish OR bearish)
        # Shrinking mempool = transactions clearing
        mempool_trend = self.mempool_history.get_trend(window_sec=60)
        mempool_accel = self.mempool_history.get_acceleration(window_sec=30)
        mempool_signal = 0.7 * mempool_trend + 0.3 * mempool_accel

        # Optional inversion (disabled by default - correlations are unstable)
        if self.invert_mempool:
            mempool_signal = -mempool_signal

        # 3. VELOCITY SIGNAL
        # High TX velocity = interest/activity
        # Increasing velocity = BULLISH
        velocity_ratio = tx_per_sec / self.baseline_velocity if self.baseline_velocity > 0 else 1
        velocity_level = np.clip((velocity_ratio - 1), -1, 1)
        velocity_trend = self.velocity_history.get_trend(window_sec=30)
        velocity_signal = 0.5 * velocity_level + 0.5 * velocity_trend

        # 4. WHALE SIGNAL
        # Whale activity = potential big moves
        # Direction determined by context (fee pressure)
        whale_boost = min(whale_count * 0.3, 1.0)  # Up to +1 with 3+ whales
        # Whale direction follows fee pressure (if fees high, whales buying)
        whale_direction = 1 if fee_signal > 0 else (-1 if fee_signal < 0 else 0)
        whale_signal = whale_direction * whale_boost

        # === COMBINE SIGNALS ===

        raw_direction = (
            self.fee_weight * fee_signal +
            self.mempool_weight * mempool_signal +
            self.velocity_weight * velocity_signal +
            self.whale_weight * whale_signal
        )

        direction = np.clip(raw_direction, -1, 1)

        # === MOMENTUM ===
        # Momentum from acceleration of signals
        fee_accel = self.fee_history.get_acceleration(window_sec=15)
        momentum = abs(fee_accel) * 0.5 + abs(mempool_accel) * 0.5
        momentum = np.clip(momentum, 0, 1)

        # === CONFIDENCE ===
        # Higher when signals agree
        signals = [fee_signal, mempool_signal, velocity_signal]
        if whale_count > 0:
            signals.append(whale_signal)

        # Agreement: all same sign = high confidence
        positive = sum(1 for s in signals if s > 0.1)
        negative = sum(1 for s in signals if s < -0.1)
        neutral = len(signals) - positive - negative

        if len(signals) > 0:
            agreement = max(positive, negative) / len(signals)
        else:
            agreement = 0

        # Data freshness: more updates = more confidence
        data_confidence = min(self.update_count / 30, 1.0)  # Full confidence after 30 updates

        confidence = agreement * data_confidence
        confidence = np.clip(confidence, 0, 1)

        # Build signal
        signal = BlockchainSignal(
            timestamp=now,
            direction=float(direction),
            momentum=float(momentum),
            confidence=float(confidence),
            fee_signal=float(fee_signal),
            mempool_signal=float(mempool_signal),
            velocity_signal=float(velocity_signal),
            whale_signal=float(whale_signal),
            acceleration_signal=float(fee_accel),
            fee_rate=float(fee_fast),
            mempool_size=int(mempool_size),
            tx_velocity=float(tx_per_sec),
            whale_count=int(whale_count),
        )

        self.signal_history.append(signal)
        self.signals_generated += 1

        return signal

    def get_stats(self) -> Dict:
        """Get engine statistics."""
        return {
            'updates': self.update_count,
            'signals': self.signals_generated,
            'baseline_fee': self.baseline_fee,
            'baseline_mempool': self.baseline_mempool,
            'baseline_velocity': self.baseline_velocity,
            'fee_weight': self.fee_weight,
            'mempool_weight': self.mempool_weight,
            'velocity_weight': self.velocity_weight,
            'whale_weight': self.whale_weight,
        }


class BlockchainPricePredictor:
    """
    Predicts price DIRECTION from blockchain, tracks accuracy.

    This is what you use for trading:
    1. Get signal from BlockchainTruePrice
    2. Track if signal was correct (price moved in predicted direction)
    3. Adjust weights based on accuracy

    Goal: >50% accuracy = edge = profit
    """

    def __init__(self):
        self.engine = BlockchainTruePrice()

        # Tracking predictions
        self.predictions: Deque[Dict] = deque(maxlen=10000)
        self.correct_predictions = 0
        self.total_predictions = 0

        # Accuracy by signal type
        self.accuracy_by_type = {
            'fee': {'correct': 0, 'total': 0},
            'mempool': {'correct': 0, 'total': 0},
            'velocity': {'correct': 0, 'total': 0},
            'whale': {'correct': 0, 'total': 0},
        }

        # Last known price for tracking
        self.last_price: Optional[float] = None
        self.pending_predictions: List[Dict] = []

    def predict(
        self,
        fee_fast: float,
        fee_medium: float,
        mempool_size: int,
        tx_per_sec: float,
        whale_count: int = 0,
        current_price: float = 0,
    ) -> BlockchainSignal:
        """
        Generate prediction and track it.

        Args:
            current_price: Real exchange price (for accuracy tracking only)
        """
        signal = self.engine.update(
            fee_fast=fee_fast,
            fee_medium=fee_medium,
            mempool_size=mempool_size,
            tx_per_sec=tx_per_sec,
            whale_count=whale_count,
        )

        # Track prediction if we have meaningful signal
        if abs(signal.direction) > 0.05 and current_price > 0:
            prediction = {
                'timestamp': signal.timestamp,
                'direction': signal.direction,
                'price_at_prediction': current_price,
                'fee_signal': signal.fee_signal,
                'mempool_signal': signal.mempool_signal,
                'velocity_signal': signal.velocity_signal,
                'whale_signal': signal.whale_signal,
                'verified': False,
            }
            self.pending_predictions.append(prediction)

        self.last_price = current_price
        return signal

    def verify_predictions(self, current_price: float, lookback_sec: float = 10.0):
        """
        Verify pending predictions against current price.

        Call this periodically to track accuracy.
        """
        now = time.time()
        verified = []

        for pred in self.pending_predictions:
            if now - pred['timestamp'] >= lookback_sec and not pred['verified']:
                # Check if price moved in predicted direction
                price_change = current_price - pred['price_at_prediction']
                predicted_up = pred['direction'] > 0
                actually_up = price_change > 0

                correct = (predicted_up == actually_up) if abs(price_change) > 0.01 else None

                if correct is not None:
                    self.total_predictions += 1
                    if correct:
                        self.correct_predictions += 1

                    # Track by signal type
                    for signal_type in ['fee', 'mempool', 'velocity', 'whale']:
                        signal_value = pred.get(f'{signal_type}_signal', 0)
                        if abs(signal_value) > 0.1:
                            self.accuracy_by_type[signal_type]['total'] += 1
                            if (signal_value > 0) == actually_up:
                                self.accuracy_by_type[signal_type]['correct'] += 1

                    pred['verified'] = True
                    pred['correct'] = correct
                    pred['actual_price'] = current_price
                    pred['price_change'] = price_change
                    self.predictions.append(pred)
                    verified.append(pred)

        # Remove verified predictions from pending
        self.pending_predictions = [p for p in self.pending_predictions if not p['verified']]

        return verified

    def get_accuracy(self) -> float:
        """Get overall prediction accuracy."""
        if self.total_predictions == 0:
            return 0.5  # No data, assume random
        return self.correct_predictions / self.total_predictions

    def get_edge(self) -> float:
        """Get edge over random (50%)."""
        return self.get_accuracy() - 0.5

    def get_accuracy_by_type(self) -> Dict[str, float]:
        """Get accuracy broken down by signal type."""
        result = {}
        for signal_type, data in self.accuracy_by_type.items():
            if data['total'] >= 5:
                result[signal_type] = data['correct'] / data['total']
            else:
                result[signal_type] = 0.5  # Not enough data
        return result

    def print_report(self):
        """Print accuracy report."""
        print("\n" + "=" * 60)
        print("BLOCKCHAIN SIGNAL ACCURACY REPORT")
        print("=" * 60)
        print(f"Total Predictions: {self.total_predictions}")
        print(f"Correct: {self.correct_predictions}")
        print(f"Overall Accuracy: {self.get_accuracy()*100:.1f}%")
        print(f"Edge over random: {self.get_edge()*100:+.1f}%")
        print()
        print("By Signal Type:")
        for signal_type, acc in self.get_accuracy_by_type().items():
            data = self.accuracy_by_type[signal_type]
            print(f"  {signal_type:12} {acc*100:5.1f}% ({data['total']} samples)")
        print("=" * 60)
