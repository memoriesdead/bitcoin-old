"""
Validity Monitor - Timeframe-Adaptive Mathematical Engine
==========================================================

Monitors strategy validity and edge decay:
1. Half-life estimation (edge decay detection)
2. Regime change detection (state transitions)
3. Performance tracking (rolling statistics)

This module answers: "Is our edge still valid?"
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque
import time
import math

from .math_primitives import (
    tae_005_edge_halflife,
    tae_005_rolling_edge_strength,
)


@dataclass
class EdgeEstimate:
    """Estimate of edge half-life."""
    half_life: float             # Time for edge to decay 50%
    decay_rate: float            # Lambda in exponential decay
    current_strength: float      # Current edge strength (0-1)
    is_valid: bool              # Is edge still valid?
    estimated_remaining: float   # Estimated time until edge invalid


@dataclass
class RegimeState:
    """Current regime state."""
    regime: str
    confidence: float
    duration: float              # Time in current regime
    transition_prob: float       # Probability of regime change


@dataclass
class PerformanceSnapshot:
    """Snapshot of performance metrics."""
    timestamp: float
    win_rate: float
    profit_factor: float
    sharpe: float
    drawdown: float
    edge_strength: float


class HalfLifeEstimator:
    """
    Estimates how quickly trading edge decays.

    Key insight: Trading edges decay over time as others discover them.
    Monitoring half-life tells us when to adapt.
    """

    def __init__(self, min_samples: int = 20):
        """
        Initialize half-life estimator.

        Args:
            min_samples: Minimum samples for reliable estimate
        """
        self.min_samples = min_samples
        self.pnl_history: List[float] = []
        self.time_history: List[float] = []
        self.start_time: Optional[float] = None

        # Cached estimates
        self.last_estimate: Optional[EdgeEstimate] = None
        self.estimate_time: float = 0.0

    def add_trade(self, pnl: float, timestamp: Optional[float] = None) -> None:
        """
        Add trade outcome.

        Args:
            pnl: Trade PnL
            timestamp: Trade timestamp (uses current time if None)
        """
        if timestamp is None:
            timestamp = time.time()

        if self.start_time is None:
            self.start_time = timestamp

        self.pnl_history.append(pnl)
        self.time_history.append(timestamp - self.start_time)

        # Keep last 500 trades
        if len(self.pnl_history) > 500:
            self.pnl_history = self.pnl_history[-500:]
            self.time_history = self.time_history[-500:]

    def estimate(self) -> EdgeEstimate:
        """
        Estimate current edge half-life.

        Returns:
            EdgeEstimate with half-life and current strength
        """
        if len(self.pnl_history) < self.min_samples:
            return EdgeEstimate(
                half_life=float('inf'),
                decay_rate=0.0,
                current_strength=1.0,
                is_valid=True,
                estimated_remaining=float('inf')
            )

        pnl_arr = np.array(self.pnl_history)
        time_arr = np.array(self.time_history)

        # Calculate half-life
        half_life, decay_rate = tae_005_edge_halflife(pnl_arr, time_arr)

        # Calculate current strength
        strength = tae_005_rolling_edge_strength(pnl_arr, window=20)

        # Determine validity
        # Edge is invalid if strength < 0.3 or half-life is very short
        is_valid = strength > 0.3 and (half_life == float('inf') or half_life > 60)

        # Estimate remaining time
        if decay_rate > 0 and strength > 0.3:
            # Time to reach 0.3 strength: 0.3 = strength * exp(-λt)
            # t = -ln(0.3/strength) / λ
            remaining = -math.log(0.3 / strength) / decay_rate
        else:
            remaining = float('inf')

        estimate = EdgeEstimate(
            half_life=half_life,
            decay_rate=decay_rate,
            current_strength=strength,
            is_valid=is_valid,
            estimated_remaining=remaining
        )

        self.last_estimate = estimate
        self.estimate_time = time.time()

        return estimate

    def get_quick_strength(self, window: int = 20) -> float:
        """Quick estimate of current edge strength."""
        if len(self.pnl_history) < 10:
            return 1.0
        return tae_005_rolling_edge_strength(np.array(self.pnl_history), window)


class RegimeChangeDetector:
    """
    Detects regime changes and state transitions.

    Uses:
    - HMM state probability changes
    - Volatility regime shifts
    - Trend direction changes
    """

    def __init__(self, change_threshold: float = 0.3):
        """
        Initialize regime detector.

        Args:
            change_threshold: Probability change threshold for detection
        """
        self.change_threshold = change_threshold
        self.current_regime = 'unknown'
        self.regime_start_time = time.time()
        self.regime_history: List[Tuple[str, float, float]] = []  # (regime, start, end)

        # State probabilities
        self.last_probs: Optional[Dict[str, float]] = None

    def update(self, regime: str, probabilities: Dict[str, float]) -> bool:
        """
        Update regime state.

        Args:
            regime: Current predicted regime
            probabilities: Probability distribution over regimes

        Returns:
            True if regime changed
        """
        now = time.time()
        changed = False

        if regime != self.current_regime:
            # Record regime change
            self.regime_history.append((
                self.current_regime,
                self.regime_start_time,
                now
            ))
            self.current_regime = regime
            self.regime_start_time = now
            changed = True

            # Trim history
            if len(self.regime_history) > 100:
                self.regime_history = self.regime_history[-100:]

        self.last_probs = probabilities
        return changed

    def get_state(self) -> RegimeState:
        """Get current regime state."""
        now = time.time()
        duration = now - self.regime_start_time

        # Estimate transition probability from history
        if len(self.regime_history) > 5:
            avg_duration = np.mean([end - start for _, start, end in self.regime_history[-10:]])
            # P(transition) increases as duration approaches average
            trans_prob = 1.0 - math.exp(-duration / avg_duration)
        else:
            trans_prob = 0.5

        confidence = self.last_probs.get(self.current_regime, 0.5) if self.last_probs else 0.5

        return RegimeState(
            regime=self.current_regime,
            confidence=confidence,
            duration=duration,
            transition_prob=trans_prob
        )

    def is_regime_stable(self, min_duration: float = 60.0) -> bool:
        """Check if regime has been stable for at least min_duration."""
        state = self.get_state()
        return state.duration >= min_duration and state.confidence > 0.6


class PerformanceTracker:
    """
    Tracks rolling performance statistics.
    """

    def __init__(self, window: int = 100):
        """
        Initialize performance tracker.

        Args:
            window: Rolling window size
        """
        self.window = window
        self.pnl_history: deque = deque(maxlen=window)
        self.win_history: deque = deque(maxlen=window)
        self.trade_times: deque = deque(maxlen=window)

        # High water mark for drawdown
        self.cumulative_pnl = 0.0
        self.high_water_mark = 0.0

        # Snapshots
        self.snapshots: List[PerformanceSnapshot] = []

    def add_trade(self, pnl: float, timestamp: Optional[float] = None) -> None:
        """Add trade to history."""
        if timestamp is None:
            timestamp = time.time()

        self.pnl_history.append(pnl)
        self.win_history.append(pnl > 0)
        self.trade_times.append(timestamp)

        self.cumulative_pnl += pnl
        self.high_water_mark = max(self.high_water_mark, self.cumulative_pnl)

    def get_win_rate(self) -> float:
        """Get rolling win rate."""
        if not self.win_history:
            return 0.5
        return sum(self.win_history) / len(self.win_history)

    def get_profit_factor(self) -> float:
        """Get rolling profit factor."""
        if not self.pnl_history:
            return 1.0

        gains = sum(p for p in self.pnl_history if p > 0)
        losses = abs(sum(p for p in self.pnl_history if p < 0))

        if losses < 1e-10:
            return 10.0 if gains > 0 else 1.0

        return gains / losses

    def get_sharpe(self, annualize: bool = True) -> float:
        """Get rolling Sharpe ratio."""
        if len(self.pnl_history) < 10:
            return 0.0

        returns = np.array(self.pnl_history)
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)

        if std_ret < 1e-10:
            return 0.0

        sharpe = mean_ret / std_ret

        # Annualize (assume ~250 trading days, ~8 trades/day)
        if annualize and len(self.trade_times) > 1:
            trades_per_day = len(self.pnl_history) / max(1, (self.trade_times[-1] - self.trade_times[0]) / 86400)
            sharpe *= math.sqrt(trades_per_day * 250)

        return sharpe

    def get_drawdown(self) -> float:
        """Get current drawdown from high water mark."""
        if self.high_water_mark <= 0:
            return 0.0
        return (self.high_water_mark - self.cumulative_pnl) / self.high_water_mark

    def get_snapshot(self, edge_strength: float = 1.0) -> PerformanceSnapshot:
        """Get current performance snapshot."""
        snapshot = PerformanceSnapshot(
            timestamp=time.time(),
            win_rate=self.get_win_rate(),
            profit_factor=self.get_profit_factor(),
            sharpe=self.get_sharpe(),
            drawdown=self.get_drawdown(),
            edge_strength=edge_strength
        )
        self.snapshots.append(snapshot)

        # Keep last 1000 snapshots
        if len(self.snapshots) > 1000:
            self.snapshots = self.snapshots[-500:]

        return snapshot


class ValidityMonitor:
    """
    Main validity monitoring engine.

    Combines half-life estimation, regime detection,
    and performance tracking to determine if edge is still valid.
    """

    def __init__(
        self,
        edge_threshold: float = 0.3,
        regime_min_duration: float = 60.0
    ):
        """
        Initialize validity monitor.

        Args:
            edge_threshold: Minimum edge strength for validity
            regime_min_duration: Minimum regime duration for stability
        """
        self.edge_threshold = edge_threshold
        self.regime_min_duration = regime_min_duration

        # Sub-components
        self.half_life_estimator = HalfLifeEstimator()
        self.regime_detector = RegimeChangeDetector()
        self.performance_tracker = PerformanceTracker()

        # Validity state
        self.is_valid = True
        self.validity_reasons: List[str] = []

    def add_trade(
        self,
        pnl: float,
        timestamp: Optional[float] = None
    ) -> None:
        """
        Add trade outcome to all trackers.

        Args:
            pnl: Trade PnL
            timestamp: Trade timestamp
        """
        self.half_life_estimator.add_trade(pnl, timestamp)
        self.performance_tracker.add_trade(pnl, timestamp)

    def update_regime(
        self,
        regime: str,
        probabilities: Dict[str, float]
    ) -> bool:
        """
        Update regime state.

        Args:
            regime: Current regime
            probabilities: Regime probabilities

        Returns:
            True if regime changed
        """
        return self.regime_detector.update(regime, probabilities)

    def check_validity(self) -> Tuple[bool, List[str]]:
        """
        Check if trading edge is still valid.

        Returns:
            Tuple of (is_valid, reasons)
        """
        reasons = []
        is_valid = True

        # Check edge strength
        edge = self.half_life_estimator.estimate()
        if edge.current_strength < self.edge_threshold:
            is_valid = False
            reasons.append(f"Edge strength too low: {edge.current_strength:.2f}")

        if edge.half_life < 60:
            is_valid = False
            reasons.append(f"Edge half-life too short: {edge.half_life:.1f}s")

        # Check performance
        perf = self.performance_tracker
        if perf.get_win_rate() < 0.45:
            is_valid = False
            reasons.append(f"Win rate too low: {perf.get_win_rate():.1%}")

        if perf.get_profit_factor() < 1.0:
            is_valid = False
            reasons.append(f"Profit factor < 1: {perf.get_profit_factor():.2f}")

        if perf.get_drawdown() > 0.25:
            is_valid = False
            reasons.append(f"Drawdown too high: {perf.get_drawdown():.1%}")

        # Check regime stability
        regime = self.regime_detector.get_state()
        if regime.transition_prob > 0.7:
            reasons.append(f"High regime change probability: {regime.transition_prob:.1%}")
            # Don't invalidate, just warn

        self.is_valid = is_valid
        self.validity_reasons = reasons

        return is_valid, reasons

    def should_trade(self) -> bool:
        """
        Quick check if we should be trading.

        Returns:
            True if safe to trade
        """
        is_valid, _ = self.check_validity()
        return is_valid and self.regime_detector.is_regime_stable(self.regime_min_duration)

    def get_edge_estimate(self) -> EdgeEstimate:
        """Get current edge estimate."""
        return self.half_life_estimator.estimate()

    def get_regime_state(self) -> RegimeState:
        """Get current regime state."""
        return self.regime_detector.get_state()

    def get_performance_snapshot(self) -> PerformanceSnapshot:
        """Get current performance snapshot."""
        edge = self.half_life_estimator.get_quick_strength()
        return self.performance_tracker.get_snapshot(edge)

    def get_diagnostics(self) -> Dict:
        """Get diagnostic information."""
        edge = self.half_life_estimator.estimate()
        regime = self.regime_detector.get_state()
        perf = self.performance_tracker

        return {
            'is_valid': self.is_valid,
            'reasons': self.validity_reasons,
            'edge': {
                'half_life': edge.half_life,
                'decay_rate': edge.decay_rate,
                'strength': edge.current_strength,
                'remaining': edge.estimated_remaining,
            },
            'regime': {
                'current': regime.regime,
                'confidence': regime.confidence,
                'duration': regime.duration,
                'transition_prob': regime.transition_prob,
            },
            'performance': {
                'win_rate': perf.get_win_rate(),
                'profit_factor': perf.get_profit_factor(),
                'sharpe': perf.get_sharpe(),
                'drawdown': perf.get_drawdown(),
            }
        }

    def reset(self) -> None:
        """Reset all state."""
        self.half_life_estimator = HalfLifeEstimator()
        self.regime_detector = RegimeChangeDetector()
        self.performance_tracker = PerformanceTracker()
        self.is_valid = True
        self.validity_reasons = []
