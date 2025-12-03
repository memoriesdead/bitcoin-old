"""
CALIBRATION SYSTEM - PAPER vs LIVE COMPARISON
==============================================
Compares realistic paper trading with live trading to calibrate simulation parameters.

PURPOSE:
1. Run paper and live trading simultaneously
2. Compare key metrics: fill rate, slippage, win rate, PnL
3. Auto-adjust simulation parameters to match reality
4. Track calibration accuracy over time

CALIBRATION TARGETS:
- Fill rate: Paper should match live within 5%
- Slippage: Paper should match live within 2 bps
- Win rate: Paper should match live within 3%
- MEV attacks: Paper should match live frequency

USAGE:
    from engine.market.calibration import CalibrationSystem

    calibrator = CalibrationSystem()
    calibrator.start_calibration(paper_engine, live_engine)

    # After trading session
    adjustments = calibrator.calculate_adjustments()
    calibrator.apply_adjustments(realistic_simulator)
"""
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from collections import deque


@dataclass
class TradeComparison:
    """Single trade comparison between paper and live."""
    timestamp: float
    side: str

    # Paper trading results
    paper_filled: bool = False
    paper_price: float = 0.0
    paper_slippage_bps: float = 0.0
    paper_queue_position: float = 0.0

    # Live trading results
    live_filled: bool = False
    live_price: float = 0.0
    live_slippage_bps: float = 0.0
    live_latency_ms: float = 0.0

    # Comparison metrics
    price_delta_bps: float = 0.0
    fill_match: bool = False


@dataclass
class CalibrationMetrics:
    """Aggregated calibration metrics."""
    # Fill rate comparison
    paper_fill_rate: float = 0.0
    live_fill_rate: float = 0.0
    fill_rate_delta: float = 0.0

    # Slippage comparison
    paper_avg_slippage_bps: float = 0.0
    live_avg_slippage_bps: float = 0.0
    slippage_delta_bps: float = 0.0

    # Win rate comparison
    paper_win_rate: float = 0.0
    live_win_rate: float = 0.0
    win_rate_delta: float = 0.0

    # MEV/Competition metrics
    paper_mev_rate: float = 0.0
    live_mev_rate: float = 0.0

    # Queue position accuracy
    avg_queue_position_error: float = 0.0

    # Overall calibration score (0-100)
    calibration_score: float = 0.0

    # Timestamp
    timestamp: float = 0.0


@dataclass
class CalibrationConfig:
    """Configuration for calibration system."""
    # Target accuracy thresholds
    fill_rate_tolerance: float = 0.05  # 5%
    slippage_tolerance_bps: float = 2.0  # 2 bps
    win_rate_tolerance: float = 0.03  # 3%

    # Adjustment limits
    max_adjustment_per_iteration: float = 0.10  # 10% max change

    # Smoothing
    ema_alpha: float = 0.2  # EMA smoothing for adjustments

    # Persistence
    calibration_file: str = "calibration_state.json"

    # Minimum trades for calibration
    min_trades_for_calibration: int = 50


class CalibrationSystem:
    """
    Calibration system for TRUE 1:1 paper trading simulation.

    Compares paper vs live results and adjusts simulation parameters.
    """

    def __init__(self, config: CalibrationConfig = None):
        self.config = config or CalibrationConfig()

        # Trade history
        self.paper_trades: List[Dict] = []
        self.live_trades: List[Dict] = []
        self.comparisons: List[TradeComparison] = []

        # Metrics history
        self.metrics_history: deque = deque(maxlen=1000)
        self.current_metrics: Optional[CalibrationMetrics] = None

        # Calibration adjustments
        self.adjustments: Dict[str, float] = {
            'fill_probability_mult': 1.0,
            'slippage_mult': 1.0,
            'mev_probability_mult': 1.0,
            'latency_race_mult': 1.0,
            'queue_position_offset': 0.0,
        }

        # Load previous calibration state
        self._load_calibration_state()

    def record_paper_trade(
        self,
        timestamp: float,
        side: str,
        filled: bool,
        price: float,
        slippage_bps: float,
        queue_position: float,
        rejection_reason: str = None,
    ):
        """Record a paper trading result."""
        self.paper_trades.append({
            'timestamp': timestamp,
            'side': side,
            'filled': filled,
            'price': price,
            'slippage_bps': slippage_bps,
            'queue_position': queue_position,
            'rejection_reason': rejection_reason,
        })

    def record_live_trade(
        self,
        timestamp: float,
        side: str,
        filled: bool,
        price: float,
        slippage_bps: float,
        latency_ms: float,
        exchange: str = None,
    ):
        """Record a live trading result."""
        self.live_trades.append({
            'timestamp': timestamp,
            'side': side,
            'filled': filled,
            'price': price,
            'slippage_bps': slippage_bps,
            'latency_ms': latency_ms,
            'exchange': exchange,
        })

    def compare_trades(self, time_window_ms: float = 100.0) -> List[TradeComparison]:
        """
        Compare paper and live trades within a time window.

        Matches paper and live trades by timestamp and side.
        """
        comparisons = []

        # Sort trades by timestamp
        paper_sorted = sorted(self.paper_trades, key=lambda x: x['timestamp'])
        live_sorted = sorted(self.live_trades, key=lambda x: x['timestamp'])

        # Match trades
        live_matched = set()

        for paper in paper_sorted:
            best_match = None
            best_delta = float('inf')

            for i, live in enumerate(live_sorted):
                if i in live_matched:
                    continue

                # Check time window
                time_delta = abs(paper['timestamp'] - live['timestamp']) * 1000
                if time_delta > time_window_ms:
                    continue

                # Check side match
                if paper['side'] != live['side']:
                    continue

                if time_delta < best_delta:
                    best_match = (i, live)
                    best_delta = time_delta

            if best_match:
                live_matched.add(best_match[0])
                live = best_match[1]

                # Calculate price delta
                price_delta_bps = 0.0
                if live['price'] > 0:
                    price_delta_bps = (paper['price'] - live['price']) / live['price'] * 10000

                comparison = TradeComparison(
                    timestamp=paper['timestamp'],
                    side=paper['side'],
                    paper_filled=paper['filled'],
                    paper_price=paper['price'],
                    paper_slippage_bps=paper['slippage_bps'],
                    paper_queue_position=paper['queue_position'],
                    live_filled=live['filled'],
                    live_price=live['price'],
                    live_slippage_bps=live['slippage_bps'],
                    live_latency_ms=live['latency_ms'],
                    price_delta_bps=price_delta_bps,
                    fill_match=(paper['filled'] == live['filled']),
                )
                comparisons.append(comparison)

        self.comparisons.extend(comparisons)
        return comparisons

    def calculate_metrics(self) -> CalibrationMetrics:
        """Calculate calibration metrics from trade history."""
        metrics = CalibrationMetrics(timestamp=time.time())

        # Paper metrics
        if self.paper_trades:
            filled = [t for t in self.paper_trades if t['filled']]
            metrics.paper_fill_rate = len(filled) / len(self.paper_trades)

            if filled:
                metrics.paper_avg_slippage_bps = sum(t['slippage_bps'] for t in filled) / len(filled)

        # Live metrics
        if self.live_trades:
            filled = [t for t in self.live_trades if t['filled']]
            metrics.live_fill_rate = len(filled) / len(self.live_trades)

            if filled:
                metrics.live_avg_slippage_bps = sum(t['slippage_bps'] for t in filled) / len(filled)

        # Calculate deltas
        metrics.fill_rate_delta = metrics.paper_fill_rate - metrics.live_fill_rate
        metrics.slippage_delta_bps = metrics.paper_avg_slippage_bps - metrics.live_avg_slippage_bps

        # Calculate calibration score (0-100)
        score = 100.0

        # Penalize fill rate mismatch
        if abs(metrics.fill_rate_delta) > self.config.fill_rate_tolerance:
            score -= min(30, abs(metrics.fill_rate_delta) * 100)

        # Penalize slippage mismatch
        if abs(metrics.slippage_delta_bps) > self.config.slippage_tolerance_bps:
            score -= min(30, abs(metrics.slippage_delta_bps) * 5)

        # Penalize win rate mismatch
        if abs(metrics.win_rate_delta) > self.config.win_rate_tolerance:
            score -= min(20, abs(metrics.win_rate_delta) * 100)

        # Queue position accuracy from comparisons
        if self.comparisons:
            queue_errors = [abs(c.paper_queue_position - 0.5) for c in self.comparisons]
            metrics.avg_queue_position_error = sum(queue_errors) / len(queue_errors)
            score -= min(20, metrics.avg_queue_position_error * 50)

        metrics.calibration_score = max(0, min(100, score))

        self.current_metrics = metrics
        self.metrics_history.append(metrics)

        return metrics

    def calculate_adjustments(self) -> Dict[str, float]:
        """
        Calculate parameter adjustments to improve calibration.

        Returns adjustment multipliers/offsets to apply to simulator.
        """
        if not self.current_metrics:
            self.calculate_metrics()

        metrics = self.current_metrics
        max_adj = self.config.max_adjustment_per_iteration
        alpha = self.config.ema_alpha

        # Adjust fill probability
        if metrics.paper_fill_rate > metrics.live_fill_rate + 0.01:
            # Paper fills too often - reduce fill probability
            adj = max(1 - max_adj, 1 - (metrics.fill_rate_delta * 2))
            self.adjustments['fill_probability_mult'] = (
                alpha * adj + (1 - alpha) * self.adjustments['fill_probability_mult']
            )
        elif metrics.paper_fill_rate < metrics.live_fill_rate - 0.01:
            # Paper fills too rarely - increase fill probability
            adj = min(1 + max_adj, 1 + abs(metrics.fill_rate_delta * 2))
            self.adjustments['fill_probability_mult'] = (
                alpha * adj + (1 - alpha) * self.adjustments['fill_probability_mult']
            )

        # Adjust slippage
        if metrics.paper_avg_slippage_bps < metrics.live_avg_slippage_bps - 0.5:
            # Paper slippage too low - increase
            adj = min(1 + max_adj, 1 + abs(metrics.slippage_delta_bps) / 10)
            self.adjustments['slippage_mult'] = (
                alpha * adj + (1 - alpha) * self.adjustments['slippage_mult']
            )
        elif metrics.paper_avg_slippage_bps > metrics.live_avg_slippage_bps + 0.5:
            # Paper slippage too high - decrease
            adj = max(1 - max_adj, 1 - abs(metrics.slippage_delta_bps) / 10)
            self.adjustments['slippage_mult'] = (
                alpha * adj + (1 - alpha) * self.adjustments['slippage_mult']
            )

        return self.adjustments

    def apply_adjustments(self, simulator) -> bool:
        """
        Apply calibration adjustments to a RealisticSimulator instance.

        Args:
            simulator: RealisticSimulator instance to calibrate

        Returns:
            True if adjustments were applied
        """
        try:
            # Apply fill probability adjustment
            if hasattr(simulator.config, 'min_fill_probability'):
                original = simulator.config.min_fill_probability
                simulator.config.min_fill_probability = original * self.adjustments['fill_probability_mult']

            # Apply latency race adjustment
            if hasattr(simulator.config, 'latency_race_frequency'):
                original = simulator.config.latency_race_frequency
                simulator.config.latency_race_frequency = min(0.95, original * self.adjustments['latency_race_mult'])

            # Apply MEV probability adjustment
            if hasattr(simulator.config, 'mev_sandwich_probability'):
                original = simulator.config.mev_sandwich_probability
                simulator.config.mev_sandwich_probability = min(0.5, original * self.adjustments['mev_probability_mult'])

            # Save calibration state
            self._save_calibration_state()

            return True

        except Exception as e:
            print(f"[CALIBRATION] Failed to apply adjustments: {e}")
            return False

    def get_report(self) -> str:
        """Generate calibration report."""
        if not self.current_metrics:
            self.calculate_metrics()

        m = self.current_metrics

        report = []
        report.append("=" * 70)
        report.append("CALIBRATION REPORT")
        report.append("=" * 70)

        report.append(f"\nCalibration Score: {m.calibration_score:.1f}/100")

        report.append("\n--- Fill Rate ---")
        report.append(f"Paper: {m.paper_fill_rate:.1%}")
        report.append(f"Live:  {m.live_fill_rate:.1%}")
        report.append(f"Delta: {m.fill_rate_delta:+.1%}")
        status = "OK" if abs(m.fill_rate_delta) <= self.config.fill_rate_tolerance else "NEEDS CALIBRATION"
        report.append(f"Status: {status}")

        report.append("\n--- Slippage ---")
        report.append(f"Paper: {m.paper_avg_slippage_bps:.2f} bps")
        report.append(f"Live:  {m.live_avg_slippage_bps:.2f} bps")
        report.append(f"Delta: {m.slippage_delta_bps:+.2f} bps")
        status = "OK" if abs(m.slippage_delta_bps) <= self.config.slippage_tolerance_bps else "NEEDS CALIBRATION"
        report.append(f"Status: {status}")

        report.append("\n--- Current Adjustments ---")
        for key, value in self.adjustments.items():
            report.append(f"  {key}: {value:.3f}")

        report.append("\n--- Trade Counts ---")
        report.append(f"Paper trades: {len(self.paper_trades)}")
        report.append(f"Live trades:  {len(self.live_trades)}")
        report.append(f"Matched comparisons: {len(self.comparisons)}")

        report.append("=" * 70)

        return "\n".join(report)

    def _save_calibration_state(self):
        """Save calibration state to disk."""
        try:
            state = {
                'adjustments': self.adjustments,
                'metrics': {
                    'calibration_score': self.current_metrics.calibration_score if self.current_metrics else 0,
                    'fill_rate_delta': self.current_metrics.fill_rate_delta if self.current_metrics else 0,
                    'slippage_delta_bps': self.current_metrics.slippage_delta_bps if self.current_metrics else 0,
                },
                'timestamp': time.time(),
                'paper_trade_count': len(self.paper_trades),
                'live_trade_count': len(self.live_trades),
            }

            path = Path(self.config.calibration_file)
            with open(path, 'w') as f:
                json.dump(state, f, indent=2)

        except Exception as e:
            print(f"[CALIBRATION] Failed to save state: {e}")

    def _load_calibration_state(self):
        """Load calibration state from disk."""
        try:
            path = Path(self.config.calibration_file)
            if path.exists():
                with open(path, 'r') as f:
                    state = json.load(f)

                if 'adjustments' in state:
                    self.adjustments.update(state['adjustments'])

                print(f"[CALIBRATION] Loaded previous state (score: {state.get('metrics', {}).get('calibration_score', 0):.1f})")

        except Exception as e:
            print(f"[CALIBRATION] No previous state found: {e}")

    def reset(self):
        """Reset calibration state."""
        self.paper_trades = []
        self.live_trades = []
        self.comparisons = []
        self.current_metrics = None
        self.adjustments = {
            'fill_probability_mult': 1.0,
            'slippage_mult': 1.0,
            'mev_probability_mult': 1.0,
            'latency_race_mult': 1.0,
            'queue_position_offset': 0.0,
        }


# Convenience exports
__all__ = [
    'CalibrationSystem',
    'CalibrationConfig',
    'CalibrationMetrics',
    'TradeComparison',
]
