"""
Edge Calculator
===============

Analyzes collected signals to determine if edge exists.

This is the RenTech moment of truth:
- Does our signal predict price movement?
- What's the actual win rate?
- What's the optimal hold time?
- Is the edge statistically significant?

Usage:
    python -m engine.sovereign.validation.analysis.calculate_edge
"""

import sqlite3
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SignalResult:
    """Result of a single signal."""
    signal_id: int
    timestamp: float
    direction: int
    confidence: float
    price_at_signal: float

    # Outcomes at different horizons
    price_1m: float
    price_5m: float
    price_15m: float
    price_60m: float

    # Was prediction correct?
    correct_1m: bool
    correct_5m: bool
    correct_15m: bool
    correct_60m: bool

    # Returns
    return_1m: float
    return_5m: float
    return_15m: float
    return_60m: float


@dataclass
class EdgeMetrics:
    """Edge metrics for a set of signals."""
    total_signals: int
    win_rate_1m: float
    win_rate_5m: float
    win_rate_15m: float
    win_rate_60m: float

    avg_return_1m: float
    avg_return_5m: float
    avg_return_15m: float
    avg_return_60m: float

    # Statistical significance
    std_error_1m: float
    std_error_5m: float
    confidence_interval_1m: Tuple[float, float]
    confidence_interval_5m: Tuple[float, float]

    # Is edge real?
    is_significant_1m: bool
    is_significant_5m: bool
    sharpe_ratio: float
    expectancy_per_trade: float

    # Optimal parameters
    optimal_horizon: str
    optimal_confidence_threshold: float


class EdgeCalculator:
    """
    Calculates edge from collected signal and price data.

    The moment of truth: Is there alpha?
    """

    def __init__(self,
                 signals_db: str = "data/validation/signals.db",
                 prices_db: str = "data/validation/prices.db"):
        """
        Initialize calculator.

        Args:
            signals_db: Path to signals database
            prices_db: Path to prices database
        """
        self.signals_db = Path(signals_db)
        self.prices_db = Path(prices_db)

        if not self.signals_db.exists():
            raise FileNotFoundError(f"Signals DB not found: {self.signals_db}")
        if not self.prices_db.exists():
            raise FileNotFoundError(f"Prices DB not found: {self.prices_db}")

        logger.info(f"EdgeCalculator initialized")
        logger.info(f"  Signals: {self.signals_db}")
        logger.info(f"  Prices: {self.prices_db}")

    def get_price_at(self, timestamp: float, tolerance_ms: int = 2000) -> Optional[float]:
        """Get price at specific timestamp."""
        conn = sqlite3.connect(self.prices_db)
        c = conn.cursor()

        target_ms = int(timestamp * 1000)

        row = c.execute("""
            SELECT price, ABS(timestamp_ms - ?) as diff
            FROM prices
            WHERE timestamp_ms BETWEEN ? AND ?
            ORDER BY diff
            LIMIT 1
        """, (target_ms, target_ms - tolerance_ms, target_ms + tolerance_ms)).fetchone()

        conn.close()
        return row[0] if row else None

    def analyze_signal(self, signal: Dict) -> Optional[SignalResult]:
        """
        Analyze a single signal against price data.

        Returns None if price data not available.
        """
        timestamp = signal['timestamp']
        direction = signal['direction']
        confidence = signal['confidence']
        price_at_signal = signal['price_at_signal']

        # Skip neutral signals
        if direction == 0:
            return None

        # Get prices at future timestamps
        price_1m = self.get_price_at(timestamp + 60)
        price_5m = self.get_price_at(timestamp + 300)
        price_15m = self.get_price_at(timestamp + 900)
        price_60m = self.get_price_at(timestamp + 3600)

        # Skip if missing price data
        if not all([price_at_signal, price_1m, price_5m]):
            return None

        # Calculate returns
        return_1m = (price_1m - price_at_signal) / price_at_signal if price_1m else 0
        return_5m = (price_5m - price_at_signal) / price_at_signal if price_5m else 0
        return_15m = (price_15m - price_at_signal) / price_at_signal if price_15m else 0
        return_60m = (price_60m - price_at_signal) / price_at_signal if price_60m else 0

        # Was prediction correct?
        if direction == 1:  # LONG
            correct_1m = return_1m > 0
            correct_5m = return_5m > 0
            correct_15m = return_15m > 0 if price_15m else False
            correct_60m = return_60m > 0 if price_60m else False
        else:  # SHORT
            correct_1m = return_1m < 0
            correct_5m = return_5m < 0
            correct_15m = return_15m < 0 if price_15m else False
            correct_60m = return_60m < 0 if price_60m else False
            # Flip returns for short
            return_1m = -return_1m
            return_5m = -return_5m
            return_15m = -return_15m
            return_60m = -return_60m

        return SignalResult(
            signal_id=signal['id'],
            timestamp=timestamp,
            direction=direction,
            confidence=confidence,
            price_at_signal=price_at_signal,
            price_1m=price_1m or 0,
            price_5m=price_5m or 0,
            price_15m=price_15m or 0,
            price_60m=price_60m or 0,
            correct_1m=correct_1m,
            correct_5m=correct_5m,
            correct_15m=correct_15m,
            correct_60m=correct_60m,
            return_1m=return_1m,
            return_5m=return_5m,
            return_15m=return_15m,
            return_60m=return_60m,
        )

    def calculate_edge(self,
                       min_confidence: float = 0.0,
                       direction_filter: Optional[int] = None) -> EdgeMetrics:
        """
        Calculate edge metrics for all signals.

        Args:
            min_confidence: Only include signals with confidence >= this
            direction_filter: Only include specific direction (1 or -1)

        Returns:
            EdgeMetrics
        """
        # Load signals
        conn = sqlite3.connect(self.signals_db)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        query = "SELECT * FROM signals WHERE direction != 0"
        params = []

        if min_confidence > 0:
            query += " AND confidence >= ?"
            params.append(min_confidence)

        if direction_filter:
            query += " AND direction = ?"
            params.append(direction_filter)

        rows = c.execute(query, params).fetchall()
        conn.close()

        logger.info(f"Analyzing {len(rows)} signals...")

        # Analyze each signal
        results: List[SignalResult] = []
        for row in rows:
            result = self.analyze_signal(dict(row))
            if result:
                results.append(result)

        if len(results) < 10:
            logger.warning(f"Only {len(results)} signals with price data. Need more data.")
            return self._empty_metrics()

        logger.info(f"Signals with price data: {len(results)}")

        # Calculate metrics
        n = len(results)

        # Win rates
        win_rate_1m = sum(1 for r in results if r.correct_1m) / n
        win_rate_5m = sum(1 for r in results if r.correct_5m) / n
        win_rate_15m = sum(1 for r in results if r.correct_15m) / n if any(r.price_15m for r in results) else 0
        win_rate_60m = sum(1 for r in results if r.correct_60m) / n if any(r.price_60m for r in results) else 0

        # Average returns
        avg_return_1m = np.mean([r.return_1m for r in results])
        avg_return_5m = np.mean([r.return_5m for r in results])
        avg_return_15m = np.mean([r.return_15m for r in results if r.price_15m])
        avg_return_60m = np.mean([r.return_60m for r in results if r.price_60m])

        # Standard errors (for confidence intervals)
        std_error_1m = np.sqrt(win_rate_1m * (1 - win_rate_1m) / n)
        std_error_5m = np.sqrt(win_rate_5m * (1 - win_rate_5m) / n)

        # 95% confidence intervals
        z = 1.96
        ci_1m = (win_rate_1m - z * std_error_1m, win_rate_1m + z * std_error_1m)
        ci_5m = (win_rate_5m - z * std_error_5m, win_rate_5m + z * std_error_5m)

        # Is edge statistically significant? (lower bound > 50%)
        is_significant_1m = ci_1m[0] > 0.50
        is_significant_5m = ci_5m[0] > 0.50

        # Sharpe ratio (annualized, assuming 1-minute trades)
        returns_1m = [r.return_1m for r in results]
        if np.std(returns_1m) > 0:
            sharpe_1m = np.mean(returns_1m) / np.std(returns_1m)
            # Annualize (525,600 minutes per year)
            sharpe_ratio = sharpe_1m * np.sqrt(525600)
        else:
            sharpe_ratio = 0

        # Expectancy per trade (including ~0.1% round-trip costs)
        cost = 0.001  # 0.1% fees
        avg_win = np.mean([r.return_1m for r in results if r.correct_1m]) if win_rate_1m > 0 else 0
        avg_loss = np.mean([abs(r.return_1m) for r in results if not r.correct_1m]) if win_rate_1m < 1 else 0
        expectancy = (win_rate_1m * avg_win) - ((1 - win_rate_1m) * avg_loss) - cost

        # Find optimal horizon
        horizons = {
            '1m': win_rate_1m,
            '5m': win_rate_5m,
            '15m': win_rate_15m,
            '60m': win_rate_60m,
        }
        optimal_horizon = max(horizons, key=horizons.get)

        # Find optimal confidence threshold
        optimal_conf = self._find_optimal_confidence(results)

        return EdgeMetrics(
            total_signals=n,
            win_rate_1m=win_rate_1m,
            win_rate_5m=win_rate_5m,
            win_rate_15m=win_rate_15m,
            win_rate_60m=win_rate_60m,
            avg_return_1m=avg_return_1m,
            avg_return_5m=avg_return_5m,
            avg_return_15m=avg_return_15m,
            avg_return_60m=avg_return_60m,
            std_error_1m=std_error_1m,
            std_error_5m=std_error_5m,
            confidence_interval_1m=ci_1m,
            confidence_interval_5m=ci_5m,
            is_significant_1m=is_significant_1m,
            is_significant_5m=is_significant_5m,
            sharpe_ratio=sharpe_ratio,
            expectancy_per_trade=expectancy,
            optimal_horizon=optimal_horizon,
            optimal_confidence_threshold=optimal_conf,
        )

    def _find_optimal_confidence(self, results: List[SignalResult]) -> float:
        """Find confidence threshold that maximizes edge."""
        best_conf = 0.5
        best_edge = 0

        for conf in np.arange(0.5, 0.9, 0.05):
            filtered = [r for r in results if r.confidence >= conf]
            if len(filtered) < 30:
                continue

            win_rate = sum(1 for r in filtered if r.correct_1m) / len(filtered)
            edge = win_rate - 0.5

            if edge > best_edge:
                best_edge = edge
                best_conf = conf

        return best_conf

    def _empty_metrics(self) -> EdgeMetrics:
        """Return empty metrics."""
        return EdgeMetrics(
            total_signals=0,
            win_rate_1m=0, win_rate_5m=0, win_rate_15m=0, win_rate_60m=0,
            avg_return_1m=0, avg_return_5m=0, avg_return_15m=0, avg_return_60m=0,
            std_error_1m=0, std_error_5m=0,
            confidence_interval_1m=(0, 0), confidence_interval_5m=(0, 0),
            is_significant_1m=False, is_significant_5m=False,
            sharpe_ratio=0, expectancy_per_trade=0,
            optimal_horizon='', optimal_confidence_threshold=0.5,
        )

    def analyze_by_direction(self) -> Dict[str, EdgeMetrics]:
        """Analyze edge separately for LONG and SHORT signals."""
        return {
            'long': self.calculate_edge(direction_filter=1),
            'short': self.calculate_edge(direction_filter=-1),
            'all': self.calculate_edge(),
        }

    def analyze_by_confidence_bucket(self) -> Dict[str, EdgeMetrics]:
        """Analyze edge by confidence level."""
        buckets = {}
        for conf in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]:
            buckets[f'>={conf}'] = self.calculate_edge(min_confidence=conf)
        return buckets

    def generate_report(self) -> str:
        """Generate full analysis report."""
        metrics = self.calculate_edge()
        by_direction = self.analyze_by_direction()
        by_confidence = self.analyze_by_confidence_bucket()

        report = []
        report.append("=" * 70)
        report.append("EDGE ANALYSIS REPORT")
        report.append("=" * 70)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Overall metrics
        report.append("OVERALL METRICS")
        report.append("-" * 40)
        report.append(f"Total Signals Analyzed: {metrics.total_signals:,}")
        report.append("")
        report.append("Win Rates by Horizon:")
        report.append(f"  1 minute:  {metrics.win_rate_1m*100:.2f}%")
        report.append(f"  5 minutes: {metrics.win_rate_5m*100:.2f}%")
        report.append(f"  15 minutes: {metrics.win_rate_15m*100:.2f}%")
        report.append(f"  60 minutes: {metrics.win_rate_60m*100:.2f}%")
        report.append("")
        report.append("Average Returns:")
        report.append(f"  1 minute:  {metrics.avg_return_1m*100:.4f}%")
        report.append(f"  5 minutes: {metrics.avg_return_5m*100:.4f}%")
        report.append("")

        # Statistical significance
        report.append("STATISTICAL SIGNIFICANCE")
        report.append("-" * 40)
        report.append(f"1m Win Rate 95% CI: [{metrics.confidence_interval_1m[0]*100:.2f}%, "
                     f"{metrics.confidence_interval_1m[1]*100:.2f}%]")
        report.append(f"5m Win Rate 95% CI: [{metrics.confidence_interval_5m[0]*100:.2f}%, "
                     f"{metrics.confidence_interval_5m[1]*100:.2f}%]")
        report.append("")
        report.append(f"Edge Significant (1m): {'YES' if metrics.is_significant_1m else 'NO'}")
        report.append(f"Edge Significant (5m): {'YES' if metrics.is_significant_5m else 'NO'}")
        report.append("")

        # Key metrics
        report.append("KEY METRICS")
        report.append("-" * 40)
        report.append(f"Sharpe Ratio (annualized): {metrics.sharpe_ratio:.2f}")
        report.append(f"Expectancy per Trade: {metrics.expectancy_per_trade*100:.4f}%")
        report.append(f"Optimal Horizon: {metrics.optimal_horizon}")
        report.append(f"Optimal Confidence Threshold: {metrics.optimal_confidence_threshold:.2f}")
        report.append("")

        # By direction
        report.append("BY DIRECTION")
        report.append("-" * 40)
        for direction, m in by_direction.items():
            report.append(f"{direction.upper()}: {m.total_signals} signals, "
                         f"{m.win_rate_1m*100:.2f}% win rate (1m)")
        report.append("")

        # By confidence
        report.append("BY CONFIDENCE LEVEL")
        report.append("-" * 40)
        for bucket, m in by_confidence.items():
            if m.total_signals > 0:
                report.append(f"{bucket}: {m.total_signals} signals, "
                             f"{m.win_rate_1m*100:.2f}% win rate")
        report.append("")

        # Decision
        report.append("=" * 70)
        report.append("DECISION")
        report.append("=" * 70)

        if metrics.is_significant_1m and metrics.expectancy_per_trade > 0:
            report.append("EDGE CONFIRMED - Proceed to paper trading")
            report.append("")
            report.append("Recommended Parameters:")
            report.append(f"  Hold Time: {metrics.optimal_horizon}")
            report.append(f"  Min Confidence: {metrics.optimal_confidence_threshold:.2f}")
            report.append(f"  Expected Return/Trade: {metrics.expectancy_per_trade*100:.3f}%")
        elif metrics.win_rate_1m > 0.51:
            report.append("POTENTIAL EDGE - Need more data")
            report.append("")
            report.append("Continue collecting signals for stronger statistical confidence.")
        else:
            report.append("NO SIGNIFICANT EDGE DETECTED")
            report.append("")
            report.append("Options:")
            report.append("  1. Collect more data")
            report.append("  2. Refine signal generation")
            report.append("  3. Test different confidence thresholds")

        report.append("=" * 70)

        return "\n".join(report)


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Calculate edge from collected data")
    parser.add_argument("--signals-db", type=str, default="data/validation/signals.db",
                       help="Path to signals database")
    parser.add_argument("--prices-db", type=str, default="data/validation/prices.db",
                       help="Path to prices database")
    parser.add_argument("--output", type=str, default=None,
                       help="Output report to file")

    args = parser.parse_args()

    try:
        calculator = EdgeCalculator(args.signals_db, args.prices_db)
        report = calculator.generate_report()

        print(report)

        if args.output:
            Path(args.output).write_text(report)
            print(f"\nReport saved to: {args.output}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nMake sure you have collected data first:")
        print("  python -m engine.sovereign.validation.data_collector")


if __name__ == "__main__":
    main()
