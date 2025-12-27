#!/usr/bin/env python3
"""
REGIME DETECTION BACKTEST
=========================

Tests the regime detection model against historical data from correlation.db.

This uses the 7,214 verified observations to validate:
1. Does regime detection produce better accuracy than 50%?
2. What RCS thresholds work best?
3. Which exchanges are most predictable?
"""

import sqlite3
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class FlowObservation:
    """A single flow observation from history."""
    timestamp: datetime
    exchange: str
    direction: str  # 'inflow' or 'outflow'
    amount_btc: float
    price_t0: float
    price_t30: Optional[float]
    price_t60: Optional[float]
    price_t300: Optional[float]


@dataclass
class RegimeWindow:
    """Aggregated flow window."""
    inflow_btc: float = 0.0
    outflow_btc: float = 0.0
    whale_inflow_btc: float = 0.0
    whale_outflow_btc: float = 0.0

    @property
    def net_flow(self) -> float:
        return self.outflow_btc - self.inflow_btc

    @property
    def total_flow(self) -> float:
        return self.inflow_btc + self.outflow_btc


def load_historical_data(db_path: str, min_amount: float = 10.0) -> Dict[str, List[FlowObservation]]:
    """Load flow observations from correlation.db."""
    print(f"Loading historical data from {db_path}...")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    query = """
        SELECT timestamp, exchange, direction, amount_btc,
               price_t0, price_t30, price_t60, price_t300
        FROM flows
        WHERE price_t0 IS NOT NULL
          AND price_t300 IS NOT NULL
          AND amount_btc >= ?
        ORDER BY timestamp
    """

    cursor = conn.execute(query, (min_amount,))

    data: Dict[str, List[FlowObservation]] = defaultdict(list)

    for row in cursor:
        obs = FlowObservation(
            timestamp=datetime.fromtimestamp(row['timestamp']),
            exchange=row['exchange'],
            direction=row['direction'].lower(),
            amount_btc=row['amount_btc'],
            price_t0=row['price_t0'],
            price_t30=row['price_t30'],
            price_t60=row['price_t60'],
            price_t300=row['price_t300']
        )
        data[obs.exchange].append(obs)

    conn.close()

    total = sum(len(v) for v in data.values())
    print(f"Loaded {total} observations across {len(data)} exchanges")

    return data


def calculate_regime_metrics(
    windows: List[RegimeWindow],
    current: RegimeWindow,
    whale_threshold: float = 100.0
) -> Dict:
    """Calculate regime detection metrics."""

    if len(windows) < 5:
        return None

    # 1. Z-Score
    net_flows = [w.net_flow for w in windows]
    mu = np.mean(net_flows)
    sigma = np.std(net_flows)

    if sigma < 0.001:
        sigma = 1.0

    current_net = current.net_flow
    z_score = (current_net - mu) / sigma

    # 2. CFI
    recent = windows[-5:]
    cumulative = sum(w.net_flow for w in recent) + current_net
    total_abs = sum(w.total_flow for w in recent) + current.total_flow

    cfi = cumulative / total_abs if total_abs > 0.001 else 0.0

    # 3. Persistence
    directions = [1 if w.net_flow > 0 else -1 for w in recent]
    directions.append(1 if current_net > 0 else -1)
    main_dir = directions[-1]
    persistence = sum(1 for d in directions if d == main_dir) / len(directions)

    # 4. Whale ratio
    total_whale = sum(w.whale_inflow_btc + w.whale_outflow_btc for w in recent)
    total_whale += current.whale_inflow_btc + current.whale_outflow_btc
    total_all = total_abs
    whale_ratio = total_whale / total_all if total_all > 0.001 else 0.0

    # 5. RCS
    rcs = (
        0.30 * abs(z_score) +
        0.30 * abs(cfi) +
        0.20 * persistence +
        0.20 * whale_ratio
    )

    # Direction
    if z_score > 0.5 and cfi > 0.2:
        direction = 1
    elif z_score < -0.5 and cfi < -0.2:
        direction = -1
    else:
        direction = 0

    return {
        'z_score': z_score,
        'cfi': cfi,
        'persistence': persistence,
        'whale_ratio': whale_ratio,
        'rcs': rcs,
        'direction': direction,
        'current_net': current_net,
        'mu': mu,
        'sigma': sigma
    }


def simulate_regime_detection(
    observations: List[FlowObservation],
    window_size_seconds: int = 600,
    lookback_windows: int = 10,
    whale_threshold: float = 100.0
) -> List[Dict]:
    """Simulate regime detection on historical data."""

    results = []
    windows: List[RegimeWindow] = []
    current_window = RegimeWindow()
    window_start = None

    for i, obs in enumerate(observations):
        # Initialize window start
        if window_start is None:
            window_start = obs.timestamp

        # Check if we need new window
        window_elapsed = (obs.timestamp - window_start).total_seconds()

        if window_elapsed >= window_size_seconds:
            # Finalize window
            if current_window.total_flow > 0:
                windows.append(current_window)
                if len(windows) > lookback_windows * 2:
                    windows = windows[-lookback_windows * 2:]

            current_window = RegimeWindow()
            window_start = obs.timestamp

        # Add to current window
        is_whale = obs.amount_btc >= whale_threshold

        if obs.direction == 'inflow':
            current_window.inflow_btc += obs.amount_btc
            if is_whale:
                current_window.whale_inflow_btc += obs.amount_btc
        else:
            current_window.outflow_btc += obs.amount_btc
            if is_whale:
                current_window.whale_outflow_btc += obs.amount_btc

        # Calculate metrics
        metrics = calculate_regime_metrics(windows, current_window, whale_threshold)

        if metrics and metrics['direction'] != 0:
            # Calculate actual price movement
            if obs.price_t300 and obs.price_t0:
                actual_delta = obs.price_t300 - obs.price_t0
                actual_pct = actual_delta / obs.price_t0

                # Direction with noise filter (0.05%)
                if actual_pct > 0.0005:
                    actual_direction = 1
                elif actual_pct < -0.0005:
                    actual_direction = -1
                else:
                    actual_direction = 0

                if actual_direction != 0:
                    predicted = metrics['direction']
                    correct = predicted == actual_direction

                    results.append({
                        'exchange': obs.exchange,
                        'timestamp': obs.timestamp,
                        'z_score': metrics['z_score'],
                        'cfi': metrics['cfi'],
                        'persistence': metrics['persistence'],
                        'whale_ratio': metrics['whale_ratio'],
                        'rcs': metrics['rcs'],
                        'predicted_direction': predicted,
                        'actual_direction': actual_direction,
                        'price_t0': obs.price_t0,
                        'price_t300': obs.price_t300,
                        'pct_change': actual_pct,
                        'correct': correct
                    })

    return results


def analyze_results(results: List[Dict]):
    """Analyze backtest results."""
    print()
    print("=" * 70)
    print("REGIME DETECTION BACKTEST RESULTS")
    print("=" * 70)
    print()

    if not results:
        print("No results to analyze.")
        return

    # Overall accuracy
    correct = sum(1 for r in results if r['correct'])
    accuracy = correct / len(results) * 100

    print(f"Total signals: {len(results)}")
    print(f"Correct:       {correct}")
    print(f"ACCURACY:      {accuracy:.1f}%")
    print()

    # By RCS threshold
    print("BY RCS THRESHOLD:")
    print("-" * 50)
    for threshold in [0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]:
        subset = [r for r in results if r['rcs'] >= threshold]
        if subset:
            correct = sum(1 for r in subset if r['correct'])
            acc = correct / len(subset) * 100
            print(f"  RCS >= {threshold:.1f}: {correct:4}/{len(subset):4} = {acc:5.1f}%")
    print()

    # By |z-score|
    print("BY |Z-SCORE| THRESHOLD:")
    print("-" * 50)
    for threshold in [1.0, 1.5, 2.0, 2.5, 3.0]:
        subset = [r for r in results if abs(r['z_score']) >= threshold]
        if subset:
            correct = sum(1 for r in subset if r['correct'])
            acc = correct / len(subset) * 100
            print(f"  |z| >= {threshold:.1f}: {correct:4}/{len(subset):4} = {acc:5.1f}%")
    print()

    # By |CFI|
    print("BY |CFI| THRESHOLD:")
    print("-" * 50)
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        subset = [r for r in results if abs(r['cfi']) >= threshold]
        if subset:
            correct = sum(1 for r in subset if r['correct'])
            acc = correct / len(subset) * 100
            print(f"  |CFI| >= {threshold:.1f}: {correct:4}/{len(subset):4} = {acc:5.1f}%")
    print()

    # By exchange
    print("BY EXCHANGE:")
    print("-" * 50)
    exchanges = set(r['exchange'] for r in results)
    for exchange in sorted(exchanges):
        subset = [r for r in results if r['exchange'] == exchange]
        if subset:
            correct = sum(1 for r in subset if r['correct'])
            acc = correct / len(subset) * 100
            print(f"  {exchange:15}: {correct:4}/{len(subset):4} = {acc:5.1f}%")
    print()

    # Optimal thresholds
    print("=" * 70)
    print("OPTIMAL CONFIGURATION")
    print("=" * 70)

    best_accuracy = 0
    best_config = {}

    for rcs in [0.6, 0.8, 1.0, 1.2]:
        for z in [1.0, 1.5, 2.0]:
            for cfi in [0.3, 0.4, 0.5]:
                subset = [r for r in results
                         if r['rcs'] >= rcs
                         and abs(r['z_score']) >= z
                         and abs(r['cfi']) >= cfi]

                if len(subset) >= 20:  # Minimum sample size
                    correct = sum(1 for r in subset if r['correct'])
                    acc = correct / len(subset) * 100

                    if acc > best_accuracy:
                        best_accuracy = acc
                        best_config = {
                            'rcs': rcs,
                            'z_threshold': z,
                            'cfi_threshold': cfi,
                            'samples': len(subset),
                            'correct': correct,
                            'accuracy': acc
                        }

    if best_config:
        print()
        print(f"BEST CONFIGURATION:")
        print(f"  RCS threshold:   >= {best_config['rcs']}")
        print(f"  Z-score:         >= {best_config['z_threshold']}")
        print(f"  CFI:             >= {best_config['cfi_threshold']}")
        print(f"  Samples:         {best_config['samples']}")
        print(f"  ACCURACY:        {best_config['accuracy']:.1f}%")

    print()
    print("=" * 70)


def main():
    import sys

    db_path = sys.argv[1] if len(sys.argv) > 1 else "/root/sovereign/correlation.db"

    print("=" * 70)
    print("REGIME DETECTION BACKTEST")
    print("=" * 70)
    print()
    print("Testing the regime detection model against historical data.")
    print("This validates if aggregate patterns work better than 50%.")
    print()

    # Load data
    data = load_historical_data(db_path, min_amount=10.0)

    if not data:
        print("No historical data found.")
        return

    # Run simulation for each exchange
    all_results = []

    for exchange, observations in sorted(data.items()):
        if len(observations) < 20:
            continue

        print(f"\nProcessing {exchange}: {len(observations)} observations...")

        results = simulate_regime_detection(
            observations,
            window_size_seconds=600,  # 10-minute windows
            lookback_windows=10,
            whale_threshold=100.0
        )

        all_results.extend(results)

        if results:
            correct = sum(1 for r in results if r['correct'])
            acc = correct / len(results) * 100
            print(f"  Signals: {len(results)}, Accuracy: {acc:.1f}%")

    # Analyze all results
    analyze_results(all_results)


if __name__ == "__main__":
    main()
