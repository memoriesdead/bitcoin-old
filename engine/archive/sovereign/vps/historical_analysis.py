#!/usr/bin/env python3
"""
OVERNIGHT HISTORICAL ANALYSIS
=============================
Runs exhaustive backtests on all available data.
Tests every signal combination to find edges.
"""

import sqlite3
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# Paths
DATA_DIR = Path("/root/validation/data")
RESULTS_FILE = DATA_DIR / "historical_results.json"
LOG_FILE = DATA_DIR / "historical_analysis.log"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class HistoricalAnalyzer:
    """Exhaustive historical backtesting."""

    def __init__(self):
        self.results = []
        self.best_strategies = []

    def load_live_data(self) -> np.ndarray:
        """Load live collected data from metrics.db."""
        logger.info("Loading live collected data...")

        conn = sqlite3.connect(DATA_DIR / "metrics.db")
        cursor = conn.execute("""
            SELECT timestamp, tx_count, total_volume_btc, tx_whale,
                   tx_mega, consolidation_ratio, price
            FROM metrics
            WHERE price > 0
            ORDER BY timestamp
        """)
        data = cursor.fetchall()
        conn.close()

        if not data:
            logger.warning("No live data found!")
            return None

        logger.info(f"Loaded {len(data):,} rows of live data")
        return np.array(data)

    def calculate_signals(self, data: np.ndarray, lookback: int = 60) -> Dict[str, np.ndarray]:
        """Calculate all possible signals from data."""
        n = len(data)

        # Extract columns
        timestamps = data[:, 0]
        tx_count = data[:, 1].astype(float)
        volume = data[:, 2].astype(float)
        whales = data[:, 3].astype(float)
        megas = data[:, 4].astype(float)
        consol = data[:, 5].astype(float)
        price = data[:, 6].astype(float)

        # Calculate rolling stats
        signals = {}

        # Z-scores
        for name, arr in [('tx', tx_count), ('vol', volume), ('whale', whales), ('consol', consol)]:
            zscore = np.zeros(n)
            for i in range(lookback, n):
                window = arr[i-lookback:i]
                mean = np.mean(window)
                std = np.std(window)
                if std > 0:
                    zscore[i] = (arr[i] - mean) / std
            signals[f'{name}_zscore'] = zscore

        # Whale detection (binary)
        signals['whale_present'] = (whales > 0).astype(float)
        signals['mega_present'] = (megas > 0).astype(float)

        # Rate of change
        for name, arr in [('tx', tx_count), ('vol', volume)]:
            roc = np.zeros(n)
            for i in range(10, n):
                if arr[i-10] > 0:
                    roc[i] = (arr[i] / arr[i-10] - 1) * 100
            signals[f'{name}_roc'] = roc

        # Price and returns
        signals['price'] = price

        # Forward returns (what we're trying to predict)
        for mins in [1, 2, 5, 10, 15]:
            shift = mins * 60  # seconds
            fwd_ret = np.zeros(n)
            for i in range(n - shift):
                if price[i] > 0:
                    fwd_ret[i] = (price[i + shift] / price[i] - 1) * 10000  # bps
            signals[f'return_{mins}m'] = fwd_ret

        return signals

    def test_strategy(
        self,
        signals: Dict[str, np.ndarray],
        signal_name: str,
        threshold: float,
        direction: int,  # 1=LONG on signal, -1=SHORT on signal
        hold_mins: int,
        fee_bps: float = 10.0
    ) -> Dict:
        """Test a single strategy configuration."""

        signal = signals[signal_name]
        returns = signals[f'return_{hold_mins}m']
        n = len(signal)

        # Find entry points
        if direction == 1:  # LONG when signal > threshold
            entries = signal > threshold
        else:  # SHORT when signal > threshold
            entries = signal > threshold

        # Get returns at entry points
        entry_returns = returns[entries]

        # Adjust for direction (SHORT profits when price drops)
        if direction == -1:
            entry_returns = -entry_returns

        # Subtract fees
        net_returns = entry_returns - fee_bps

        if len(net_returns) < 5:
            return None

        # Calculate stats
        n_trades = len(net_returns)
        wins = np.sum(net_returns > 0)
        win_rate = wins / n_trades * 100
        avg_return = np.mean(net_returns)
        total_return = np.sum(net_returns)
        std_return = np.std(net_returns)
        sharpe = avg_return / std_return * np.sqrt(252 * 24 * 12) if std_return > 0 else 0  # Annualized for 5-min bars

        # T-test for significance
        from scipy import stats
        t_stat, p_value = stats.ttest_1samp(net_returns, 0)

        return {
            'signal': signal_name,
            'threshold': float(threshold),
            'direction': 'LONG' if direction == 1 else 'SHORT',
            'hold_mins': int(hold_mins),
            'n_trades': int(n_trades),
            'win_rate': float(round(win_rate, 1)),
            'avg_return_bps': float(round(avg_return, 2)),
            'total_return_bps': float(round(total_return, 1)),
            'sharpe': float(round(sharpe, 2)),
            't_stat': float(round(t_stat, 3)),
            'p_value': float(round(p_value, 4)),
            'significant': bool(p_value < 0.05 and avg_return > 0)
        }

    def run_exhaustive_backtest(self, signals: Dict[str, np.ndarray]) -> List[Dict]:
        """Test all strategy combinations."""

        results = []

        # Signal names to test
        signal_names = ['tx_zscore', 'vol_zscore', 'whale_zscore', 'consol_zscore',
                       'whale_present', 'mega_present', 'tx_roc', 'vol_roc']

        # Thresholds to test
        thresholds = [1.0, 1.5, 2.0, 2.5, 3.0]

        # Directions
        directions = [1, -1]  # LONG or SHORT

        # Hold times
        hold_times = [1, 2, 5, 10, 15]

        total_tests = len(signal_names) * len(thresholds) * len(directions) * len(hold_times)
        logger.info(f"Running {total_tests} strategy combinations...")

        count = 0
        for sig_name in signal_names:
            for thresh in thresholds:
                for direction in directions:
                    for hold in hold_times:
                        try:
                            result = self.test_strategy(
                                signals, sig_name, thresh, direction, hold
                            )
                            if result:
                                results.append(result)

                                # Log significant findings immediately
                                if result['significant']:
                                    logger.info(f"*** SIGNIFICANT: {result['signal']} > {result['threshold']} "
                                              f"{result['direction']} {result['hold_mins']}m: "
                                              f"{result['avg_return_bps']:+.1f}bps, p={result['p_value']:.4f}")
                        except Exception as e:
                            pass

                        count += 1
                        if count % 100 == 0:
                            logger.info(f"Progress: {count}/{total_tests}")

        return results

    def test_combined_signals(self, signals: Dict[str, np.ndarray]) -> List[Dict]:
        """Test combinations of multiple signals."""

        results = []
        logger.info("Testing combined signals...")

        # Combine tx + vol
        combined_tests = [
            ('tx_vol_spike', 'tx_zscore', 'vol_zscore', 1.5, 1.5),
            ('tx_vol_spike_high', 'tx_zscore', 'vol_zscore', 2.0, 2.0),
            ('tx_whale', 'tx_zscore', 'whale_present', 1.5, 0.5),
        ]

        for name, sig1_name, sig2_name, thresh1, thresh2 in combined_tests:
            sig1 = signals[sig1_name]
            sig2 = signals[sig2_name]

            # Combined condition
            combined = (sig1 > thresh1) & (sig2 > thresh2)

            for hold in [1, 2, 5, 10]:
                returns = signals[f'return_{hold}m']

                for direction in [1, -1]:
                    entry_returns = returns[combined]
                    if direction == -1:
                        entry_returns = -entry_returns

                    net_returns = entry_returns - 10  # fees

                    if len(net_returns) < 5:
                        continue

                    from scipy import stats
                    t_stat, p_value = stats.ttest_1samp(net_returns, 0)

                    result = {
                        'signal': f'{name}',
                        'threshold': f'{thresh1}/{thresh2}',
                        'direction': 'LONG' if direction == 1 else 'SHORT',
                        'hold_mins': int(hold),
                        'n_trades': int(len(net_returns)),
                        'win_rate': float(round(np.sum(net_returns > 0) / len(net_returns) * 100, 1)),
                        'avg_return_bps': float(round(np.mean(net_returns), 2)),
                        'total_return_bps': float(round(np.sum(net_returns), 1)),
                        'sharpe': float(round(np.mean(net_returns) / np.std(net_returns) * np.sqrt(252*24*12), 2)) if np.std(net_returns) > 0 else 0.0,
                        't_stat': float(round(t_stat, 3)),
                        'p_value': float(round(p_value, 4)),
                        'significant': bool(p_value < 0.05 and np.mean(net_returns) > 0)
                    }
                    results.append(result)

                    if result['significant']:
                        logger.info(f"*** COMBINED SIGNIFICANT: {name} {result['direction']} "
                                  f"{hold}m: {result['avg_return_bps']:+.1f}bps, p={result['p_value']:.4f}")

        return results

    def analyze_by_time(self, signals: Dict[str, np.ndarray], data: np.ndarray) -> List[Dict]:
        """Analyze if signals work better at certain times."""

        logger.info("Analyzing time-based patterns...")
        results = []

        timestamps = data[:, 0]

        # Convert to hours (UTC)
        hours = np.array([(int(ts) % 86400) // 3600 for ts in timestamps])

        # Test each signal during different time windows
        time_windows = [
            ('asian', 0, 8),      # 00:00-08:00 UTC
            ('european', 8, 16),   # 08:00-16:00 UTC
            ('american', 16, 24),  # 16:00-24:00 UTC
        ]

        for window_name, start_hour, end_hour in time_windows:
            time_mask = (hours >= start_hour) & (hours < end_hour)

            for sig_name in ['tx_zscore', 'vol_zscore']:
                for direction in [1, -1]:
                    for thresh in [1.5, 2.0]:
                        signal = signals[sig_name]
                        returns = signals['return_5m']

                        # Apply time filter
                        condition = (signal > thresh) & time_mask
                        entry_returns = returns[condition]

                        if direction == -1:
                            entry_returns = -entry_returns

                        net_returns = entry_returns - 10

                        if len(net_returns) < 5:
                            continue

                        from scipy import stats
                        t_stat, p_value = stats.ttest_1samp(net_returns, 0)

                        result = {
                            'signal': f'{sig_name}_{window_name}',
                            'threshold': float(thresh),
                            'direction': 'LONG' if direction == 1 else 'SHORT',
                            'hold_mins': 5,
                            'n_trades': int(len(net_returns)),
                            'win_rate': float(round(np.sum(net_returns > 0) / len(net_returns) * 100, 1)),
                            'avg_return_bps': float(round(np.mean(net_returns), 2)),
                            'p_value': float(round(p_value, 4)),
                            'significant': bool(p_value < 0.05 and np.mean(net_returns) > 0)
                        }
                        results.append(result)

                        if result['significant']:
                            logger.info(f"*** TIME SIGNIFICANT: {sig_name} {window_name} "
                                      f"{result['direction']}: {result['avg_return_bps']:+.1f}bps")

        return results

    def save_results(self, all_results: List[Dict]):
        """Save all results to JSON file."""

        # Sort by significance and return
        significant = [r for r in all_results if r.get('significant', False)]
        significant.sort(key=lambda x: x.get('avg_return_bps', 0), reverse=True)

        # Sort all by p-value
        all_results.sort(key=lambda x: x.get('p_value', 1))

        output = {
            'timestamp': datetime.now().isoformat(),
            'total_strategies_tested': len(all_results),
            'significant_strategies': len(significant),
            'best_strategies': significant[:20],
            'all_results': all_results[:100]  # Top 100 by p-value
        }

        with open(RESULTS_FILE, 'w') as f:
            json.dump(output, f, indent=2)

        logger.info(f"Results saved to {RESULTS_FILE}")

    def run(self):
        """Run full analysis."""

        logger.info("="*60)
        logger.info("  HISTORICAL ANALYSIS STARTING")
        logger.info("="*60)

        start_time = time.time()

        # Load data
        data = self.load_live_data()
        if data is None or len(data) < 100:
            logger.error("Not enough data for analysis. Need at least 100 rows.")
            return

        # Calculate signals
        logger.info("Calculating signals...")
        signals = self.calculate_signals(data)

        all_results = []

        # Run exhaustive backtest
        results = self.run_exhaustive_backtest(signals)
        all_results.extend(results)

        # Test combined signals
        combined = self.test_combined_signals(signals)
        all_results.extend(combined)

        # Time-based analysis
        time_results = self.analyze_by_time(signals, data)
        all_results.extend(time_results)

        # Save results
        self.save_results(all_results)

        # Summary
        elapsed = time.time() - start_time
        significant = [r for r in all_results if r.get('significant', False)]

        logger.info("="*60)
        logger.info("  ANALYSIS COMPLETE")
        logger.info("="*60)
        logger.info(f"Time elapsed: {elapsed/60:.1f} minutes")
        logger.info(f"Strategies tested: {len(all_results)}")
        logger.info(f"Significant edges found: {len(significant)}")

        if significant:
            logger.info("")
            logger.info("TOP 5 SIGNIFICANT STRATEGIES:")
            for i, s in enumerate(significant[:5]):
                logger.info(f"  {i+1}. {s['signal']} > {s['threshold']} {s['direction']} {s['hold_mins']}m")
                logger.info(f"     Return: {s['avg_return_bps']:+.1f}bps | Win: {s['win_rate']}% | p={s['p_value']}")
        else:
            logger.info("")
            logger.info("No statistically significant edges found yet.")
            logger.info("Need more data - analysis will run again as data accumulates.")

        logger.info("="*60)


def main():
    analyzer = HistoricalAnalyzer()

    # Run analysis in a loop (every hour)
    while True:
        try:
            analyzer.run()
            logger.info("Sleeping 1 hour before next analysis...")
            time.sleep(3600)  # 1 hour
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            break
        except Exception as e:
            logger.error(f"Error in analysis: {e}")
            time.sleep(300)  # 5 min on error


if __name__ == "__main__":
    main()
