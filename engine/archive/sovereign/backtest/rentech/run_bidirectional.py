"""
Bidirectional RenTech Backtest - LONG vs SHORT Analysis
=======================================================

Tests ALL patterns in BOTH directions on 16 years Bitcoin data (2009-2025).
RenTech principle: Find what works for LONGS and SHORTS separately.

Key Questions:
1. Which patterns work better for LONG vs SHORT?
2. Are there SHORT-specific patterns?
3. What's the optimal direction for each signal?

Usage:
    python -m engine.sovereign.backtest.rentech.run_bidirectional
"""

import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import numpy as np

from .data_loader import RentechDataLoader
from .feature_engine import FeatureEngine
from .advanced_features import AdvancedFeatureEngine
from .exhaustive_features import ExhaustiveFeatureEngine
from .walk_forward import WalkForwardEngine, WFConfig, Trade


class Direction(Enum):
    LONG = 1
    SHORT = -1


@dataclass
class BidirectionalStrategy:
    """Strategy that tests both LONG and SHORT."""
    name: str
    category: str
    description: str
    entry_column: str
    entry_op: str
    entry_threshold: float
    hold_days: int = 5

    def check_entry(self, row: pd.Series) -> bool:
        """Check entry condition."""
        if self.entry_column not in row.index:
            return False

        value = row[self.entry_column]
        if pd.isna(value):
            return False

        if self.entry_op == '>':
            return value > self.entry_threshold
        elif self.entry_op == '<':
            return value < self.entry_threshold
        elif self.entry_op == '>=':
            return value >= self.entry_threshold
        elif self.entry_op == '<=':
            return value <= self.entry_threshold
        elif self.entry_op == '==':
            return value == self.entry_threshold
        elif self.entry_op == 'between':
            return self.entry_threshold <= value <= self.entry_threshold_2

        return False


@dataclass
class BidirectionalResult:
    """Results for both LONG and SHORT."""
    strategy_name: str
    category: str
    description: str

    # LONG results
    long_trades: int
    long_wins: int
    long_win_rate: float
    long_pnl: float
    long_sharpe: float

    # SHORT results
    short_trades: int
    short_wins: int
    short_win_rate: float
    short_pnl: float
    short_sharpe: float

    # Comparison
    better_direction: str
    direction_edge: float  # How much better


def generate_bidirectional_strategies() -> List[BidirectionalStrategy]:
    """Generate strategies to test in both directions."""
    strategies = []

    # ========== RSI PATTERNS ==========
    for period in [7, 14, 21]:
        # Oversold (typically LONG)
        for thresh in [20, 25, 30, 35]:
            for hold in [1, 3, 5, 7]:
                strategies.append(BidirectionalStrategy(
                    name=f"RSI{period}_BELOW_{thresh}_{hold}d",
                    category="RSI",
                    description=f"RSI{period} < {thresh}",
                    entry_column=f"rsi_{period}",
                    entry_op="<",
                    entry_threshold=thresh,
                    hold_days=hold,
                ))

        # Overbought (typically SHORT)
        for thresh in [65, 70, 75, 80]:
            for hold in [1, 3, 5, 7]:
                strategies.append(BidirectionalStrategy(
                    name=f"RSI{period}_ABOVE_{thresh}_{hold}d",
                    category="RSI",
                    description=f"RSI{period} > {thresh}",
                    entry_column=f"rsi_{period}",
                    entry_op=">",
                    entry_threshold=thresh,
                    hold_days=hold,
                ))

    # ========== MACD PATTERNS ==========
    for hold in [1, 3, 5, 7]:
        # MACD histogram positive
        strategies.append(BidirectionalStrategy(
            name=f"MACD_HIST_POS_{hold}d",
            category="MACD",
            description="MACD histogram > 0",
            entry_column="macd_histogram",
            entry_op=">",
            entry_threshold=0,
            hold_days=hold,
        ))

        # MACD histogram negative
        strategies.append(BidirectionalStrategy(
            name=f"MACD_HIST_NEG_{hold}d",
            category="MACD",
            description="MACD histogram < 0",
            entry_column="macd_histogram",
            entry_op="<",
            entry_threshold=0,
            hold_days=hold,
        ))

    # MACD z-score extremes
    for thresh in [1.5, 2.0, 2.5]:
        for hold in [3, 5, 7]:
            strategies.append(BidirectionalStrategy(
                name=f"MACD_Z_HIGH_{thresh}_{hold}d",
                category="MACD",
                description=f"MACD zscore > {thresh}",
                entry_column="macd_zscore",
                entry_op=">",
                entry_threshold=thresh,
                hold_days=hold,
            ))

            strategies.append(BidirectionalStrategy(
                name=f"MACD_Z_LOW_{thresh}_{hold}d",
                category="MACD",
                description=f"MACD zscore < -{thresh}",
                entry_column="macd_zscore",
                entry_op="<",
                entry_threshold=-thresh,
                hold_days=hold,
            ))

    # ========== MOMENTUM / ROC ==========
    for period in [1, 3, 5, 10, 20]:
        for thresh in [2, 3, 5, 7, 10]:
            for hold in [1, 3, 5]:
                # Positive momentum
                strategies.append(BidirectionalStrategy(
                    name=f"ROC_{period}d_UP_{thresh}pct_{hold}d",
                    category="MOMENTUM",
                    description=f"{period}d return > {thresh}%",
                    entry_column=f"roc_{period}",
                    entry_op=">",
                    entry_threshold=thresh,
                    hold_days=hold,
                ))

                # Negative momentum
                strategies.append(BidirectionalStrategy(
                    name=f"ROC_{period}d_DN_{thresh}pct_{hold}d",
                    category="MOMENTUM",
                    description=f"{period}d return < -{thresh}%",
                    entry_column=f"roc_{period}",
                    entry_op="<",
                    entry_threshold=-thresh,
                    hold_days=hold,
                ))

    # ========== MEAN REVERSION - PRICE vs SMA ==========
    for deviation in [3, 5, 7, 10, 15]:
        for hold in [1, 3, 5, 7]:
            # Price below SMA
            strategies.append(BidirectionalStrategy(
                name=f"PRICE_BELOW_SMA20_{deviation}pct_{hold}d",
                category="MEAN_REVERSION",
                description=f"Price {deviation}% below SMA20",
                entry_column="price_vs_sma20",
                entry_op="<",
                entry_threshold=-deviation,
                hold_days=hold,
            ))

            # Price above SMA
            strategies.append(BidirectionalStrategy(
                name=f"PRICE_ABOVE_SMA20_{deviation}pct_{hold}d",
                category="MEAN_REVERSION",
                description=f"Price {deviation}% above SMA20",
                entry_column="price_vs_sma20",
                entry_op=">",
                entry_threshold=deviation,
                hold_days=hold,
            ))

    # ========== BOLLINGER BANDS ==========
    for hold in [1, 3, 5]:
        # Below lower band
        strategies.append(BidirectionalStrategy(
            name=f"BB_BELOW_LOWER_{hold}d",
            category="BOLLINGER",
            description="Price below lower BB",
            entry_column="bb_position",
            entry_op="<",
            entry_threshold=0,
            hold_days=hold,
        ))

        # Above upper band
        strategies.append(BidirectionalStrategy(
            name=f"BB_ABOVE_UPPER_{hold}d",
            category="BOLLINGER",
            description="Price above upper BB",
            entry_column="bb_position",
            entry_op=">",
            entry_threshold=1,
            hold_days=hold,
        ))

    # ========== VOLUME ==========
    for window in [5, 10, 20]:
        for thresh in [1.5, 2.0, 2.5]:
            for hold in [1, 3, 5]:
                # High volume
                strategies.append(BidirectionalStrategy(
                    name=f"VOL_{window}d_HIGH_{thresh}z_{hold}d",
                    category="VOLUME",
                    description=f"Volume z > {thresh}",
                    entry_column=f"volume_zscore_{window}",
                    entry_op=">",
                    entry_threshold=thresh,
                    hold_days=hold,
                ))

                # Low volume
                strategies.append(BidirectionalStrategy(
                    name=f"VOL_{window}d_LOW_{thresh}z_{hold}d",
                    category="VOLUME",
                    description=f"Volume z < -{thresh}",
                    entry_column=f"volume_zscore_{window}",
                    entry_op="<",
                    entry_threshold=-thresh,
                    hold_days=hold,
                ))

    # ========== ATR / VOLATILITY ==========
    for thresh in [1.0, 1.5, 2.0, 2.5]:
        for hold in [1, 3, 5]:
            # High volatility
            strategies.append(BidirectionalStrategy(
                name=f"ATR_HIGH_{thresh}z_{hold}d",
                category="VOLATILITY",
                description=f"ATR z > {thresh}",
                entry_column="atr_zscore",
                entry_op=">",
                entry_threshold=thresh,
                hold_days=hold,
            ))

            # Low volatility
            strategies.append(BidirectionalStrategy(
                name=f"ATR_LOW_{thresh}z_{hold}d",
                category="VOLATILITY",
                description=f"ATR z < -{thresh}",
                entry_column="atr_zscore",
                entry_op="<",
                entry_threshold=-thresh,
                hold_days=hold,
            ))

    # ========== TX Z-SCORE (BLOCKCHAIN) ==========
    for thresh in [1.0, 1.5, 2.0, 2.5, 3.0]:
        for hold in [1, 3, 5, 7, 10]:
            # High TX activity
            strategies.append(BidirectionalStrategy(
                name=f"TX_HIGH_{thresh}z_{hold}d",
                category="BLOCKCHAIN",
                description=f"TX zscore > {thresh}",
                entry_column="tx_count_zscore",
                entry_op=">",
                entry_threshold=thresh,
                hold_days=hold,
            ))

            # Low TX activity
            strategies.append(BidirectionalStrategy(
                name=f"TX_LOW_{thresh}z_{hold}d",
                category="BLOCKCHAIN",
                description=f"TX zscore < -{thresh}",
                entry_column="tx_count_zscore",
                entry_op="<",
                entry_threshold=-thresh,
                hold_days=hold,
            ))

    # ========== MA CROSS ==========
    for hold in [3, 5, 7, 10]:
        # Golden cross
        strategies.append(BidirectionalStrategy(
            name=f"MA_GOLDEN_{hold}d",
            category="MA_CROSS",
            description="SMA10 > SMA20",
            entry_column="ma_cross_10_20",
            entry_op="==",
            entry_threshold=1,
            hold_days=hold,
        ))

        # Death cross
        strategies.append(BidirectionalStrategy(
            name=f"MA_DEATH_{hold}d",
            category="MA_CROSS",
            description="SMA10 < SMA20",
            entry_column="ma_cross_10_20",
            entry_op="==",
            entry_threshold=0,
            hold_days=hold,
        ))

    # ========== SEASONALITY ==========
    dow_names = ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']
    for day in range(7):
        strategies.append(BidirectionalStrategy(
            name=f"DOW_{dow_names[day]}_1d",
            category="SEASONALITY",
            description=f"Day = {dow_names[day]}",
            entry_column="dow",
            entry_op="==",
            entry_threshold=day,
            hold_days=1,
        ))

    # Month effects
    for month in range(1, 13):
        for hold in [5, 10]:
            strategies.append(BidirectionalStrategy(
                name=f"MONTH_{month}_{hold}d",
                category="SEASONALITY",
                description=f"Month = {month}",
                entry_column="month",
                entry_op="==",
                entry_threshold=month,
                hold_days=hold,
            ))

    # Quarter effects
    for q in range(1, 5):
        strategies.append(BidirectionalStrategy(
            name=f"QUARTER_Q{q}_10d",
            category="SEASONALITY",
            description=f"Quarter = Q{q}",
            entry_column="quarter",
            entry_op="==",
            entry_threshold=q,
            hold_days=10,
        ))

    # ========== CONSECUTIVE DAYS ==========
    for streak in [2, 3, 4, 5]:
        for hold in [1, 3, 5]:
            # Consecutive up
            strategies.append(BidirectionalStrategy(
                name=f"STREAK_UP_{streak}d_{hold}d",
                category="MICROSTRUCTURE",
                description=f"{streak} consecutive up days",
                entry_column=f"consecutive_up_{streak}",
                entry_op="==",
                entry_threshold=1,
                hold_days=hold,
            ))

            # Consecutive down
            strategies.append(BidirectionalStrategy(
                name=f"STREAK_DN_{streak}d_{hold}d",
                category="MICROSTRUCTURE",
                description=f"{streak} consecutive down days",
                entry_column=f"consecutive_down_{streak}",
                entry_op="==",
                entry_threshold=1,
                hold_days=hold,
            ))

    # ========== NEW HIGHS/LOWS ==========
    for lookback in [20, 50, 100]:
        for hold in [3, 5, 10]:
            strategies.append(BidirectionalStrategy(
                name=f"NEW_HIGH_{lookback}d_{hold}d",
                category="EXTREME",
                description=f"New {lookback}-day high",
                entry_column=f"is_high_{lookback}",
                entry_op="==",
                entry_threshold=1,
                hold_days=hold,
            ))

            strategies.append(BidirectionalStrategy(
                name=f"NEW_LOW_{lookback}d_{hold}d",
                category="EXTREME",
                description=f"New {lookback}-day low",
                entry_column=f"is_low_{lookback}",
                entry_op="==",
                entry_threshold=1,
                hold_days=hold,
            ))

    return strategies


def backtest_bidirectional(
    strategy: BidirectionalStrategy,
    df: pd.DataFrame,
    wf_engine: WalkForwardEngine,
) -> BidirectionalResult:
    """Backtest strategy in BOTH directions."""

    # LONG adapter
    class LongAdapter:
        def __init__(self, s):
            self.strategy = s
            self.name = s.name + "_LONG"
            self.category = s.category
            self.description = s.description
            self.hold_days = s.hold_days

        def check_entry(self, row):
            return self.strategy.check_entry(row)

        def get_signal_direction(self):
            return 1  # LONG

    # SHORT adapter
    class ShortAdapter:
        def __init__(self, s):
            self.strategy = s
            self.name = s.name + "_SHORT"
            self.category = s.category
            self.description = s.description
            self.hold_days = s.hold_days

        def check_entry(self, row):
            return self.strategy.check_entry(row)

        def get_signal_direction(self):
            return -1  # SHORT

    # Run LONG
    long_adapter = LongAdapter(strategy)
    long_trades = wf_engine.get_all_test_trades(long_adapter, df)

    # Run SHORT
    short_adapter = ShortAdapter(strategy)
    short_trades = wf_engine.get_all_test_trades(short_adapter, df)

    # Calculate LONG stats
    long_wins = sum(1 for t in long_trades if t.pnl_pct > 0)
    long_win_rate = long_wins / len(long_trades) if long_trades else 0
    long_pnl = sum(t.pnl_pct for t in long_trades)
    long_returns = [t.pnl_pct for t in long_trades]
    long_sharpe = (np.mean(long_returns) / np.std(long_returns) * np.sqrt(252)) if long_returns and np.std(long_returns) > 0 else 0

    # Calculate SHORT stats
    short_wins = sum(1 for t in short_trades if t.pnl_pct > 0)
    short_win_rate = short_wins / len(short_trades) if short_trades else 0
    short_pnl = sum(t.pnl_pct for t in short_trades)
    short_returns = [t.pnl_pct for t in short_trades]
    short_sharpe = (np.mean(short_returns) / np.std(short_returns) * np.sqrt(252)) if short_returns and np.std(short_returns) > 0 else 0

    # Determine better direction
    if long_sharpe > short_sharpe:
        better = "LONG"
        edge = long_sharpe - short_sharpe
    elif short_sharpe > long_sharpe:
        better = "SHORT"
        edge = short_sharpe - long_sharpe
    else:
        better = "NEUTRAL"
        edge = 0

    return BidirectionalResult(
        strategy_name=strategy.name,
        category=strategy.category,
        description=strategy.description,
        long_trades=len(long_trades),
        long_wins=long_wins,
        long_win_rate=long_win_rate,
        long_pnl=long_pnl,
        long_sharpe=long_sharpe,
        short_trades=len(short_trades),
        short_wins=short_wins,
        short_win_rate=short_win_rate,
        short_pnl=short_pnl,
        short_sharpe=short_sharpe,
        better_direction=better,
        direction_edge=edge,
    )


def main():
    print("=" * 90)
    print("BIDIRECTIONAL RENTECH BACKTEST - LONG vs SHORT")
    print("=" * 90)
    print("\nTesting EVERY pattern in BOTH directions on Bitcoin 2009-2025")

    start_time = time.time()

    # Load data
    print("\n[1/5] Loading data...")
    loader = RentechDataLoader()
    summary = loader.get_summary()
    print(f"  Data: {summary['total_start']} to {summary['total_end']} ({summary['total_days']:,} days)")

    df = loader.load_merged_data()

    # Calculate features
    print("\n[2/5] Calculating features...")
    df = FeatureEngine().add_all_features(df)
    df = AdvancedFeatureEngine().add_all_features(df)
    df = ExhaustiveFeatureEngine().add_all_features(df)
    print(f"  Total columns: {len(df.columns)}")

    # Generate strategies
    print("\n[3/5] Generating bidirectional strategies...")
    strategies = generate_bidirectional_strategies()
    print(f"  Strategies to test: {len(strategies)}")
    print(f"  Each tested as LONG and SHORT = {len(strategies) * 2} total tests")

    # Run backtest
    print("\n[4/5] Running bidirectional backtest...")
    wf_config = WFConfig(train_years=2, test_months=6, step_months=6)
    wf_engine = WalkForwardEngine(wf_config)

    results = []
    for i, strategy in enumerate(strategies):
        if (i + 1) % 50 == 0:
            print(f"    Progress: {i+1}/{len(strategies)} ({(i+1)/len(strategies)*100:.1f}%)")

        result = backtest_bidirectional(strategy, df, wf_engine)
        results.append(result)

    # Export and analyze
    print("\n[5/5] Analyzing results...")

    # Convert to exportable format
    output = {
        'run_date': datetime.now().isoformat(),
        'data_range': f"{summary['total_start']} to {summary['total_end']}",
        'total_days': summary['total_days'],
        'strategies_tested': len(strategies),
        'results': []
    }

    for r in results:
        output['results'].append({
            'name': r.strategy_name,
            'category': r.category,
            'description': r.description,
            'long_trades': r.long_trades,
            'long_win_rate': round(r.long_win_rate, 4),
            'long_pnl': round(r.long_pnl, 2),
            'long_sharpe': round(r.long_sharpe, 2),
            'short_trades': r.short_trades,
            'short_win_rate': round(r.short_win_rate, 4),
            'short_pnl': round(r.short_pnl, 2),
            'short_sharpe': round(r.short_sharpe, 2),
            'better_direction': r.better_direction,
            'direction_edge': round(r.direction_edge, 2),
        })

    Path('data').mkdir(exist_ok=True)
    with open('data/bidirectional_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    # Print analysis
    print("\n" + "=" * 90)
    print("BIDIRECTIONAL ANALYSIS RESULTS")
    print("=" * 90)

    # Summary stats
    long_better = [r for r in results if r.better_direction == "LONG" and r.long_trades >= 100]
    short_better = [r for r in results if r.better_direction == "SHORT" and r.short_trades >= 100]

    print(f"\nStrategies where LONG is better: {len(long_better)}")
    print(f"Strategies where SHORT is better: {len(short_better)}")

    # Best LONG strategies
    print("\n" + "=" * 90)
    print("TOP 30 LONG STRATEGIES (by Sharpe, 100+ trades)")
    print("=" * 90)

    long_sorted = sorted([r for r in results if r.long_trades >= 100],
                        key=lambda x: x.long_sharpe, reverse=True)

    print(f"\n{'Strategy':<45} {'Cat':<12} {'WR':>6} {'#':>5} {'PnL':>8} {'Sharpe':>7}")
    print("-" * 90)

    for r in long_sorted[:30]:
        print(f"{r.strategy_name:<45} {r.category:<12} {r.long_win_rate*100:>5.1f}% "
              f"{r.long_trades:>5} {r.long_pnl:>7.0f}% {r.long_sharpe:>7.2f}")

    # Best SHORT strategies
    print("\n" + "=" * 90)
    print("TOP 30 SHORT STRATEGIES (by Sharpe, 100+ trades)")
    print("=" * 90)

    short_sorted = sorted([r for r in results if r.short_trades >= 100],
                         key=lambda x: x.short_sharpe, reverse=True)

    print(f"\n{'Strategy':<45} {'Cat':<12} {'WR':>6} {'#':>5} {'PnL':>8} {'Sharpe':>7}")
    print("-" * 90)

    for r in short_sorted[:30]:
        print(f"{r.strategy_name:<45} {r.category:<12} {r.short_win_rate*100:>5.1f}% "
              f"{r.short_trades:>5} {r.short_pnl:>7.0f}% {r.short_sharpe:>7.2f}")

    # Strategies where SHORT works (WR > 50%)
    print("\n" + "=" * 90)
    print("SHORT STRATEGIES WITH EDGE (WR > 50%, 100+ trades)")
    print("=" * 90)

    short_edge = [r for r in results if r.short_win_rate > 0.50 and r.short_trades >= 100]
    short_edge.sort(key=lambda x: x.short_win_rate, reverse=True)

    print(f"\n{'Strategy':<45} {'Cat':<12} {'WR':>6} {'#':>5} {'PnL':>8} {'Sharpe':>7}")
    print("-" * 90)

    for r in short_edge[:30]:
        print(f"{r.strategy_name:<45} {r.category:<12} {r.short_win_rate*100:>5.1f}% "
              f"{r.short_trades:>5} {r.short_pnl:>7.0f}% {r.short_sharpe:>7.2f}")

    # LONG vs SHORT by Category
    print("\n" + "=" * 90)
    print("LONG vs SHORT BY CATEGORY")
    print("=" * 90)

    categories = {}
    for r in results:
        if r.category not in categories:
            categories[r.category] = {
                'long_wins': 0, 'short_wins': 0,
                'long_total': 0, 'short_total': 0,
                'long_avg_wr': [], 'short_avg_wr': []
            }
        if r.long_trades >= 50:
            categories[r.category]['long_total'] += 1
            categories[r.category]['long_avg_wr'].append(r.long_win_rate)
            if r.long_win_rate > 0.5:
                categories[r.category]['long_wins'] += 1
        if r.short_trades >= 50:
            categories[r.category]['short_total'] += 1
            categories[r.category]['short_avg_wr'].append(r.short_win_rate)
            if r.short_win_rate > 0.5:
                categories[r.category]['short_wins'] += 1

    print(f"\n{'Category':<15} {'LONG Avg WR':>12} {'LONG Edge':>12} {'SHORT Avg WR':>13} {'SHORT Edge':>12}")
    print("-" * 65)

    for cat, info in sorted(categories.items()):
        long_avg = np.mean(info['long_avg_wr']) * 100 if info['long_avg_wr'] else 0
        short_avg = np.mean(info['short_avg_wr']) * 100 if info['short_avg_wr'] else 0
        long_edge = f"{info['long_wins']}/{info['long_total']}"
        short_edge = f"{info['short_wins']}/{info['short_total']}"

        print(f"{cat:<15} {long_avg:>11.1f}% {long_edge:>12} {short_avg:>12.1f}% {short_edge:>12}")

    elapsed = time.time() - start_time
    print(f"\n{'='*90}")
    print(f"Completed in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"Results saved to: data/bidirectional_results.json")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
