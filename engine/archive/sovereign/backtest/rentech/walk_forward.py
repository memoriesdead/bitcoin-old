"""
Walk-Forward Validation Engine

Implements proper out-of-sample testing to avoid look-ahead bias:
1. Split data into sequential windows
2. Train (optimize) on window i
3. Test on window i+1
4. Aggregate results across all windows

This simulates real trading: always testing on future data.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from .strategy_factory import Strategy, Direction


@dataclass
class WFConfig:
    """Walk-forward configuration."""
    train_years: int = 2        # Years of training data
    test_months: int = 6        # Months of test data
    step_months: int = 6        # Step forward by this many months
    min_train_trades: int = 50  # Minimum trades in training window
    min_test_trades: int = 20   # Minimum trades in test window


@dataclass
class Trade:
    """Single trade result."""
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    direction: int  # 1=long, -1=short
    pnl_pct: float
    hold_days: int


@dataclass
class WindowResult:
    """Results for a single walk-forward window."""
    window_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str

    # Training results
    train_trades: int
    train_wins: int
    train_win_rate: float
    train_pnl: float

    # Test results (out-of-sample)
    test_trades: int
    test_wins: int
    test_win_rate: float
    test_pnl: float

    # Degradation
    degradation: float  # (train_wr - test_wr) / train_wr


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward results."""
    strategy_name: str
    windows: List[WindowResult]

    # Aggregate metrics
    total_train_trades: int = 0
    total_test_trades: int = 0
    avg_train_win_rate: float = 0.0
    avg_test_win_rate: float = 0.0
    total_test_pnl: float = 0.0
    avg_degradation: float = 0.0
    win_rate_std: float = 0.0  # Consistency across windows

    def __post_init__(self):
        if self.windows:
            self._calculate_aggregates()

    def _calculate_aggregates(self):
        """Calculate aggregate statistics from windows."""
        self.total_train_trades = sum(w.train_trades for w in self.windows)
        self.total_test_trades = sum(w.test_trades for w in self.windows)

        if self.windows:
            train_wrs = [w.train_win_rate for w in self.windows if w.train_trades > 0]
            test_wrs = [w.test_win_rate for w in self.windows if w.test_trades > 0]

            self.avg_train_win_rate = np.mean(train_wrs) if train_wrs else 0
            self.avg_test_win_rate = np.mean(test_wrs) if test_wrs else 0
            self.win_rate_std = np.std(test_wrs) if test_wrs else 0

            self.total_test_pnl = sum(w.test_pnl for w in self.windows)

            degradations = [w.degradation for w in self.windows
                           if w.train_trades > 0 and w.test_trades > 0]
            self.avg_degradation = np.mean(degradations) if degradations else 0


class WalkForwardEngine:
    """
    Walk-forward validation for trading strategies.

    Process:
    1. Generate sequential train/test windows
    2. Run strategy on each window
    3. Track in-sample (train) vs out-of-sample (test) performance
    4. Identify overfitting via degradation metrics
    """

    def __init__(self, config: WFConfig = None):
        self.config = config or WFConfig()

    def generate_windows(
        self,
        start_date: str,
        end_date: str
    ) -> List[Tuple[str, str, str, str]]:
        """
        Generate train/test window boundaries.

        Returns list of (train_start, train_end, test_start, test_end) tuples.
        """
        windows = []

        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')

        train_days = self.config.train_years * 365
        test_days = self.config.test_months * 30
        step_days = self.config.step_months * 30

        current_train_start = start

        while True:
            train_end = current_train_start + timedelta(days=train_days)
            test_start = train_end + timedelta(days=1)
            test_end = test_start + timedelta(days=test_days)

            if test_end > end:
                break

            windows.append((
                current_train_start.strftime('%Y-%m-%d'),
                train_end.strftime('%Y-%m-%d'),
                test_start.strftime('%Y-%m-%d'),
                test_end.strftime('%Y-%m-%d'),
            ))

            current_train_start += timedelta(days=step_days)

        return windows

    def run_strategy_on_window(
        self,
        strategy: Strategy,
        df: pd.DataFrame,
        train_start: str,
        train_end: str,
        test_start: str,
        test_end: str,
        window_id: int
    ) -> WindowResult:
        """Run strategy on a single train/test window."""

        # Split data
        train_df = df[(df['date'] >= train_start) & (df['date'] <= train_end)].copy()
        test_df = df[(df['date'] >= test_start) & (df['date'] <= test_end)].copy()

        # Run on train
        train_trades = self._backtest_strategy(strategy, train_df)
        train_wins = sum(1 for t in train_trades if t.pnl_pct > 0)
        train_win_rate = train_wins / len(train_trades) if train_trades else 0
        train_pnl = sum(t.pnl_pct for t in train_trades)

        # Run on test
        test_trades = self._backtest_strategy(strategy, test_df)
        test_wins = sum(1 for t in test_trades if t.pnl_pct > 0)
        test_win_rate = test_wins / len(test_trades) if test_trades else 0
        test_pnl = sum(t.pnl_pct for t in test_trades)

        # Calculate degradation
        if train_win_rate > 0:
            degradation = (train_win_rate - test_win_rate) / train_win_rate
        else:
            degradation = 0

        return WindowResult(
            window_id=window_id,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            train_trades=len(train_trades),
            train_wins=train_wins,
            train_win_rate=train_win_rate,
            train_pnl=train_pnl,
            test_trades=len(test_trades),
            test_wins=test_wins,
            test_win_rate=test_win_rate,
            test_pnl=test_pnl,
            degradation=degradation,
        )

    def _backtest_strategy(
        self,
        strategy: Strategy,
        df: pd.DataFrame
    ) -> List[Trade]:
        """
        Backtest strategy on data.

        Simple implementation:
        1. Check entry condition each day
        2. If triggered, enter trade
        3. Exit after hold_days or stop/take profit
        """
        trades = []
        in_position = False
        position_entry_idx = None

        df = df.reset_index(drop=True)

        for i, row in df.iterrows():
            # Check exit if in position
            if in_position:
                days_held = i - position_entry_idx

                if days_held >= strategy.hold_days:
                    # Time-based exit
                    exit_price = row['close']
                    entry_price = df.loc[position_entry_idx, 'close']

                    # Skip trade if price data missing
                    if pd.isna(exit_price) or pd.isna(entry_price):
                        in_position = False
                        position_entry_idx = None
                        continue

                    direction = strategy.get_signal_direction()

                    if direction == 1:  # Long
                        pnl_pct = (exit_price / entry_price - 1) * 100
                    else:  # Short
                        pnl_pct = (entry_price / exit_price - 1) * 100

                    trades.append(Trade(
                        entry_date=df.loc[position_entry_idx, 'date'],
                        exit_date=row['date'],
                        entry_price=entry_price,
                        exit_price=exit_price,
                        direction=direction,
                        pnl_pct=pnl_pct,
                        hold_days=days_held,
                    ))

                    in_position = False
                    position_entry_idx = None

            # Check entry if not in position
            if not in_position:
                # Only enter if price data is available
                if pd.notna(row.get('close')) and strategy.check_entry(row):
                    # Enter position
                    in_position = True
                    position_entry_idx = i

        return trades

    def run_full_walkforward(
        self,
        strategy: Strategy,
        df: pd.DataFrame
    ) -> WalkForwardResult:
        """Run complete walk-forward validation."""

        # Get date range
        start_date = df['date'].min()
        end_date = df['date'].max()

        # Generate windows
        windows = self.generate_windows(start_date, end_date)

        if not windows:
            return WalkForwardResult(
                strategy_name=strategy.name,
                windows=[],
            )

        # Run on each window
        window_results = []
        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            result = self.run_strategy_on_window(
                strategy, df,
                train_start, train_end,
                test_start, test_end,
                window_id=i,
            )
            window_results.append(result)

        return WalkForwardResult(
            strategy_name=strategy.name,
            windows=window_results,
        )

    def get_all_test_trades(
        self,
        strategy: Strategy,
        df: pd.DataFrame
    ) -> List[Trade]:
        """Get all out-of-sample trades across all windows."""
        result = self.run_full_walkforward(strategy, df)

        # Re-run to collect trades (simplified approach)
        all_trades = []

        start_date = df['date'].min()
        end_date = df['date'].max()
        windows = self.generate_windows(start_date, end_date)

        for train_start, train_end, test_start, test_end in windows:
            test_df = df[(df['date'] >= test_start) & (df['date'] <= test_end)].copy()
            trades = self._backtest_strategy(strategy, test_df)
            all_trades.extend(trades)

        return all_trades


def quick_test():
    """Quick test of walk-forward engine."""
    from .data_loader import RentechDataLoader
    from .feature_engine import FeatureEngine
    from .strategy_factory import StrategyFactory

    print("Testing Walk-Forward Engine...")

    # Load and prepare data
    loader = RentechDataLoader()
    df = loader.load_merged_data()

    engine = FeatureEngine()
    df = engine.add_all_features(df)

    # Get a sample strategy
    factory = StrategyFactory()
    strategies = factory.generate_all()
    strategy = strategies[0]  # First strategy

    print(f"\nTesting strategy: {strategy.name}")
    print(f"  {strategy.description}")

    # Run walk-forward
    wf = WalkForwardEngine()
    result = wf.run_full_walkforward(strategy, df)

    print(f"\nWalk-Forward Results:")
    print(f"  Windows: {len(result.windows)}")
    print(f"  Total train trades: {result.total_train_trades:,}")
    print(f"  Total test trades: {result.total_test_trades:,}")
    print(f"  Avg train win rate: {result.avg_train_win_rate*100:.1f}%")
    print(f"  Avg test win rate: {result.avg_test_win_rate*100:.1f}%")
    print(f"  Win rate std: {result.win_rate_std*100:.1f}%")
    print(f"  Avg degradation: {result.avg_degradation*100:.1f}%")
    print(f"  Total test PnL: {result.total_test_pnl:.1f}%")

    # Show window details
    print("\nWindow Details:")
    for w in result.windows[:5]:
        print(f"  W{w.window_id}: Train {w.train_start}-{w.train_end}, "
              f"Test {w.test_start}-{w.test_end}")
        print(f"       Train: {w.train_trades} trades, {w.train_win_rate*100:.1f}% WR")
        print(f"       Test:  {w.test_trades} trades, {w.test_win_rate*100:.1f}% WR")


if __name__ == "__main__":
    quick_test()
