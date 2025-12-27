"""
Strategy Factory - Generate trading strategy configurations

Creates testable strategy definitions with parameter grids for:
- Z-score mean reversion/momentum
- Whale activity patterns
- Block fullness signals
- Seasonality effects
- Regime-based trading
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Callable, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class Direction(Enum):
    LONG = 1
    SHORT = -1
    BOTH = 0


@dataclass
class Strategy:
    """Trading strategy definition."""
    name: str
    category: str
    description: str
    direction: Direction

    # Entry condition
    entry_column: str
    entry_op: str  # '>', '<', '==', 'between'
    entry_threshold: float
    entry_threshold_2: Optional[float] = None  # For 'between'

    # Exit condition
    hold_days: int = 5
    stop_loss: float = -0.10  # -10%
    take_profit: float = 0.20  # +20%

    # Additional filters
    min_price: float = 0  # Minimum BTC price to trade
    regime_filter: Optional[str] = None  # Only trade in this regime

    def check_entry(self, row: pd.Series) -> bool:
        """Check if entry condition is met."""
        if self.entry_column not in row.index:
            return False

        value = row[self.entry_column]
        if pd.isna(value):
            return False

        # Check regime filter
        if self.regime_filter and row.get('regime') != self.regime_filter:
            return False

        # Check min price
        if self.min_price > 0 and row.get('close', 0) < self.min_price:
            return False

        # Check entry condition
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

    def get_signal_direction(self) -> int:
        """Get trade direction as int."""
        if self.direction == Direction.LONG:
            return 1
        elif self.direction == Direction.SHORT:
            return -1
        return 0


class StrategyFactory:
    """
    Generate strategy configurations for backtesting.

    Categories:
    1. TX_ZSCORE: Mean reversion on transaction count z-scores
    2. WHALE: Follow or fade whale activity
    3. FULLNESS: Block fullness congestion signals
    4. VALUE: Large value flow detection
    5. SEASONALITY: Day-of-week and month effects
    6. REGIME: Trade based on HMM regime
    """

    def generate_all(self) -> List[Strategy]:
        """Generate all strategy variations."""
        strategies = []

        strategies.extend(self._generate_tx_zscore_strategies())
        strategies.extend(self._generate_whale_strategies())
        strategies.extend(self._generate_fullness_strategies())
        strategies.extend(self._generate_value_strategies())
        strategies.extend(self._generate_seasonality_strategies())
        strategies.extend(self._generate_regime_strategies())

        return strategies

    def _generate_tx_zscore_strategies(self) -> List[Strategy]:
        """TX count z-score mean reversion and momentum strategies."""
        strategies = []

        # Mean reversion: High z-score -> expect reversal
        for threshold in [1.5, 2.0, 2.5, 3.0]:
            for hold in [1, 3, 5, 7, 10]:
                # High TX -> SHORT (mean reversion)
                strategies.append(Strategy(
                    name=f"TX_MR_HIGH_{threshold}_{hold}d_SHORT",
                    category="TX_ZSCORE",
                    description=f"Short when TX z-score > {threshold}, hold {hold}d",
                    direction=Direction.SHORT,
                    entry_column="tx_count_zscore",
                    entry_op=">",
                    entry_threshold=threshold,
                    hold_days=hold,
                ))

                # Low TX -> LONG (mean reversion)
                strategies.append(Strategy(
                    name=f"TX_MR_LOW_{threshold}_{hold}d_LONG",
                    category="TX_ZSCORE",
                    description=f"Long when TX z-score < -{threshold}, hold {hold}d",
                    direction=Direction.LONG,
                    entry_column="tx_count_zscore",
                    entry_op="<",
                    entry_threshold=-threshold,
                    hold_days=hold,
                ))

        # Momentum: High z-score -> continue trend
        for threshold in [1.5, 2.0]:
            for hold in [3, 5, 7]:
                # High TX -> LONG (momentum)
                strategies.append(Strategy(
                    name=f"TX_MOM_HIGH_{threshold}_{hold}d_LONG",
                    category="TX_ZSCORE",
                    description=f"Long when TX z-score > {threshold} (momentum), hold {hold}d",
                    direction=Direction.LONG,
                    entry_column="tx_count_zscore",
                    entry_op=">",
                    entry_threshold=threshold,
                    hold_days=hold,
                ))

        return strategies

    def _generate_whale_strategies(self) -> List[Strategy]:
        """Whale activity strategies (ORBITAAL data only)."""
        strategies = []

        for threshold in [1.0, 1.5, 2.0, 2.5]:
            for hold in [1, 3, 5, 7]:
                # High whale activity -> LONG (follow smart money)
                strategies.append(Strategy(
                    name=f"WHALE_FOLLOW_{threshold}_{hold}d_LONG",
                    category="WHALE",
                    description=f"Long when whale z-score > {threshold}, hold {hold}d",
                    direction=Direction.LONG,
                    entry_column="whale_tx_count_zscore",
                    entry_op=">",
                    entry_threshold=threshold,
                    hold_days=hold,
                ))

                # High whale activity -> SHORT (fade, distribution)
                strategies.append(Strategy(
                    name=f"WHALE_FADE_{threshold}_{hold}d_SHORT",
                    category="WHALE",
                    description=f"Short when whale z-score > {threshold} (fade), hold {hold}d",
                    direction=Direction.SHORT,
                    entry_column="whale_tx_count_zscore",
                    entry_op=">",
                    entry_threshold=threshold,
                    hold_days=hold,
                ))

        return strategies

    def _generate_fullness_strategies(self) -> List[Strategy]:
        """Block fullness strategies (downloaded data only)."""
        strategies = []

        for threshold in [1.5, 2.0, 2.5]:
            for hold in [1, 3, 5]:
                # High fullness (congestion) -> SHORT
                strategies.append(Strategy(
                    name=f"FULLNESS_HIGH_{threshold}_{hold}d_SHORT",
                    category="FULLNESS",
                    description=f"Short when fullness z-score > {threshold}, hold {hold}d",
                    direction=Direction.SHORT,
                    entry_column="avg_block_fullness_zscore",
                    entry_op=">",
                    entry_threshold=threshold,
                    hold_days=hold,
                ))

                # Low fullness -> LONG
                strategies.append(Strategy(
                    name=f"FULLNESS_LOW_{threshold}_{hold}d_LONG",
                    category="FULLNESS",
                    description=f"Long when fullness z-score < -{threshold}, hold {hold}d",
                    direction=Direction.LONG,
                    entry_column="avg_block_fullness_zscore",
                    entry_op="<",
                    entry_threshold=-threshold,
                    hold_days=hold,
                ))

        return strategies

    def _generate_value_strategies(self) -> List[Strategy]:
        """Total value flow strategies (ORBITAAL data only)."""
        strategies = []

        for threshold in [1.5, 2.0, 2.5]:
            for hold in [1, 3, 5, 7]:
                # High value flow -> LONG (big money moving)
                strategies.append(Strategy(
                    name=f"VALUE_HIGH_{threshold}_{hold}d_LONG",
                    category="VALUE",
                    description=f"Long when value z-score > {threshold}, hold {hold}d",
                    direction=Direction.LONG,
                    entry_column="total_value_btc_zscore",
                    entry_op=">",
                    entry_threshold=threshold,
                    hold_days=hold,
                ))

                # Low value flow -> SHORT
                strategies.append(Strategy(
                    name=f"VALUE_LOW_{threshold}_{hold}d_SHORT",
                    category="VALUE",
                    description=f"Short when value z-score < -{threshold}, hold {hold}d",
                    direction=Direction.SHORT,
                    entry_column="total_value_btc_zscore",
                    entry_op="<",
                    entry_threshold=-threshold,
                    hold_days=hold,
                ))

        return strategies

    def _generate_seasonality_strategies(self) -> List[Strategy]:
        """Day-of-week and month seasonality strategies."""
        strategies = []

        # Day of week (0=Monday, 6=Sunday)
        day_names = ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']

        for day in range(7):
            for direction in [Direction.LONG, Direction.SHORT]:
                dir_str = 'LONG' if direction == Direction.LONG else 'SHORT'
                strategies.append(Strategy(
                    name=f"DOW_{day_names[day]}_{dir_str}",
                    category="SEASONALITY",
                    description=f"{dir_str} on {day_names[day]}s, hold 1d",
                    direction=direction,
                    entry_column="day_of_week",
                    entry_op="==",
                    entry_threshold=day,
                    hold_days=1,
                ))

        # Month effects
        month_names = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
                       'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

        for month in range(1, 13):
            for direction in [Direction.LONG, Direction.SHORT]:
                dir_str = 'LONG' if direction == Direction.LONG else 'SHORT'
                strategies.append(Strategy(
                    name=f"MONTH_{month_names[month-1]}_{dir_str}",
                    category="SEASONALITY",
                    description=f"{dir_str} in {month_names[month-1]}, hold 5d",
                    direction=direction,
                    entry_column="month",
                    entry_op="==",
                    entry_threshold=month,
                    hold_days=5,
                ))

        return strategies

    def _generate_regime_strategies(self) -> List[Strategy]:
        """Regime-based strategies."""
        strategies = []

        regime_directions = {
            'ACCUMULATION': Direction.LONG,
            'DISTRIBUTION': Direction.SHORT,
            'CAPITULATION': Direction.LONG,  # Contrarian
            'EUPHORIA': Direction.SHORT,  # Contrarian
        }

        for regime, direction in regime_directions.items():
            for hold in [1, 3, 5, 7]:
                dir_str = 'LONG' if direction == Direction.LONG else 'SHORT'
                strategies.append(Strategy(
                    name=f"REGIME_{regime}_{hold}d_{dir_str}",
                    category="REGIME",
                    description=f"{dir_str} in {regime} regime, hold {hold}d",
                    direction=direction,
                    entry_column="regime",
                    entry_op="==",
                    entry_threshold=regime,
                    hold_days=hold,
                ))

        return strategies

    def get_strategy_count(self) -> Dict[str, int]:
        """Get count of strategies by category."""
        strategies = self.generate_all()
        counts = {}
        for s in strategies:
            counts[s.category] = counts.get(s.category, 0) + 1
        return counts


def quick_test():
    """Quick test of strategy factory."""
    factory = StrategyFactory()
    strategies = factory.generate_all()

    print(f"Total strategies: {len(strategies)}")
    print("\nBy category:")
    for cat, count in factory.get_strategy_count().items():
        print(f"  {cat}: {count}")

    print("\nSample strategies:")
    for s in strategies[:5]:
        print(f"  {s.name}: {s.description}")


if __name__ == "__main__":
    quick_test()
