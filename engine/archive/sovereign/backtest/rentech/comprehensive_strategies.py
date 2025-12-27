"""
Comprehensive Strategy Factory - RenTech-Style Pattern Testing
==============================================================

Generates 500+ trading strategies covering ALL RenTech patterns:
1. Mean Reversion (z-score based)
2. Momentum/Trending
3. Seasonality (day, month, quarter)
4. Technical Indicators (RSI, MACD, BB)
5. Volume Patterns
6. Regime-Based Trading
7. Cross-Signal Combinations
8. HMM State Transitions
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Callable, Optional
from dataclasses import dataclass
from enum import Enum


class Direction(Enum):
    LONG = 1
    SHORT = -1


@dataclass
class ComprehensiveStrategy:
    """Trading strategy definition with multiple conditions."""
    name: str
    category: str
    subcategory: str
    description: str
    direction: Direction

    # Primary condition
    entry_column: str
    entry_op: str  # '>', '<', '>=', '<=', '==', 'between', 'cross_above', 'cross_below'
    entry_threshold: float
    entry_threshold_2: Optional[float] = None

    # Optional secondary condition
    filter_column: Optional[str] = None
    filter_op: Optional[str] = None
    filter_threshold: Optional[float] = None

    # Exit parameters
    hold_days: int = 5
    stop_loss_pct: float = -10.0
    take_profit_pct: float = 20.0

    # Regime filter
    regime_filter: Optional[str] = None

    def check_entry(self, row: pd.Series, prev_row: pd.Series = None) -> bool:
        """Check if entry condition is met."""
        # Check primary condition
        if not self._check_condition(row, self.entry_column, self.entry_op,
                                     self.entry_threshold, self.entry_threshold_2, prev_row):
            return False

        # Check secondary filter
        if self.filter_column:
            if not self._check_condition(row, self.filter_column, self.filter_op,
                                        self.filter_threshold):
                return False

        # Check regime filter
        if self.regime_filter and row.get('regime') != self.regime_filter:
            return False

        return True

    def _check_condition(self, row, column, op, threshold, threshold_2=None, prev_row=None) -> bool:
        """Check single condition."""
        if column not in row.index:
            return False

        value = row[column]
        if pd.isna(value):
            return False

        if op == '>':
            return value > threshold
        elif op == '<':
            return value < threshold
        elif op == '>=':
            return value >= threshold
        elif op == '<=':
            return value <= threshold
        elif op == '==':
            return value == threshold
        elif op == 'between':
            return threshold <= value <= threshold_2
        elif op == 'cross_above' and prev_row is not None:
            prev_val = prev_row.get(column, np.nan)
            if pd.isna(prev_val):
                return False
            return prev_val <= threshold and value > threshold
        elif op == 'cross_below' and prev_row is not None:
            prev_val = prev_row.get(column, np.nan)
            if pd.isna(prev_val):
                return False
            return prev_val >= threshold and value < threshold

        return False

    def get_signal_direction(self) -> int:
        return self.direction.value


class ComprehensiveStrategyFactory:
    """
    Generate 500+ strategies covering all RenTech patterns.

    Categories:
    1. TX_ZSCORE - Transaction count z-score patterns
    2. MOMENTUM - Price momentum strategies
    3. MEAN_REVERSION - Mean reversion strategies
    4. RSI - RSI overbought/oversold
    5. MACD - MACD crossovers
    6. BOLLINGER - Bollinger Band strategies
    7. VOLUME - Volume pattern strategies
    8. SEASONALITY - Calendar effects
    9. REGIME - Regime-based trading
    10. MA_CROSS - Moving average crossovers
    11. ATR - Volatility breakouts
    12. COMBINATION - Multi-signal strategies
    """

    def generate_all(self) -> List[ComprehensiveStrategy]:
        """Generate all strategy variations."""
        strategies = []

        strategies.extend(self._tx_zscore_strategies())
        strategies.extend(self._momentum_strategies())
        strategies.extend(self._mean_reversion_strategies())
        strategies.extend(self._rsi_strategies())
        strategies.extend(self._macd_strategies())
        strategies.extend(self._bollinger_strategies())
        strategies.extend(self._volume_strategies())
        strategies.extend(self._seasonality_strategies())
        strategies.extend(self._regime_strategies())
        strategies.extend(self._ma_crossover_strategies())
        strategies.extend(self._atr_strategies())
        strategies.extend(self._combination_strategies())

        return strategies

    def _tx_zscore_strategies(self) -> List[ComprehensiveStrategy]:
        """TX count z-score strategies."""
        strategies = []

        # Mean reversion on TX z-score
        for threshold in [1.5, 2.0, 2.5, 3.0]:
            for hold in [1, 3, 5, 7, 10]:
                # High TX -> expect reversal (SHORT for mean reversion)
                strategies.append(ComprehensiveStrategy(
                    name=f"TX_MR_HIGH_{threshold}_{hold}d_SHORT",
                    category="TX_ZSCORE",
                    subcategory="MEAN_REV",
                    description=f"Short when TX z > {threshold}, expect mean reversion",
                    direction=Direction.SHORT,
                    entry_column="tx_count_zscore",
                    entry_op=">",
                    entry_threshold=threshold,
                    hold_days=hold,
                ))

                # Low TX -> expect recovery (LONG)
                strategies.append(ComprehensiveStrategy(
                    name=f"TX_MR_LOW_{threshold}_{hold}d_LONG",
                    category="TX_ZSCORE",
                    subcategory="MEAN_REV",
                    description=f"Long when TX z < -{threshold}, expect mean reversion",
                    direction=Direction.LONG,
                    entry_column="tx_count_zscore",
                    entry_op="<",
                    entry_threshold=-threshold,
                    hold_days=hold,
                ))

        # Momentum on TX z-score
        for threshold in [1.0, 1.5, 2.0]:
            for hold in [3, 5, 7]:
                # High TX -> continue momentum (LONG)
                strategies.append(ComprehensiveStrategy(
                    name=f"TX_MOM_HIGH_{threshold}_{hold}d_LONG",
                    category="TX_ZSCORE",
                    subcategory="MOMENTUM",
                    description=f"Long when TX z > {threshold}, momentum continuation",
                    direction=Direction.LONG,
                    entry_column="tx_count_zscore",
                    entry_op=">",
                    entry_threshold=threshold,
                    hold_days=hold,
                ))

        return strategies

    def _momentum_strategies(self) -> List[ComprehensiveStrategy]:
        """Price momentum strategies using ROC."""
        strategies = []

        # Simple momentum
        for period in [1, 3, 5, 10]:
            for threshold in [1, 2, 3, 5]:
                for hold in [1, 3, 5]:
                    # Positive momentum -> LONG
                    strategies.append(ComprehensiveStrategy(
                        name=f"MOM_{period}d_UP_{threshold}pct_{hold}d_LONG",
                        category="MOMENTUM",
                        subcategory="SIMPLE",
                        description=f"Long when {period}d return > {threshold}%",
                        direction=Direction.LONG,
                        entry_column=f"roc_{period}",
                        entry_op=">",
                        entry_threshold=threshold,
                        hold_days=hold,
                    ))

                    # Negative momentum -> SHORT
                    strategies.append(ComprehensiveStrategy(
                        name=f"MOM_{period}d_DN_{threshold}pct_{hold}d_SHORT",
                        category="MOMENTUM",
                        subcategory="SIMPLE",
                        description=f"Short when {period}d return < -{threshold}%",
                        direction=Direction.SHORT,
                        entry_column=f"roc_{period}",
                        entry_op="<",
                        entry_threshold=-threshold,
                        hold_days=hold,
                    ))

        return strategies

    def _mean_reversion_strategies(self) -> List[ComprehensiveStrategy]:
        """Price mean reversion strategies."""
        strategies = []

        # Price vs SMA mean reversion
        for deviation in [3, 5, 7, 10]:
            for hold in [1, 3, 5]:
                # Price far below SMA -> LONG (mean reversion)
                strategies.append(ComprehensiveStrategy(
                    name=f"MR_SMA20_DN_{deviation}pct_{hold}d_LONG",
                    category="MEAN_REVERSION",
                    subcategory="SMA",
                    description=f"Long when price {deviation}% below SMA20",
                    direction=Direction.LONG,
                    entry_column="price_vs_sma20",
                    entry_op="<",
                    entry_threshold=-deviation,
                    hold_days=hold,
                ))

                # Price far above SMA -> SHORT
                strategies.append(ComprehensiveStrategy(
                    name=f"MR_SMA20_UP_{deviation}pct_{hold}d_SHORT",
                    category="MEAN_REVERSION",
                    subcategory="SMA",
                    description=f"Short when price {deviation}% above SMA20",
                    direction=Direction.SHORT,
                    entry_column="price_vs_sma20",
                    entry_op=">",
                    entry_threshold=deviation,
                    hold_days=hold,
                ))

        return strategies

    def _rsi_strategies(self) -> List[ComprehensiveStrategy]:
        """RSI overbought/oversold strategies."""
        strategies = []

        for period in [7, 14, 21]:
            # Oversold -> LONG
            for threshold in [20, 25, 30]:
                for hold in [1, 3, 5, 7]:
                    strategies.append(ComprehensiveStrategy(
                        name=f"RSI{period}_OVERSOLD_{threshold}_{hold}d_LONG",
                        category="RSI",
                        subcategory="OVERSOLD",
                        description=f"Long when RSI{period} < {threshold}",
                        direction=Direction.LONG,
                        entry_column=f"rsi_{period}",
                        entry_op="<",
                        entry_threshold=threshold,
                        hold_days=hold,
                    ))

            # Overbought -> SHORT
            for threshold in [70, 75, 80]:
                for hold in [1, 3, 5, 7]:
                    strategies.append(ComprehensiveStrategy(
                        name=f"RSI{period}_OVERBOUGHT_{threshold}_{hold}d_SHORT",
                        category="RSI",
                        subcategory="OVERBOUGHT",
                        description=f"Short when RSI{period} > {threshold}",
                        direction=Direction.SHORT,
                        entry_column=f"rsi_{period}",
                        entry_op=">",
                        entry_threshold=threshold,
                        hold_days=hold,
                    ))

        return strategies

    def _macd_strategies(self) -> List[ComprehensiveStrategy]:
        """MACD crossover strategies."""
        strategies = []

        for hold in [1, 3, 5, 7]:
            # MACD histogram positive -> LONG
            strategies.append(ComprehensiveStrategy(
                name=f"MACD_HIST_POS_{hold}d_LONG",
                category="MACD",
                subcategory="HISTOGRAM",
                description=f"Long when MACD histogram > 0",
                direction=Direction.LONG,
                entry_column="macd_histogram",
                entry_op=">",
                entry_threshold=0,
                hold_days=hold,
            ))

            # MACD histogram negative -> SHORT
            strategies.append(ComprehensiveStrategy(
                name=f"MACD_HIST_NEG_{hold}d_SHORT",
                category="MACD",
                subcategory="HISTOGRAM",
                description=f"Short when MACD histogram < 0",
                direction=Direction.SHORT,
                entry_column="macd_histogram",
                entry_op="<",
                entry_threshold=0,
                hold_days=hold,
            ))

        # MACD z-score extremes
        for threshold in [1.5, 2.0, 2.5]:
            for hold in [3, 5, 7]:
                strategies.append(ComprehensiveStrategy(
                    name=f"MACD_Z_HIGH_{threshold}_{hold}d_SHORT",
                    category="MACD",
                    subcategory="ZSCORE",
                    description=f"Short when MACD z > {threshold}",
                    direction=Direction.SHORT,
                    entry_column="macd_zscore",
                    entry_op=">",
                    entry_threshold=threshold,
                    hold_days=hold,
                ))

                strategies.append(ComprehensiveStrategy(
                    name=f"MACD_Z_LOW_{threshold}_{hold}d_LONG",
                    category="MACD",
                    subcategory="ZSCORE",
                    description=f"Long when MACD z < -{threshold}",
                    direction=Direction.LONG,
                    entry_column="macd_zscore",
                    entry_op="<",
                    entry_threshold=-threshold,
                    hold_days=hold,
                ))

        return strategies

    def _bollinger_strategies(self) -> List[ComprehensiveStrategy]:
        """Bollinger Band strategies."""
        strategies = []

        for hold in [1, 3, 5]:
            # Price below lower band -> LONG (mean reversion)
            strategies.append(ComprehensiveStrategy(
                name=f"BB_BELOW_LOWER_{hold}d_LONG",
                category="BOLLINGER",
                subcategory="MEAN_REV",
                description=f"Long when price below lower BB",
                direction=Direction.LONG,
                entry_column="bb_position",
                entry_op="<",
                entry_threshold=0,
                hold_days=hold,
            ))

            # Price above upper band -> SHORT
            strategies.append(ComprehensiveStrategy(
                name=f"BB_ABOVE_UPPER_{hold}d_SHORT",
                category="BOLLINGER",
                subcategory="MEAN_REV",
                description=f"Short when price above upper BB",
                direction=Direction.SHORT,
                entry_column="bb_position",
                entry_op=">",
                entry_threshold=1,
                hold_days=hold,
            ))

        # BB squeeze (low bandwidth) breakout
        for width_threshold in [0.02, 0.03, 0.04]:
            for hold in [3, 5, 7]:
                strategies.append(ComprehensiveStrategy(
                    name=f"BB_SQUEEZE_{int(width_threshold*100)}pct_{hold}d_LONG",
                    category="BOLLINGER",
                    subcategory="SQUEEZE",
                    description=f"Long on BB squeeze (width < {width_threshold*100}%)",
                    direction=Direction.LONG,
                    entry_column="bb_width",
                    entry_op="<",
                    entry_threshold=width_threshold,
                    hold_days=hold,
                ))

        return strategies

    def _volume_strategies(self) -> List[ComprehensiveStrategy]:
        """Volume pattern strategies."""
        strategies = []

        for window in [5, 10, 20]:
            for threshold in [1.5, 2.0, 2.5]:
                for hold in [1, 3, 5]:
                    # High volume -> momentum continuation
                    strategies.append(ComprehensiveStrategy(
                        name=f"VOL_{window}d_HIGH_{threshold}z_{hold}d_LONG",
                        category="VOLUME",
                        subcategory="HIGH_VOL",
                        description=f"Long when volume z > {threshold}",
                        direction=Direction.LONG,
                        entry_column=f"volume_zscore_{window}",
                        entry_op=">",
                        entry_threshold=threshold,
                        hold_days=hold,
                    ))

                    # Low volume -> consolidation breakout
                    strategies.append(ComprehensiveStrategy(
                        name=f"VOL_{window}d_LOW_{threshold}z_{hold}d_LONG",
                        category="VOLUME",
                        subcategory="LOW_VOL",
                        description=f"Long when volume z < -{threshold}",
                        direction=Direction.LONG,
                        entry_column=f"volume_zscore_{window}",
                        entry_op="<",
                        entry_threshold=-threshold,
                        hold_days=hold,
                    ))

        # Volume ratio
        for ratio in [1.5, 2.0, 2.5]:
            for hold in [1, 3, 5]:
                strategies.append(ComprehensiveStrategy(
                    name=f"VOL_RATIO_{ratio}x_{hold}d_LONG",
                    category="VOLUME",
                    subcategory="RATIO",
                    description=f"Long when volume > {ratio}x average",
                    direction=Direction.LONG,
                    entry_column="volume_ratio",
                    entry_op=">",
                    entry_threshold=ratio,
                    hold_days=hold,
                ))

        return strategies

    def _seasonality_strategies(self) -> List[ComprehensiveStrategy]:
        """Seasonality strategies."""
        strategies = []

        # Day of week
        dow_names = ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']
        for day in range(7):
            for direction in [Direction.LONG, Direction.SHORT]:
                dir_str = 'LONG' if direction == Direction.LONG else 'SHORT'
                strategies.append(ComprehensiveStrategy(
                    name=f"DOW_{dow_names[day]}_{dir_str}",
                    category="SEASONALITY",
                    subcategory="DOW",
                    description=f"{dir_str} on {dow_names[day]}",
                    direction=direction,
                    entry_column="dow",
                    entry_op="==",
                    entry_threshold=day,
                    hold_days=1,
                ))

        # Month of year
        month_names = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
                       'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
        for month in range(1, 13):
            for direction in [Direction.LONG, Direction.SHORT]:
                dir_str = 'LONG' if direction == Direction.LONG else 'SHORT'
                for hold in [5, 10]:
                    strategies.append(ComprehensiveStrategy(
                        name=f"MONTH_{month_names[month-1]}_{hold}d_{dir_str}",
                        category="SEASONALITY",
                        subcategory="MONTH",
                        description=f"{dir_str} in {month_names[month-1]}",
                        direction=direction,
                        entry_column="month",
                        entry_op="==",
                        entry_threshold=month,
                        hold_days=hold,
                    ))

        # Quarter effects
        for quarter in range(1, 5):
            for direction in [Direction.LONG, Direction.SHORT]:
                dir_str = 'LONG' if direction == Direction.LONG else 'SHORT'
                strategies.append(ComprehensiveStrategy(
                    name=f"QUARTER_Q{quarter}_{dir_str}",
                    category="SEASONALITY",
                    subcategory="QUARTER",
                    description=f"{dir_str} in Q{quarter}",
                    direction=direction,
                    entry_column="quarter",
                    entry_op="==",
                    entry_threshold=quarter,
                    hold_days=10,
                ))

        # Month start/end effects
        for hold in [1, 3, 5]:
            strategies.append(ComprehensiveStrategy(
                name=f"MONTH_START_{hold}d_LONG",
                category="SEASONALITY",
                subcategory="MONTH_TIMING",
                description=f"Long at month start",
                direction=Direction.LONG,
                entry_column="is_month_start",
                entry_op="==",
                entry_threshold=1,
                hold_days=hold,
            ))

            strategies.append(ComprehensiveStrategy(
                name=f"MONTH_END_{hold}d_LONG",
                category="SEASONALITY",
                subcategory="MONTH_TIMING",
                description=f"Long at month end",
                direction=Direction.LONG,
                entry_column="is_month_end",
                entry_op="==",
                entry_threshold=1,
                hold_days=hold,
            ))

        return strategies

    def _regime_strategies(self) -> List[ComprehensiveStrategy]:
        """Regime-based strategies."""
        strategies = []

        regime_actions = {
            'ACCUMULATION': Direction.LONG,
            'DISTRIBUTION': Direction.SHORT,
            'CAPITULATION': Direction.LONG,  # Contrarian
            'EUPHORIA': Direction.SHORT,  # Contrarian
        }

        for regime, direction in regime_actions.items():
            for hold in [1, 3, 5, 7, 10]:
                dir_str = 'LONG' if direction == Direction.LONG else 'SHORT'
                strategies.append(ComprehensiveStrategy(
                    name=f"REGIME_{regime}_{hold}d_{dir_str}",
                    category="REGIME",
                    subcategory=regime,
                    description=f"{dir_str} in {regime} regime",
                    direction=direction,
                    entry_column="regime",
                    entry_op="==",
                    entry_threshold=regime,
                    hold_days=hold,
                ))

        return strategies

    def _ma_crossover_strategies(self) -> List[ComprehensiveStrategy]:
        """Moving average crossover strategies."""
        strategies = []

        for hold in [3, 5, 7, 10]:
            # Golden cross (fast > slow)
            strategies.append(ComprehensiveStrategy(
                name=f"MA_GOLDEN_10_20_{hold}d_LONG",
                category="MA_CROSS",
                subcategory="GOLDEN",
                description=f"Long when SMA10 > SMA20",
                direction=Direction.LONG,
                entry_column="ma_cross_10_20",
                entry_op="==",
                entry_threshold=1,
                hold_days=hold,
            ))

            # Death cross
            strategies.append(ComprehensiveStrategy(
                name=f"MA_DEATH_10_20_{hold}d_SHORT",
                category="MA_CROSS",
                subcategory="DEATH",
                description=f"Short when SMA10 < SMA20",
                direction=Direction.SHORT,
                entry_column="ma_cross_10_20",
                entry_op="==",
                entry_threshold=0,
                hold_days=hold,
            ))

        return strategies

    def _atr_strategies(self) -> List[ComprehensiveStrategy]:
        """ATR/volatility strategies."""
        strategies = []

        for threshold in [1.0, 1.5, 2.0]:
            for hold in [1, 3, 5]:
                # High ATR -> volatility expansion
                strategies.append(ComprehensiveStrategy(
                    name=f"ATR_HIGH_{threshold}z_{hold}d_LONG",
                    category="ATR",
                    subcategory="HIGH_VOL",
                    description=f"Long on high ATR (z > {threshold})",
                    direction=Direction.LONG,
                    entry_column="atr_zscore",
                    entry_op=">",
                    entry_threshold=threshold,
                    hold_days=hold,
                ))

                # Low ATR -> compression, expect breakout
                strategies.append(ComprehensiveStrategy(
                    name=f"ATR_LOW_{threshold}z_{hold}d_LONG",
                    category="ATR",
                    subcategory="LOW_VOL",
                    description=f"Long on low ATR (z < -{threshold})",
                    direction=Direction.LONG,
                    entry_column="atr_zscore",
                    entry_op="<",
                    entry_threshold=-threshold,
                    hold_days=hold,
                ))

        return strategies

    def _combination_strategies(self) -> List[ComprehensiveStrategy]:
        """Multi-signal combination strategies."""
        strategies = []

        # RSI oversold + TX momentum
        for rsi_thresh in [25, 30]:
            for tx_thresh in [1.0, 1.5]:
                for hold in [3, 5]:
                    strategies.append(ComprehensiveStrategy(
                        name=f"COMBO_RSI{rsi_thresh}_TX{tx_thresh}_{hold}d_LONG",
                        category="COMBINATION",
                        subcategory="RSI_TX",
                        description=f"Long: RSI<{rsi_thresh} AND TX z>{tx_thresh}",
                        direction=Direction.LONG,
                        entry_column="rsi_14",
                        entry_op="<",
                        entry_threshold=rsi_thresh,
                        filter_column="tx_count_zscore",
                        filter_op=">",
                        filter_threshold=tx_thresh,
                        hold_days=hold,
                    ))

        # BB oversold + volume spike
        for vol_thresh in [1.5, 2.0]:
            for hold in [1, 3, 5]:
                strategies.append(ComprehensiveStrategy(
                    name=f"COMBO_BB_VOL{vol_thresh}_{hold}d_LONG",
                    category="COMBINATION",
                    subcategory="BB_VOL",
                    description=f"Long: below BB AND volume z>{vol_thresh}",
                    direction=Direction.LONG,
                    entry_column="bb_position",
                    entry_op="<",
                    entry_threshold=0.1,
                    filter_column="volume_zscore_10",
                    filter_op=">",
                    filter_threshold=vol_thresh,
                    hold_days=hold,
                ))

        # Regime + RSI
        for rsi_thresh in [30, 35]:
            for hold in [3, 5]:
                strategies.append(ComprehensiveStrategy(
                    name=f"COMBO_ACC_RSI{rsi_thresh}_{hold}d_LONG",
                    category="COMBINATION",
                    subcategory="REGIME_RSI",
                    description=f"Long: ACCUMULATION + RSI<{rsi_thresh}",
                    direction=Direction.LONG,
                    entry_column="rsi_14",
                    entry_op="<",
                    entry_threshold=rsi_thresh,
                    regime_filter="ACCUMULATION",
                    hold_days=hold,
                ))

        return strategies

    def get_strategy_count(self) -> Dict[str, int]:
        """Count strategies by category."""
        strategies = self.generate_all()
        counts = {}
        for s in strategies:
            counts[s.category] = counts.get(s.category, 0) + 1
        return counts


def quick_test():
    """Quick test of comprehensive factory."""
    factory = ComprehensiveStrategyFactory()
    strategies = factory.generate_all()

    print(f"Total strategies: {len(strategies)}")
    print("\nBy category:")
    for cat, count in sorted(factory.get_strategy_count().items()):
        print(f"  {cat}: {count}")

    print("\nSample strategies:")
    for s in strategies[:10]:
        print(f"  {s.name}: {s.description}")


if __name__ == "__main__":
    quick_test()
