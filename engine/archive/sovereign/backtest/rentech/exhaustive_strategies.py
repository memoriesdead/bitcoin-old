"""
Exhaustive Strategy Factory - Find EVERY Pattern Like RenTech
=============================================================

RenTech principle: Test everything, keep what works statistically.

This module tests patterns NOT covered in comprehensive_strategies.py:
1. Microstructure (consecutive days, gaps, streaks)
2. Halving cycles (Bitcoin-specific 4-year cycle)
3. Advanced calendar (week of month, year patterns)
4. Cross-correlations (TX-price, volume-price relationships)
5. Volatility regimes (clustering, mean reversion)
6. Multi-timeframe (short vs long term alignment)
7. HMM transitions (regime change patterns)
8. Blockchain-specific (difficulty, fees, block time)
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime


class Direction(Enum):
    LONG = 1
    SHORT = -1


@dataclass
class ExhaustiveStrategy:
    """Extended strategy with complex entry logic."""
    name: str
    category: str
    subcategory: str
    description: str
    direction: Direction
    hold_days: int = 5

    # Entry logic - list of conditions (ALL must be met)
    conditions: List[Dict] = None

    # Custom entry function for complex logic
    custom_entry: callable = None

    def __post_init__(self):
        if self.conditions is None:
            self.conditions = []

    def check_entry(self, row: pd.Series, prev_row: pd.Series = None,
                   history: pd.DataFrame = None) -> bool:
        """Check if entry condition is met."""
        # Use custom entry if provided
        if self.custom_entry:
            return self.custom_entry(row, prev_row, history)

        # Check all conditions
        for cond in self.conditions:
            if not self._check_single_condition(row, prev_row, history, cond):
                return False
        return True

    def _check_single_condition(self, row, prev_row, history, cond) -> bool:
        """Check a single condition."""
        col = cond.get('column')
        op = cond.get('op')
        val = cond.get('value')
        val2 = cond.get('value2')

        if col not in row.index:
            return False

        x = row[col]
        if pd.isna(x):
            return False

        if op == '>':
            return x > val
        elif op == '<':
            return x < val
        elif op == '>=':
            return x >= val
        elif op == '<=':
            return x <= val
        elif op == '==':
            return x == val
        elif op == 'between':
            return val <= x <= val2
        elif op == 'cross_above' and prev_row is not None:
            prev_x = prev_row.get(col, np.nan)
            return not pd.isna(prev_x) and prev_x <= val and x > val
        elif op == 'cross_below' and prev_row is not None:
            prev_x = prev_row.get(col, np.nan)
            return not pd.isna(prev_x) and prev_x >= val and x < val

        return False

    def get_signal_direction(self) -> int:
        return self.direction.value


class ExhaustiveStrategyFactory:
    """
    Generate EVERY possible pattern for RenTech-style testing.

    New Categories (500+ more strategies):
    - MICROSTRUCTURE: Consecutive days, gaps, streaks
    - HALVING: Bitcoin 4-year halving cycle
    - WEEK_OF_MONTH: First/last week effects
    - CROSS_CORRELATION: TX-price, volume-price dynamics
    - VOL_REGIME: Volatility clustering patterns
    - MULTI_TIMEFRAME: Short/long term alignment
    - REGIME_TRANSITION: HMM state change patterns
    - BLOCKCHAIN: Difficulty, block time, fees
    """

    def generate_all(self) -> List[ExhaustiveStrategy]:
        """Generate all exhaustive strategies."""
        strategies = []

        strategies.extend(self._microstructure_strategies())
        strategies.extend(self._halving_cycle_strategies())
        strategies.extend(self._week_of_month_strategies())
        strategies.extend(self._cross_correlation_strategies())
        strategies.extend(self._volatility_regime_strategies())
        strategies.extend(self._multi_timeframe_strategies())
        strategies.extend(self._regime_transition_strategies())
        strategies.extend(self._blockchain_specific_strategies())
        strategies.extend(self._extreme_event_strategies())
        strategies.extend(self._pattern_sequence_strategies())

        return strategies

    def _microstructure_strategies(self) -> List[ExhaustiveStrategy]:
        """Consecutive days, streaks, and gap patterns."""
        strategies = []

        # Consecutive up/down days
        for streak in [2, 3, 4, 5]:
            for hold in [1, 3, 5]:
                # After N consecutive up days
                strategies.append(ExhaustiveStrategy(
                    name=f"STREAK_UP_{streak}d_CONT_{hold}d_LONG",
                    category="MICROSTRUCTURE",
                    subcategory="STREAK",
                    description=f"Long after {streak} consecutive up days (momentum)",
                    direction=Direction.LONG,
                    hold_days=hold,
                    conditions=[{'column': f'consecutive_up_{streak}', 'op': '==', 'value': 1}],
                ))

                strategies.append(ExhaustiveStrategy(
                    name=f"STREAK_UP_{streak}d_REV_{hold}d_SHORT",
                    category="MICROSTRUCTURE",
                    subcategory="STREAK",
                    description=f"Short after {streak} consecutive up days (reversal)",
                    direction=Direction.SHORT,
                    hold_days=hold,
                    conditions=[{'column': f'consecutive_up_{streak}', 'op': '==', 'value': 1}],
                ))

                # After N consecutive down days
                strategies.append(ExhaustiveStrategy(
                    name=f"STREAK_DN_{streak}d_CONT_{hold}d_SHORT",
                    category="MICROSTRUCTURE",
                    subcategory="STREAK",
                    description=f"Short after {streak} consecutive down days (momentum)",
                    direction=Direction.SHORT,
                    hold_days=hold,
                    conditions=[{'column': f'consecutive_down_{streak}', 'op': '==', 'value': 1}],
                ))

                strategies.append(ExhaustiveStrategy(
                    name=f"STREAK_DN_{streak}d_REV_{hold}d_LONG",
                    category="MICROSTRUCTURE",
                    subcategory="STREAK",
                    description=f"Long after {streak} consecutive down days (reversal)",
                    direction=Direction.LONG,
                    hold_days=hold,
                    conditions=[{'column': f'consecutive_down_{streak}', 'op': '==', 'value': 1}],
                ))

        # Gap patterns (open vs previous close)
        for gap_pct in [1, 2, 3, 5]:
            for hold in [1, 3, 5]:
                # Gap up continuation
                strategies.append(ExhaustiveStrategy(
                    name=f"GAP_UP_{gap_pct}pct_CONT_{hold}d_LONG",
                    category="MICROSTRUCTURE",
                    subcategory="GAP",
                    description=f"Long on gap up > {gap_pct}% (momentum)",
                    direction=Direction.LONG,
                    hold_days=hold,
                    conditions=[{'column': 'gap_pct', 'op': '>', 'value': gap_pct}],
                ))

                # Gap up fade
                strategies.append(ExhaustiveStrategy(
                    name=f"GAP_UP_{gap_pct}pct_FADE_{hold}d_SHORT",
                    category="MICROSTRUCTURE",
                    subcategory="GAP",
                    description=f"Short on gap up > {gap_pct}% (fade)",
                    direction=Direction.SHORT,
                    hold_days=hold,
                    conditions=[{'column': 'gap_pct', 'op': '>', 'value': gap_pct}],
                ))

                # Gap down
                strategies.append(ExhaustiveStrategy(
                    name=f"GAP_DN_{gap_pct}pct_CONT_{hold}d_SHORT",
                    category="MICROSTRUCTURE",
                    subcategory="GAP",
                    description=f"Short on gap down > {gap_pct}% (momentum)",
                    direction=Direction.SHORT,
                    hold_days=hold,
                    conditions=[{'column': 'gap_pct', 'op': '<', 'value': -gap_pct}],
                ))

                strategies.append(ExhaustiveStrategy(
                    name=f"GAP_DN_{gap_pct}pct_FADE_{hold}d_LONG",
                    category="MICROSTRUCTURE",
                    subcategory="GAP",
                    description=f"Long on gap down > {gap_pct}% (fade)",
                    direction=Direction.LONG,
                    hold_days=hold,
                    conditions=[{'column': 'gap_pct', 'op': '<', 'value': -gap_pct}],
                ))

        # Intraday range patterns
        for range_mult in [1.5, 2.0, 2.5]:
            for hold in [1, 3, 5]:
                strategies.append(ExhaustiveStrategy(
                    name=f"WIDE_RANGE_{range_mult}x_{hold}d_LONG",
                    category="MICROSTRUCTURE",
                    subcategory="RANGE",
                    description=f"Long on wide range day (> {range_mult}x ATR)",
                    direction=Direction.LONG,
                    hold_days=hold,
                    conditions=[{'column': 'range_vs_atr', 'op': '>', 'value': range_mult}],
                ))

                strategies.append(ExhaustiveStrategy(
                    name=f"NARROW_RANGE_{range_mult}x_{hold}d_LONG",
                    category="MICROSTRUCTURE",
                    subcategory="RANGE",
                    description=f"Long on narrow range day (< 1/{range_mult}x ATR)",
                    direction=Direction.LONG,
                    hold_days=hold,
                    conditions=[{'column': 'range_vs_atr', 'op': '<', 'value': 1/range_mult}],
                ))

        return strategies

    def _halving_cycle_strategies(self) -> List[ExhaustiveStrategy]:
        """Bitcoin halving cycle patterns (every ~4 years)."""
        strategies = []

        # Days since/until halving
        # Halvings: 2012-11-28, 2016-07-09, 2020-05-11, 2024-04-19
        for days_range in [(0, 30), (30, 90), (90, 180), (180, 365), (365, 730)]:
            for hold in [5, 10, 20]:
                # Days AFTER halving
                strategies.append(ExhaustiveStrategy(
                    name=f"HALVING_AFTER_{days_range[0]}_{days_range[1]}d_{hold}d_LONG",
                    category="HALVING",
                    subcategory="POST_HALVING",
                    description=f"Long {days_range[0]}-{days_range[1]} days after halving",
                    direction=Direction.LONG,
                    hold_days=hold,
                    conditions=[
                        {'column': 'days_since_halving', 'op': '>=', 'value': days_range[0]},
                        {'column': 'days_since_halving', 'op': '<', 'value': days_range[1]},
                    ],
                ))

                # Days BEFORE halving
                strategies.append(ExhaustiveStrategy(
                    name=f"HALVING_BEFORE_{days_range[0]}_{days_range[1]}d_{hold}d_LONG",
                    category="HALVING",
                    subcategory="PRE_HALVING",
                    description=f"Long {days_range[0]}-{days_range[1]} days before halving",
                    direction=Direction.LONG,
                    hold_days=hold,
                    conditions=[
                        {'column': 'days_until_halving', 'op': '>=', 'value': days_range[0]},
                        {'column': 'days_until_halving', 'op': '<', 'value': days_range[1]},
                    ],
                ))

        # Halving cycle phase (0-4 years = 0.0-1.0)
        for phase_start, phase_end in [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]:
            for direction in [Direction.LONG, Direction.SHORT]:
                dir_str = 'LONG' if direction == Direction.LONG else 'SHORT'
                for hold in [10, 20]:
                    strategies.append(ExhaustiveStrategy(
                        name=f"HALVING_PHASE_{int(phase_start*100)}_{int(phase_end*100)}_{hold}d_{dir_str}",
                        category="HALVING",
                        subcategory="CYCLE_PHASE",
                        description=f"{dir_str} in halving cycle phase {phase_start}-{phase_end}",
                        direction=direction,
                        hold_days=hold,
                        conditions=[
                            {'column': 'halving_cycle_phase', 'op': '>=', 'value': phase_start},
                            {'column': 'halving_cycle_phase', 'op': '<', 'value': phase_end},
                        ],
                    ))

        return strategies

    def _week_of_month_strategies(self) -> List[ExhaustiveStrategy]:
        """Week of month and turn-of-month effects."""
        strategies = []

        # Week of month (1-5)
        for week in range(1, 6):
            for direction in [Direction.LONG, Direction.SHORT]:
                dir_str = 'LONG' if direction == Direction.LONG else 'SHORT'
                for hold in [3, 5]:
                    strategies.append(ExhaustiveStrategy(
                        name=f"WEEK_{week}_OF_MONTH_{hold}d_{dir_str}",
                        category="WEEK_OF_MONTH",
                        subcategory="SPECIFIC_WEEK",
                        description=f"{dir_str} in week {week} of month",
                        direction=direction,
                        hold_days=hold,
                        conditions=[{'column': 'week_of_month', 'op': '==', 'value': week}],
                    ))

        # First/last 3 days of month
        for hold in [1, 3, 5]:
            strategies.append(ExhaustiveStrategy(
                name=f"FIRST_3_DAYS_{hold}d_LONG",
                category="WEEK_OF_MONTH",
                subcategory="TURN_OF_MONTH",
                description=f"Long in first 3 days of month",
                direction=Direction.LONG,
                hold_days=hold,
                conditions=[{'column': 'dom', 'op': '<=', 'value': 3}],
            ))

            strategies.append(ExhaustiveStrategy(
                name=f"LAST_3_DAYS_{hold}d_LONG",
                category="WEEK_OF_MONTH",
                subcategory="TURN_OF_MONTH",
                description=f"Long in last 3 days of month",
                direction=Direction.LONG,
                hold_days=hold,
                conditions=[{'column': 'dom', 'op': '>=', 'value': 28}],
            ))

        # Year effects
        for month in [1, 12]:  # January effect, December tax selling
            month_name = 'JAN' if month == 1 else 'DEC'
            for hold in [5, 10, 20]:
                strategies.append(ExhaustiveStrategy(
                    name=f"{month_name}_EFFECT_{hold}d_LONG",
                    category="WEEK_OF_MONTH",
                    subcategory="YEAR_EFFECTS",
                    description=f"Long in {month_name} (January effect)",
                    direction=Direction.LONG,
                    hold_days=hold,
                    conditions=[{'column': 'month', 'op': '==', 'value': month}],
                ))

        return strategies

    def _cross_correlation_strategies(self) -> List[ExhaustiveStrategy]:
        """TX-price and volume-price correlation patterns."""
        strategies = []

        # TX-price correlation extremes
        for threshold in [0.5, 0.7, 0.8]:
            for hold in [3, 5, 7]:
                # High positive correlation - TX and price move together
                strategies.append(ExhaustiveStrategy(
                    name=f"TX_PRICE_CORR_HIGH_{int(threshold*100)}_{hold}d_LONG",
                    category="CROSS_CORRELATION",
                    subcategory="TX_PRICE",
                    description=f"Long when TX-price corr > {threshold}",
                    direction=Direction.LONG,
                    hold_days=hold,
                    conditions=[{'column': 'tx_price_corr_20', 'op': '>', 'value': threshold}],
                ))

                # Negative correlation - divergence
                strategies.append(ExhaustiveStrategy(
                    name=f"TX_PRICE_CORR_NEG_{int(threshold*100)}_{hold}d_SHORT",
                    category="CROSS_CORRELATION",
                    subcategory="TX_PRICE",
                    description=f"Short when TX-price corr < -{threshold}",
                    direction=Direction.SHORT,
                    hold_days=hold,
                    conditions=[{'column': 'tx_price_corr_20', 'op': '<', 'value': -threshold}],
                ))

        # TX leads price (predictive)
        for threshold in [0.3, 0.5]:
            for hold in [1, 3, 5]:
                strategies.append(ExhaustiveStrategy(
                    name=f"TX_LEADS_PRICE_{int(threshold*100)}_{hold}d_LONG",
                    category="CROSS_CORRELATION",
                    subcategory="TX_LEAD",
                    description=f"Long when TX lead correlation > {threshold}",
                    direction=Direction.LONG,
                    hold_days=hold,
                    conditions=[{'column': 'tx_lead_corr_20', 'op': '>', 'value': threshold}],
                ))

        return strategies

    def _volatility_regime_strategies(self) -> List[ExhaustiveStrategy]:
        """Volatility clustering and regime patterns."""
        strategies = []

        # Volatility percentile
        for pct_low, pct_high in [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]:
            for direction in [Direction.LONG, Direction.SHORT]:
                dir_str = 'LONG' if direction == Direction.LONG else 'SHORT'
                for hold in [3, 5]:
                    strategies.append(ExhaustiveStrategy(
                        name=f"VOL_PCT_{pct_low}_{pct_high}_{hold}d_{dir_str}",
                        category="VOL_REGIME",
                        subcategory="PERCENTILE",
                        description=f"{dir_str} when vol in {pct_low}-{pct_high} percentile",
                        direction=direction,
                        hold_days=hold,
                        conditions=[
                            {'column': 'volatility_percentile', 'op': '>=', 'value': pct_low},
                            {'column': 'volatility_percentile', 'op': '<', 'value': pct_high},
                        ],
                    ))

        # Volatility expansion/contraction
        for change_thresh in [1.5, 2.0]:
            for hold in [1, 3, 5]:
                # Vol expansion
                strategies.append(ExhaustiveStrategy(
                    name=f"VOL_EXPAND_{change_thresh}x_{hold}d_LONG",
                    category="VOL_REGIME",
                    subcategory="VOL_CHANGE",
                    description=f"Long when vol expanded > {change_thresh}x",
                    direction=Direction.LONG,
                    hold_days=hold,
                    conditions=[{'column': 'vol_change_ratio', 'op': '>', 'value': change_thresh}],
                ))

                # Vol contraction
                strategies.append(ExhaustiveStrategy(
                    name=f"VOL_CONTRACT_{change_thresh}x_{hold}d_LONG",
                    category="VOL_REGIME",
                    subcategory="VOL_CHANGE",
                    description=f"Long when vol contracted < 1/{change_thresh}",
                    direction=Direction.LONG,
                    hold_days=hold,
                    conditions=[{'column': 'vol_change_ratio', 'op': '<', 'value': 1/change_thresh}],
                ))

        return strategies

    def _multi_timeframe_strategies(self) -> List[ExhaustiveStrategy]:
        """Short-term vs long-term signal alignment."""
        strategies = []

        # Short-term momentum aligned with long-term trend
        for hold in [3, 5, 7]:
            # Both bullish
            strategies.append(ExhaustiveStrategy(
                name=f"MTF_BULL_ALIGN_{hold}d_LONG",
                category="MULTI_TIMEFRAME",
                subcategory="ALIGNMENT",
                description=f"Long when short & long term both bullish",
                direction=Direction.LONG,
                hold_days=hold,
                conditions=[
                    {'column': 'roc_5', 'op': '>', 'value': 0},
                    {'column': 'roc_20', 'op': '>', 'value': 0},
                    {'column': 'ma_cross_10_20', 'op': '==', 'value': 1},
                ],
            ))

            # Both bearish
            strategies.append(ExhaustiveStrategy(
                name=f"MTF_BEAR_ALIGN_{hold}d_SHORT",
                category="MULTI_TIMEFRAME",
                subcategory="ALIGNMENT",
                description=f"Short when short & long term both bearish",
                direction=Direction.SHORT,
                hold_days=hold,
                conditions=[
                    {'column': 'roc_5', 'op': '<', 'value': 0},
                    {'column': 'roc_20', 'op': '<', 'value': 0},
                    {'column': 'ma_cross_10_20', 'op': '==', 'value': 0},
                ],
            ))

            # Divergence - short bullish, long bearish (reversal)
            strategies.append(ExhaustiveStrategy(
                name=f"MTF_DIV_UP_{hold}d_LONG",
                category="MULTI_TIMEFRAME",
                subcategory="DIVERGENCE",
                description=f"Long on bullish divergence (short up, long down)",
                direction=Direction.LONG,
                hold_days=hold,
                conditions=[
                    {'column': 'roc_5', 'op': '>', 'value': 2},
                    {'column': 'roc_20', 'op': '<', 'value': 0},
                ],
            ))

        return strategies

    def _regime_transition_strategies(self) -> List[ExhaustiveStrategy]:
        """HMM regime change patterns."""
        strategies = []

        # Transitions between regimes
        transitions = [
            ('NEUTRAL', 'ACCUMULATION', Direction.LONG),
            ('DISTRIBUTION', 'NEUTRAL', Direction.SHORT),
            ('CAPITULATION', 'NEUTRAL', Direction.LONG),
            ('NEUTRAL', 'EUPHORIA', Direction.SHORT),
            ('ACCUMULATION', 'EUPHORIA', Direction.LONG),
            ('EUPHORIA', 'DISTRIBUTION', Direction.SHORT),
        ]

        for from_regime, to_regime, direction in transitions:
            dir_str = 'LONG' if direction == Direction.LONG else 'SHORT'
            for hold in [3, 5, 10]:
                strategies.append(ExhaustiveStrategy(
                    name=f"TRANS_{from_regime[:3]}_{to_regime[:3]}_{hold}d_{dir_str}",
                    category="REGIME_TRANSITION",
                    subcategory="STATE_CHANGE",
                    description=f"{dir_str} on {from_regime}->{to_regime} transition",
                    direction=direction,
                    hold_days=hold,
                    conditions=[
                        {'column': 'prev_regime', 'op': '==', 'value': from_regime},
                        {'column': 'regime', 'op': '==', 'value': to_regime},
                    ],
                ))

        # Days in regime
        for regime in ['ACCUMULATION', 'DISTRIBUTION', 'EUPHORIA', 'CAPITULATION']:
            for days in [3, 5, 10, 20]:
                dir_map = {'ACCUMULATION': Direction.LONG, 'DISTRIBUTION': Direction.SHORT,
                          'EUPHORIA': Direction.SHORT, 'CAPITULATION': Direction.LONG}
                direction = dir_map.get(regime, Direction.LONG)
                dir_str = 'LONG' if direction == Direction.LONG else 'SHORT'

                for hold in [3, 5]:
                    strategies.append(ExhaustiveStrategy(
                        name=f"REGIME_{regime[:4]}_{days}d_IN_{hold}d_{dir_str}",
                        category="REGIME_TRANSITION",
                        subcategory="TIME_IN_REGIME",
                        description=f"{dir_str} after {days}+ days in {regime}",
                        direction=direction,
                        hold_days=hold,
                        conditions=[
                            {'column': 'regime', 'op': '==', 'value': regime},
                            {'column': 'days_in_regime', 'op': '>=', 'value': days},
                        ],
                    ))

        return strategies

    def _blockchain_specific_strategies(self) -> List[ExhaustiveStrategy]:
        """Bitcoin blockchain-specific patterns."""
        strategies = []

        # Difficulty adjustment effects
        for direction in [Direction.LONG, Direction.SHORT]:
            dir_str = 'LONG' if direction == Direction.LONG else 'SHORT'
            for hold in [3, 7, 14]:
                # After difficulty increase
                strategies.append(ExhaustiveStrategy(
                    name=f"DIFF_INCREASE_{hold}d_{dir_str}",
                    category="BLOCKCHAIN",
                    subcategory="DIFFICULTY",
                    description=f"{dir_str} after difficulty increase",
                    direction=direction,
                    hold_days=hold,
                    conditions=[{'column': 'difficulty_change', 'op': '>', 'value': 0}],
                ))

                # After difficulty decrease
                strategies.append(ExhaustiveStrategy(
                    name=f"DIFF_DECREASE_{hold}d_{dir_str}",
                    category="BLOCKCHAIN",
                    subcategory="DIFFICULTY",
                    description=f"{dir_str} after difficulty decrease",
                    direction=direction,
                    hold_days=hold,
                    conditions=[{'column': 'difficulty_change', 'op': '<', 'value': 0}],
                ))

        # Block time patterns (fast/slow blocks)
        for threshold in [8, 9, 11, 12]:  # Normal is ~10 min
            for hold in [1, 3, 5]:
                if threshold < 10:
                    strategies.append(ExhaustiveStrategy(
                        name=f"FAST_BLOCKS_{threshold}m_{hold}d_LONG",
                        category="BLOCKCHAIN",
                        subcategory="BLOCK_TIME",
                        description=f"Long when avg block time < {threshold}min",
                        direction=Direction.LONG,
                        hold_days=hold,
                        conditions=[{'column': 'avg_block_time', 'op': '<', 'value': threshold}],
                    ))
                else:
                    strategies.append(ExhaustiveStrategy(
                        name=f"SLOW_BLOCKS_{threshold}m_{hold}d_SHORT",
                        category="BLOCKCHAIN",
                        subcategory="BLOCK_TIME",
                        description=f"Short when avg block time > {threshold}min",
                        direction=Direction.SHORT,
                        hold_days=hold,
                        conditions=[{'column': 'avg_block_time', 'op': '>', 'value': threshold}],
                    ))

        # Hash rate changes
        for change_pct in [5, 10, 15]:
            for hold in [3, 7]:
                strategies.append(ExhaustiveStrategy(
                    name=f"HASHRATE_UP_{change_pct}pct_{hold}d_LONG",
                    category="BLOCKCHAIN",
                    subcategory="HASHRATE",
                    description=f"Long when hash rate up {change_pct}%+",
                    direction=Direction.LONG,
                    hold_days=hold,
                    conditions=[{'column': 'hashrate_change_pct', 'op': '>', 'value': change_pct}],
                ))

                strategies.append(ExhaustiveStrategy(
                    name=f"HASHRATE_DN_{change_pct}pct_{hold}d_SHORT",
                    category="BLOCKCHAIN",
                    subcategory="HASHRATE",
                    description=f"Short when hash rate down {change_pct}%+",
                    direction=Direction.SHORT,
                    hold_days=hold,
                    conditions=[{'column': 'hashrate_change_pct', 'op': '<', 'value': -change_pct}],
                ))

        # Fee patterns
        for threshold in [1.5, 2.0, 3.0]:
            for hold in [1, 3, 5]:
                strategies.append(ExhaustiveStrategy(
                    name=f"HIGH_FEES_{threshold}z_{hold}d_LONG",
                    category="BLOCKCHAIN",
                    subcategory="FEES",
                    description=f"Long when fee z-score > {threshold}",
                    direction=Direction.LONG,
                    hold_days=hold,
                    conditions=[{'column': 'fee_zscore', 'op': '>', 'value': threshold}],
                ))

        return strategies

    def _extreme_event_strategies(self) -> List[ExhaustiveStrategy]:
        """Extreme price move patterns."""
        strategies = []

        # Large single-day moves
        for pct in [5, 7, 10]:
            for hold in [1, 3, 5, 7]:
                # After big up day
                strategies.append(ExhaustiveStrategy(
                    name=f"BIG_UP_{pct}pct_CONT_{hold}d_LONG",
                    category="EXTREME",
                    subcategory="BIG_MOVE",
                    description=f"Long after {pct}%+ up day (continuation)",
                    direction=Direction.LONG,
                    hold_days=hold,
                    conditions=[{'column': 'roc_1', 'op': '>', 'value': pct}],
                ))

                strategies.append(ExhaustiveStrategy(
                    name=f"BIG_UP_{pct}pct_REV_{hold}d_SHORT",
                    category="EXTREME",
                    subcategory="BIG_MOVE",
                    description=f"Short after {pct}%+ up day (reversal)",
                    direction=Direction.SHORT,
                    hold_days=hold,
                    conditions=[{'column': 'roc_1', 'op': '>', 'value': pct}],
                ))

                # After big down day
                strategies.append(ExhaustiveStrategy(
                    name=f"BIG_DN_{pct}pct_CONT_{hold}d_SHORT",
                    category="EXTREME",
                    subcategory="BIG_MOVE",
                    description=f"Short after {pct}%+ down day (continuation)",
                    direction=Direction.SHORT,
                    hold_days=hold,
                    conditions=[{'column': 'roc_1', 'op': '<', 'value': -pct}],
                ))

                strategies.append(ExhaustiveStrategy(
                    name=f"BIG_DN_{pct}pct_REV_{hold}d_LONG",
                    category="EXTREME",
                    subcategory="BIG_MOVE",
                    description=f"Long after {pct}%+ down day (reversal)",
                    direction=Direction.LONG,
                    hold_days=hold,
                    conditions=[{'column': 'roc_1', 'op': '<', 'value': -pct}],
                ))

        # New highs/lows
        for lookback in [20, 50, 100]:
            for hold in [3, 5, 10]:
                strategies.append(ExhaustiveStrategy(
                    name=f"NEW_HIGH_{lookback}d_{hold}d_LONG",
                    category="EXTREME",
                    subcategory="NEW_EXTREME",
                    description=f"Long on new {lookback}-day high",
                    direction=Direction.LONG,
                    hold_days=hold,
                    conditions=[{'column': f'is_high_{lookback}', 'op': '==', 'value': 1}],
                ))

                strategies.append(ExhaustiveStrategy(
                    name=f"NEW_LOW_{lookback}d_{hold}d_LONG",
                    category="EXTREME",
                    subcategory="NEW_EXTREME",
                    description=f"Long on new {lookback}-day low (reversal)",
                    direction=Direction.LONG,
                    hold_days=hold,
                    conditions=[{'column': f'is_low_{lookback}', 'op': '==', 'value': 1}],
                ))

        return strategies

    def _pattern_sequence_strategies(self) -> List[ExhaustiveStrategy]:
        """Multi-day pattern sequences."""
        strategies = []

        # Inside day (range within previous day)
        for hold in [1, 3, 5]:
            strategies.append(ExhaustiveStrategy(
                name=f"INSIDE_DAY_{hold}d_LONG",
                category="PATTERN",
                subcategory="INSIDE",
                description=f"Long after inside day (volatility contraction)",
                direction=Direction.LONG,
                hold_days=hold,
                conditions=[{'column': 'is_inside_day', 'op': '==', 'value': 1}],
            ))

        # Outside day (range exceeds previous day)
        for hold in [1, 3, 5]:
            strategies.append(ExhaustiveStrategy(
                name=f"OUTSIDE_DAY_UP_{hold}d_LONG",
                category="PATTERN",
                subcategory="OUTSIDE",
                description=f"Long after bullish outside day",
                direction=Direction.LONG,
                hold_days=hold,
                conditions=[
                    {'column': 'is_outside_day', 'op': '==', 'value': 1},
                    {'column': 'roc_1', 'op': '>', 'value': 0},
                ],
            ))

            strategies.append(ExhaustiveStrategy(
                name=f"OUTSIDE_DAY_DN_{hold}d_SHORT",
                category="PATTERN",
                subcategory="OUTSIDE",
                description=f"Short after bearish outside day",
                direction=Direction.SHORT,
                hold_days=hold,
                conditions=[
                    {'column': 'is_outside_day', 'op': '==', 'value': 1},
                    {'column': 'roc_1', 'op': '<', 'value': 0},
                ],
            ))

        # Doji (open == close, small body)
        for hold in [1, 3, 5]:
            strategies.append(ExhaustiveStrategy(
                name=f"DOJI_REVERSAL_{hold}d_LONG",
                category="PATTERN",
                subcategory="CANDLESTICK",
                description=f"Long after doji in downtrend",
                direction=Direction.LONG,
                hold_days=hold,
                conditions=[
                    {'column': 'is_doji', 'op': '==', 'value': 1},
                    {'column': 'roc_5', 'op': '<', 'value': -3},
                ],
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
    """Quick test of exhaustive factory."""
    factory = ExhaustiveStrategyFactory()
    strategies = factory.generate_all()

    print(f"Total exhaustive strategies: {len(strategies)}")
    print("\nBy category:")
    for cat, count in sorted(factory.get_strategy_count().items()):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    quick_test()
