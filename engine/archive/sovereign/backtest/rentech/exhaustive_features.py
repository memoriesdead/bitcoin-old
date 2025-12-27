"""
Exhaustive Feature Engineering - ALL RenTech-Style Features
============================================================

Calculates EVERY feature needed for exhaustive strategy testing:
1. Microstructure features (streaks, gaps, ranges)
2. Halving cycle features (days since/until, cycle phase)
3. Calendar features (week of month, year effects)
4. Volatility regime features (percentile, changes)
5. Pattern features (inside/outside days, doji)
6. Extreme event features (new highs/lows)
7. Blockchain features (difficulty, hash rate, fees)
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime, timedelta


# Bitcoin halving dates
HALVING_DATES = [
    datetime(2012, 11, 28),  # Block 210,000
    datetime(2016, 7, 9),    # Block 420,000
    datetime(2020, 5, 11),   # Block 630,000
    datetime(2024, 4, 19),   # Block 840,000 (approximate)
    datetime(2028, 4, 1),    # Block 1,050,000 (approximate)
]

# Average halving cycle in days (~4 years)
HALVING_CYCLE_DAYS = 4 * 365


class ExhaustiveFeatureEngine:
    """Calculate ALL features for exhaustive strategy testing."""

    def add_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all exhaustive features."""
        df = df.copy()

        # Ensure datetime column
        if 'datetime' not in df.columns and 'date' in df.columns:
            df['datetime'] = pd.to_datetime(df['date'])

        # Add features in order
        if 'close' in df.columns and df['close'].notna().any():
            df = self.add_streak_features(df)
            df = self.add_gap_features(df)
            df = self.add_range_features(df)
            df = self.add_extreme_features(df)
            df = self.add_pattern_features(df)
            df = self.add_volatility_regime_features(df)

        if 'datetime' in df.columns:
            df = self.add_halving_features(df)
            df = self.add_extended_calendar_features(df)

        if 'regime' in df.columns:
            df = self.add_regime_transition_features(df)

        return df

    def add_streak_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add consecutive up/down day features."""
        df = df.copy()

        if 'close' not in df.columns:
            return df

        # Daily returns
        df['daily_return'] = df['close'].pct_change()
        df['up_day'] = (df['daily_return'] > 0).astype(int)
        df['down_day'] = (df['daily_return'] < 0).astype(int)

        # Count consecutive days
        def count_streak(series, direction):
            """Count consecutive occurrences."""
            streak = pd.Series(index=series.index, dtype=int)
            count = 0
            for i in range(len(series)):
                if series.iloc[i] == direction:
                    count += 1
                else:
                    count = 0
                streak.iloc[i] = count
            return streak

        up_streak = count_streak(df['up_day'], 1)
        down_streak = count_streak(df['down_day'], 1)

        # Mark N consecutive days
        for n in [2, 3, 4, 5]:
            df[f'consecutive_up_{n}'] = (up_streak >= n).astype(int)
            df[f'consecutive_down_{n}'] = (down_streak >= n).astype(int)

        df['up_streak_count'] = up_streak
        df['down_streak_count'] = down_streak

        return df

    def add_gap_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add gap (open vs prev close) features."""
        df = df.copy()

        if 'open' not in df.columns or 'close' not in df.columns:
            # If no open, estimate from close
            if 'close' in df.columns:
                df['gap_pct'] = df['close'].pct_change() * 100
            return df

        prev_close = df['close'].shift(1)
        df['gap_pct'] = (df['open'] / prev_close - 1) * 100

        return df

    def add_range_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add intraday range features."""
        df = df.copy()

        if 'high' not in df.columns or 'low' not in df.columns:
            return df

        df['daily_range'] = df['high'] - df['low']
        df['daily_range_pct'] = df['daily_range'] / df['close'] * 100

        if 'atr' in df.columns:
            df['range_vs_atr'] = df['daily_range'] / df['atr'].replace(0, np.nan)
        else:
            # Calculate simple ATR
            df['true_range'] = np.maximum(
                df['high'] - df['low'],
                np.maximum(
                    abs(df['high'] - df['close'].shift(1)),
                    abs(df['low'] - df['close'].shift(1))
                )
            )
            atr = df['true_range'].rolling(14).mean()
            df['range_vs_atr'] = df['daily_range'] / atr.replace(0, np.nan)

        return df

    def add_extreme_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add new high/low features."""
        df = df.copy()

        if 'close' not in df.columns:
            return df

        for lookback in [20, 50, 100]:
            rolling_high = df['close'].rolling(lookback).max()
            rolling_low = df['close'].rolling(lookback).min()

            df[f'is_high_{lookback}'] = (df['close'] >= rolling_high).astype(int)
            df[f'is_low_{lookback}'] = (df['close'] <= rolling_low).astype(int)

            # Distance from high/low
            df[f'dist_from_high_{lookback}'] = (df['close'] / rolling_high - 1) * 100
            df[f'dist_from_low_{lookback}'] = (df['close'] / rolling_low - 1) * 100

        return df

    def add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add candlestick pattern features."""
        df = df.copy()

        required = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required):
            # Create dummy features
            df['is_inside_day'] = 0
            df['is_outside_day'] = 0
            df['is_doji'] = 0
            return df

        # Previous day's range
        prev_high = df['high'].shift(1)
        prev_low = df['low'].shift(1)

        # Inside day: today's range within yesterday's range
        df['is_inside_day'] = (
            (df['high'] <= prev_high) & (df['low'] >= prev_low)
        ).astype(int)

        # Outside day: today's range exceeds yesterday's range
        df['is_outside_day'] = (
            (df['high'] > prev_high) & (df['low'] < prev_low)
        ).astype(int)

        # Doji: open ~= close (small body)
        body = abs(df['close'] - df['open'])
        daily_range = df['high'] - df['low']
        df['is_doji'] = (body < 0.1 * daily_range).astype(int)

        # Body position in range
        df['body_position'] = (df['close'] - df['low']) / daily_range.replace(0, np.nan)

        return df

    def add_volatility_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility regime features."""
        df = df.copy()

        if 'close' not in df.columns:
            return df

        # Calculate rolling volatility
        returns = df['close'].pct_change()
        vol_20 = returns.rolling(20).std() * np.sqrt(365)  # Annualized
        vol_60 = returns.rolling(60).std() * np.sqrt(365)

        # Volatility percentile (expanding window)
        def rolling_percentile(series, window=252):
            """Calculate rolling percentile rank."""
            result = pd.Series(index=series.index, dtype=float)
            for i in range(window, len(series)):
                window_data = series.iloc[i-window:i]
                current = series.iloc[i]
                result.iloc[i] = (window_data < current).sum() / window * 100
            return result

        df['volatility_20d'] = vol_20
        df['volatility_60d'] = vol_60
        df['volatility_percentile'] = rolling_percentile(vol_20)

        # Volatility change ratio
        df['vol_change_ratio'] = vol_20 / vol_60.replace(0, np.nan)

        # High/low volatility flags
        df['vol_high'] = (df['volatility_percentile'] > 80).astype(int)
        df['vol_low'] = (df['volatility_percentile'] < 20).astype(int)

        return df

    def add_halving_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Bitcoin halving cycle features."""
        df = df.copy()

        if 'datetime' not in df.columns:
            return df

        df['datetime'] = pd.to_datetime(df['datetime'])

        def get_halving_features(dt):
            """Calculate halving-related features for a date."""
            if pd.isna(dt):
                return np.nan, np.nan, np.nan

            # Find nearest past and future halvings
            past_halvings = [h for h in HALVING_DATES if h <= dt]
            future_halvings = [h for h in HALVING_DATES if h > dt]

            days_since = (dt - past_halvings[-1]).days if past_halvings else np.nan
            days_until = (future_halvings[0] - dt).days if future_halvings else np.nan

            # Cycle phase (0 = just after halving, 1 = just before next)
            if past_halvings and future_halvings:
                cycle_length = (future_halvings[0] - past_halvings[-1]).days
                phase = days_since / cycle_length if cycle_length > 0 else 0
            elif days_since is not None:
                phase = min(days_since / HALVING_CYCLE_DAYS, 1.0)
            else:
                phase = np.nan

            return days_since, days_until, phase

        # Apply to all rows
        results = df['datetime'].apply(get_halving_features)
        df['days_since_halving'] = results.apply(lambda x: x[0])
        df['days_until_halving'] = results.apply(lambda x: x[1])
        df['halving_cycle_phase'] = results.apply(lambda x: x[2])

        # Halving proximity flags
        df['near_halving'] = (
            (df['days_until_halving'] < 90) | (df['days_since_halving'] < 90)
        ).astype(int)

        return df

    def add_extended_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add extended calendar features."""
        df = df.copy()

        if 'datetime' not in df.columns:
            return df

        dt = df['datetime']

        # Week of month (1-5)
        df['week_of_month'] = ((dt.dt.day - 1) // 7 + 1).clip(upper=5)

        # Day position in month (0-1)
        days_in_month = dt.dt.days_in_month
        df['day_position_in_month'] = dt.dt.day / days_in_month

        # Year fraction (0-1)
        df['year_fraction'] = (dt.dt.dayofyear - 1) / 365

        # Is end of year
        df['is_year_end'] = ((dt.dt.month == 12) & (dt.dt.day >= 20)).astype(int)
        df['is_year_start'] = ((dt.dt.month == 1) & (dt.dt.day <= 10)).astype(int)

        # Trading day of week (0-4 for Mon-Fri, excluding weekends)
        df['is_weekend'] = (dt.dt.dayofweek >= 5).astype(int)

        return df

    def add_regime_transition_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add regime transition features."""
        df = df.copy()

        if 'regime' not in df.columns:
            return df

        # Previous regime
        df['prev_regime'] = df['regime'].shift(1)

        # Is transition
        df['is_regime_change'] = (df['regime'] != df['prev_regime']).astype(int)

        # Days in current regime
        def count_days_in_regime(regime_series):
            """Count consecutive days in same regime."""
            days = pd.Series(index=regime_series.index, dtype=int)
            count = 1
            for i in range(len(regime_series)):
                if i == 0:
                    days.iloc[i] = 1
                elif regime_series.iloc[i] == regime_series.iloc[i-1]:
                    count += 1
                    days.iloc[i] = count
                else:
                    count = 1
                    days.iloc[i] = count
            return days

        df['days_in_regime'] = count_days_in_regime(df['regime'])

        return df


def add_blockchain_extended_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add extended blockchain features if available."""
    df = df.copy()

    # Difficulty change (requires difficulty column)
    if 'difficulty' in df.columns:
        df['difficulty_change'] = df['difficulty'].pct_change()
        df['difficulty_change_7d'] = df['difficulty'].pct_change(7)

    # Block time (requires block_time column)
    if 'avg_block_time' not in df.columns and 'block_count' in df.columns:
        # Estimate: 144 blocks per day at 10 min each
        df['avg_block_time'] = 1440 / df['block_count'].replace(0, np.nan)  # minutes

    # Hash rate change (requires hashrate column)
    if 'hashrate' in df.columns:
        df['hashrate_change_pct'] = df['hashrate'].pct_change() * 100
        df['hashrate_change_7d_pct'] = df['hashrate'].pct_change(7) * 100

    # Fee z-score (requires avg_fee column)
    if 'avg_fee' in df.columns:
        mean = df['avg_fee'].rolling(30).mean()
        std = df['avg_fee'].rolling(30).std()
        df['fee_zscore'] = (df['avg_fee'] - mean) / std.replace(0, np.nan)

    return df


def quick_test():
    """Quick test of exhaustive features."""
    from .data_loader import RentechDataLoader
    from .feature_engine import FeatureEngine
    from .advanced_features import AdvancedFeatureEngine

    print("Testing Exhaustive Features...")

    # Load data
    loader = RentechDataLoader()
    df = loader.load_merged_data()

    # Add basic features
    basic = FeatureEngine()
    df = basic.add_all_features(df)

    # Add advanced features
    advanced = AdvancedFeatureEngine()
    df = advanced.add_all_features(df)

    # Add exhaustive features
    exhaustive = ExhaustiveFeatureEngine()
    df = exhaustive.add_all_features(df)

    print(f"\nDataFrame shape: {df.shape}")
    print(f"Total columns: {len(df.columns)}")

    # Show new columns
    new_cols = [c for c in df.columns if any(
        x in c for x in ['consecutive', 'gap', 'range_vs', 'is_high', 'is_low',
                        'inside', 'outside', 'doji', 'halving', 'week_of_month',
                        'volatility_percentile', 'regime_change', 'days_in']
    )]
    print(f"\nExhaustive feature columns ({len(new_cols)}):")
    for col in new_cols[:25]:
        non_null = df[col].notna().sum()
        print(f"  {col}: {non_null:,} non-null values")

    # Show halving features
    print("\nHalving Features (sample):")
    sample = df[['date', 'days_since_halving', 'days_until_halving',
                 'halving_cycle_phase']].dropna().tail(10)
    print(sample.to_string(index=False))


if __name__ == "__main__":
    quick_test()
