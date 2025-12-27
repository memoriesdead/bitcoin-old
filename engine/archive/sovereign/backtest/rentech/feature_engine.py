"""
Feature Engine - Calculate z-scores, momentum, and regime detection

Transforms raw daily features into trading signals using:
- Rolling z-scores (mean reversion detection)
- Momentum calculations (trend following)
- Regime detection (market state classification)
"""
import pandas as pd
import numpy as np
from typing import List, Optional, Dict
from dataclasses import dataclass


@dataclass
class FeatureConfig:
    """Configuration for feature calculation."""
    zscore_windows: List[int] = None
    momentum_periods: List[int] = None
    volatility_window: int = 20

    def __post_init__(self):
        if self.zscore_windows is None:
            self.zscore_windows = [7, 14, 30, 60]
        if self.momentum_periods is None:
            self.momentum_periods = [1, 3, 5, 10]


class FeatureEngine:
    """
    Calculate features for backtesting.

    Features:
    - Z-scores: Deviation from rolling mean (mean reversion signals)
    - Momentum: Rate of change over periods
    - Volatility: Rolling standard deviation
    - Regimes: Market state classification (5 states)
    """

    # Regime definitions based on z-score combinations
    REGIMES = {
        'ACCUMULATION': 'whale_activity high + low volatility',
        'DISTRIBUTION': 'whale_activity high + rising volatility',
        'NEUTRAL': 'normal activity',
        'CAPITULATION': 'extreme selling + high volatility',
        'EUPHORIA': 'extreme buying + high volatility',
    }

    def __init__(self, config: FeatureConfig = None):
        self.config = config or FeatureConfig()

    def add_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all features to DataFrame."""
        df = df.copy()

        # 1. Z-scores for available columns
        zscore_cols = ['tx_count']

        # Add ORBITAAL-specific columns if available
        if 'whale_tx_count' in df.columns and df['whale_tx_count'].notna().any():
            zscore_cols.append('whale_tx_count')
        if 'total_value_btc' in df.columns and df['total_value_btc'].notna().any():
            zscore_cols.append('total_value_btc')
        if 'unique_senders' in df.columns and df['unique_senders'].notna().any():
            zscore_cols.append('unique_senders')

        # Add block-specific columns if available
        if 'avg_block_fullness' in df.columns and df['avg_block_fullness'].notna().any():
            zscore_cols.append('avg_block_fullness')

        df = self.add_zscores(df, zscore_cols)

        # 2. Momentum (requires price data)
        if 'close' in df.columns:
            df = self.add_momentum(df)
            df = self.add_volatility(df)

        # 3. Regime detection
        df = self.detect_regimes(df)

        return df

    def add_zscores(
        self,
        df: pd.DataFrame,
        columns: List[str],
        windows: List[int] = None
    ) -> pd.DataFrame:
        """
        Add z-score columns for specified features.

        Z-score = (value - rolling_mean) / rolling_std

        Args:
            df: Input DataFrame
            columns: Columns to calculate z-scores for
            windows: Lookback windows (default from config)

        Returns:
            DataFrame with new z-score columns
        """
        df = df.copy()
        windows = windows or self.config.zscore_windows

        for col in columns:
            if col not in df.columns:
                continue

            # Ensure numeric type
            try:
                col_data = pd.to_numeric(df[col], errors='coerce')
            except Exception:
                continue

            for window in windows:
                col_name = f'{col}_zscore_{window}d'

                rolling_mean = col_data.rolling(window=window, min_periods=window//2).mean()
                rolling_std = col_data.rolling(window=window, min_periods=window//2).std()

                # Avoid division by zero
                rolling_std = rolling_std.replace(0, np.nan)

                df[col_name] = (col_data - rolling_mean) / rolling_std

        # Create a "main" z-score using 30-day window (most common)
        for col in columns:
            if col not in df.columns:
                continue
            main_zscore = f'{col}_zscore_30d'
            if main_zscore in df.columns:
                df[f'{col}_zscore'] = df[main_zscore]

        return df

    def add_momentum(
        self,
        df: pd.DataFrame,
        periods: List[int] = None
    ) -> pd.DataFrame:
        """
        Add momentum (return) columns.

        Momentum = price_t / price_(t-n) - 1

        Args:
            df: Input DataFrame with 'close' column
            periods: Lookback periods in days

        Returns:
            DataFrame with momentum columns
        """
        df = df.copy()
        periods = periods or self.config.momentum_periods

        if 'close' not in df.columns:
            return df

        for period in periods:
            df[f'momentum_{period}d'] = df['close'].pct_change(period)

        return df

    def add_volatility(
        self,
        df: pd.DataFrame,
        window: int = None
    ) -> pd.DataFrame:
        """
        Add rolling volatility.

        Volatility = rolling std of daily returns

        Args:
            df: Input DataFrame with 'close' column
            window: Rolling window (default 20)

        Returns:
            DataFrame with volatility column
        """
        df = df.copy()
        window = window or self.config.volatility_window

        if 'close' not in df.columns:
            return df

        daily_returns = df['close'].pct_change()
        df['volatility'] = daily_returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
        df['volatility_zscore'] = (
            (df['volatility'] - df['volatility'].rolling(60).mean()) /
            df['volatility'].rolling(60).std()
        )

        return df

    def detect_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect market regimes based on feature combinations.

        Regimes:
        - ACCUMULATION: High whale activity, low volatility (bullish setup)
        - DISTRIBUTION: High whale activity, high volatility (bearish setup)
        - NEUTRAL: Normal activity
        - CAPITULATION: Extreme negative momentum + high volatility
        - EUPHORIA: Extreme positive momentum + high volatility

        Returns:
            DataFrame with 'regime' column
        """
        df = df.copy()
        df['regime'] = 'NEUTRAL'

        # Get available z-scores
        tx_zscore = df.get('tx_count_zscore', pd.Series(0, index=df.index))
        whale_zscore = df.get('whale_tx_count_zscore', pd.Series(0, index=df.index))
        vol_zscore = df.get('volatility_zscore', pd.Series(0, index=df.index))
        mom_5d = df.get('momentum_5d', pd.Series(0, index=df.index))

        # Fill NaN with 0 for regime detection
        tx_zscore = tx_zscore.fillna(0)
        whale_zscore = whale_zscore.fillna(0)
        vol_zscore = vol_zscore.fillna(0)
        mom_5d = mom_5d.fillna(0)

        # ACCUMULATION: High activity + low volatility + slight positive momentum
        accumulation = (
            ((whale_zscore > 1.0) | (tx_zscore > 1.0)) &
            (vol_zscore < 0.5) &
            (mom_5d > -0.05)
        )
        df.loc[accumulation, 'regime'] = 'ACCUMULATION'

        # DISTRIBUTION: High activity + high/rising volatility
        distribution = (
            ((whale_zscore > 1.0) | (tx_zscore > 1.5)) &
            (vol_zscore > 1.0)
        )
        df.loc[distribution, 'regime'] = 'DISTRIBUTION'

        # CAPITULATION: Extreme negative momentum + very high volatility
        capitulation = (
            (mom_5d < -0.15) &
            (vol_zscore > 1.5)
        )
        df.loc[capitulation, 'regime'] = 'CAPITULATION'

        # EUPHORIA: Extreme positive momentum + very high volatility
        euphoria = (
            (mom_5d > 0.15) &
            (vol_zscore > 1.5)
        )
        df.loc[euphoria, 'regime'] = 'EUPHORIA'

        # Encode as numeric for modeling
        regime_map = {
            'ACCUMULATION': 0,
            'DISTRIBUTION': 1,
            'NEUTRAL': 2,
            'CAPITULATION': 3,
            'EUPHORIA': 4,
        }
        df['regime_code'] = df['regime'].map(regime_map)

        return df

    def get_feature_summary(self, df: pd.DataFrame) -> Dict:
        """Get summary statistics for all features."""
        summary = {}

        # Find all z-score columns
        zscore_cols = [c for c in df.columns if 'zscore' in c.lower()]

        for col in zscore_cols:
            summary[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'pct_above_2': (df[col] > 2).mean() * 100,
                'pct_below_minus_2': (df[col] < -2).mean() * 100,
            }

        # Regime distribution
        if 'regime' in df.columns:
            summary['regimes'] = df['regime'].value_counts().to_dict()

        return summary


def quick_test():
    """Quick test of feature engine."""
    from .data_loader import RentechDataLoader

    print("Testing Feature Engine...")

    loader = RentechDataLoader()
    df = loader.load_merged_data()

    engine = FeatureEngine()
    df = engine.add_all_features(df)

    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Show feature summary
    summary = engine.get_feature_summary(df)
    print("\nFeature Summary:")
    for feat, stats in summary.items():
        if feat != 'regimes':
            print(f"  {feat}: mean={stats['mean']:.2f}, std={stats['std']:.2f}")

    if 'regimes' in summary:
        print("\nRegime Distribution:")
        for regime, count in summary['regimes'].items():
            print(f"  {regime}: {count:,} days ({count/len(df)*100:.1f}%)")

    # Show sample with features
    print("\nSample data with features:")
    cols = ['date', 'tx_count', 'tx_count_zscore', 'regime', 'close']
    cols = [c for c in cols if c in df.columns]
    print(df[cols].dropna().tail(10).to_string())


if __name__ == "__main__":
    quick_test()
