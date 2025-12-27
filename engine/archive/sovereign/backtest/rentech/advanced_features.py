"""
Advanced Feature Engineering - RenTech-Style Indicators
========================================================

Calculates comprehensive features for backtesting:
- Technical indicators (RSI, MACD, Bollinger Bands)
- Volume patterns
- Volatility measures
- HMM regime probabilities
- Cross-correlations
- Seasonality encodings
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class AdvancedFeatureConfig:
    """Configuration for advanced features."""
    # RSI
    rsi_periods: List[int] = None
    # MACD
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    # Bollinger Bands
    bb_window: int = 20
    bb_std: float = 2.0
    # ATR
    atr_window: int = 14
    # Volume
    volume_windows: List[int] = None
    # Momentum
    roc_periods: List[int] = None

    def __post_init__(self):
        if self.rsi_periods is None:
            self.rsi_periods = [7, 14, 21]
        if self.volume_windows is None:
            self.volume_windows = [5, 10, 20]
        if self.roc_periods is None:
            self.roc_periods = [1, 3, 5, 10, 20]


class AdvancedFeatureEngine:
    """
    Calculate advanced technical and blockchain features.

    Features calculated:
    - RSI (multiple periods)
    - MACD (histogram, signal)
    - Bollinger Bands (position, width)
    - ATR (Average True Range)
    - Volume z-scores
    - Rate of Change
    - Moving Average crossovers
    - Seasonality encodings
    - HMM regime features
    """

    def __init__(self, config: AdvancedFeatureConfig = None):
        self.config = config or AdvancedFeatureConfig()

    def add_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all advanced features to DataFrame."""
        df = df.copy()

        # Only calculate if we have price data
        if 'close' in df.columns and df['close'].notna().any():
            df = self.add_rsi(df)
            df = self.add_macd(df)
            df = self.add_bollinger_bands(df)
            df = self.add_atr(df)
            df = self.add_roc(df)
            df = self.add_ma_crossovers(df)

        # Volume features (from blockchain data)
        if 'volume' in df.columns:
            df = self.add_volume_features(df)

        # Seasonality
        df = self.add_seasonality_features(df)

        # Cross-correlations with TX data
        if 'tx_count' in df.columns and 'close' in df.columns:
            df = self.add_tx_price_correlation(df)

        return df

    def add_rsi(self, df: pd.DataFrame, column: str = 'close') -> pd.DataFrame:
        """
        Add RSI (Relative Strength Index).

        RSI = 100 - (100 / (1 + RS))
        RS = avg_gain / avg_loss
        """
        df = df.copy()

        if column not in df.columns:
            return df

        delta = df[column].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        for period in self.config.rsi_periods:
            avg_gain = gain.rolling(window=period, min_periods=period//2).mean()
            avg_loss = loss.rolling(window=period, min_periods=period//2).mean()

            rs = avg_gain / avg_loss.replace(0, np.nan)
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        return df

    def add_macd(self, df: pd.DataFrame, column: str = 'close') -> pd.DataFrame:
        """
        Add MACD (Moving Average Convergence Divergence).

        MACD = EMA_fast - EMA_slow
        Signal = EMA(MACD, signal_period)
        Histogram = MACD - Signal
        """
        df = df.copy()

        if column not in df.columns:
            return df

        ema_fast = df[column].ewm(span=self.config.macd_fast, adjust=False).mean()
        ema_slow = df[column].ewm(span=self.config.macd_slow, adjust=False).mean()

        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=self.config.macd_signal, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        # Normalize
        df['macd_zscore'] = (df['macd'] - df['macd'].rolling(30).mean()) / df['macd'].rolling(30).std()

        return df

    def add_bollinger_bands(self, df: pd.DataFrame, column: str = 'close') -> pd.DataFrame:
        """
        Add Bollinger Bands.

        BB_upper = SMA + k*std
        BB_lower = SMA - k*std
        BB_position = (price - lower) / (upper - lower)
        BB_width = (upper - lower) / SMA
        """
        df = df.copy()

        if column not in df.columns:
            return df

        sma = df[column].rolling(window=self.config.bb_window).mean()
        std = df[column].rolling(window=self.config.bb_window).std()

        df['bb_upper'] = sma + self.config.bb_std * std
        df['bb_lower'] = sma - self.config.bb_std * std
        df['bb_middle'] = sma

        # Position within bands (0 to 1, can exceed)
        df['bb_position'] = (df[column] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # Bandwidth (volatility indicator)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / sma

        return df

    def add_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add ATR (Average True Range).

        TR = max(high-low, |high-prev_close|, |low-prev_close|)
        ATR = EMA(TR, period)
        """
        df = df.copy()

        required = ['high', 'low', 'close']
        if not all(c in df.columns for c in required):
            return df

        prev_close = df['close'].shift(1)

        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - prev_close)
        tr3 = abs(df['low'] - prev_close)

        df['true_range'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = df['true_range'].ewm(span=self.config.atr_window, adjust=False).mean()

        # ATR as percentage of price
        df['atr_pct'] = df['atr'] / df['close'] * 100

        # ATR z-score (volatility expansion/contraction)
        df['atr_zscore'] = (df['atr'] - df['atr'].rolling(30).mean()) / df['atr'].rolling(30).std()

        return df

    def add_roc(self, df: pd.DataFrame, column: str = 'close') -> pd.DataFrame:
        """
        Add Rate of Change.

        ROC = (price_t / price_{t-n} - 1) * 100
        """
        df = df.copy()

        if column not in df.columns:
            return df

        for period in self.config.roc_periods:
            df[f'roc_{period}'] = df[column].pct_change(period) * 100

        return df

    def add_ma_crossovers(self, df: pd.DataFrame, column: str = 'close') -> pd.DataFrame:
        """
        Add Moving Average crossover signals.

        Signals:
        - Golden cross (fast > slow)
        - Death cross (fast < slow)
        - Price vs MAs
        """
        df = df.copy()

        if column not in df.columns:
            return df

        # Calculate MAs
        df['sma_10'] = df[column].rolling(10).mean()
        df['sma_20'] = df[column].rolling(20).mean()
        df['sma_50'] = df[column].rolling(50).mean()
        df['ema_10'] = df[column].ewm(span=10, adjust=False).mean()
        df['ema_20'] = df[column].ewm(span=20, adjust=False).mean()

        # Crossover signals
        df['ma_cross_10_20'] = (df['sma_10'] > df['sma_20']).astype(int)
        df['ma_cross_20_50'] = (df['sma_20'] > df['sma_50']).astype(int)

        # Price position relative to MAs
        df['price_vs_sma20'] = (df[column] / df['sma_20'] - 1) * 100
        df['price_vs_sma50'] = (df[column] / df['sma_50'] - 1) * 100

        return df

    def add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        df = df.copy()

        if 'volume' not in df.columns:
            return df

        vol = pd.to_numeric(df['volume'], errors='coerce')

        # Volume z-scores
        for window in self.config.volume_windows:
            mean = vol.rolling(window).mean()
            std = vol.rolling(window).std()
            df[f'volume_zscore_{window}'] = (vol - mean) / std.replace(0, np.nan)

        # Volume moving averages
        df['volume_sma_10'] = vol.rolling(10).mean()
        df['volume_sma_20'] = vol.rolling(20).mean()

        # Volume ratio
        df['volume_ratio'] = vol / df['volume_sma_20']

        # On-balance volume proxy (cumulative directional volume)
        if 'close' in df.columns:
            direction = np.sign(df['close'].diff())
            df['obv_proxy'] = (direction * vol).cumsum()

        return df

    def add_seasonality_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add seasonality encodings."""
        df = df.copy()

        if 'datetime' not in df.columns and 'date' in df.columns:
            df['datetime'] = pd.to_datetime(df['date'])

        if 'datetime' in df.columns:
            # Day of week (0=Monday, 6=Sunday)
            df['dow'] = df['datetime'].dt.dayofweek

            # Day of week one-hot
            for i in range(7):
                df[f'dow_{i}'] = (df['dow'] == i).astype(int)

            # Month
            df['month'] = df['datetime'].dt.month

            # Month one-hot
            for i in range(1, 13):
                df[f'month_{i}'] = (df['month'] == i).astype(int)

            # Quarter
            df['quarter'] = df['datetime'].dt.quarter

            # Day of month (1-31)
            df['dom'] = df['datetime'].dt.day

            # Week of year
            df['woy'] = df['datetime'].dt.isocalendar().week

            # Is month start/end
            df['is_month_start'] = df['datetime'].dt.is_month_start.astype(int)
            df['is_month_end'] = df['datetime'].dt.is_month_end.astype(int)

            # Is quarter start/end
            df['is_quarter_start'] = df['datetime'].dt.is_quarter_start.astype(int)
            df['is_quarter_end'] = df['datetime'].dt.is_quarter_end.astype(int)

        return df

    def add_tx_price_correlation(self, df: pd.DataFrame, windows: List[int] = None) -> pd.DataFrame:
        """Add TX count and price correlation features."""
        df = df.copy()

        windows = windows or [10, 20, 30]

        if 'tx_count' not in df.columns or 'close' not in df.columns:
            return df

        tx = pd.to_numeric(df['tx_count'], errors='coerce')
        price = df['close']

        for window in windows:
            # Rolling correlation
            df[f'tx_price_corr_{window}'] = tx.rolling(window).corr(price)

            # TX leads price?
            tx_return = tx.pct_change()
            price_return = price.pct_change().shift(-1)  # Future price return
            df[f'tx_lead_corr_{window}'] = tx_return.rolling(window).corr(price_return.shift(1))

        return df


def quick_test():
    """Quick test of advanced features."""
    from .data_loader import RentechDataLoader
    from .feature_engine import FeatureEngine

    print("Testing Advanced Features...")

    loader = RentechDataLoader()
    df = loader.load_merged_data()

    # Basic features
    basic_engine = FeatureEngine()
    df = basic_engine.add_all_features(df)

    # Advanced features
    advanced_engine = AdvancedFeatureEngine()
    df = advanced_engine.add_all_features(df)

    print(f"\nDataFrame shape: {df.shape}")
    print(f"Total columns: {len(df.columns)}")

    # Show new columns
    advanced_cols = [c for c in df.columns if any(
        x in c for x in ['rsi', 'macd', 'bb_', 'atr', 'roc_', 'sma_', 'ema_', 'volume_', 'dow_', 'month_']
    )]
    print(f"\nAdvanced feature columns ({len(advanced_cols)}):")
    for col in advanced_cols[:20]:
        non_null = df[col].notna().sum()
        print(f"  {col}: {non_null:,} non-null values")


if __name__ == "__main__":
    quick_test()
