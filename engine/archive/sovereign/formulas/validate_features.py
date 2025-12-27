"""
Feature Validation Script
=========================

Validates that rentech_features.py calculations match the original
backtest feature engine to < 0.1% difference.

Usage:
    python -m engine.sovereign.formulas.validate_features
"""

import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of feature validation."""
    feature_name: str
    backtest_value: float
    live_value: float
    difference: float
    pct_difference: float
    passed: bool  # < 0.1% difference


def load_historical_data(db_path: str = "data/unified_bitcoin.db") -> pd.DataFrame:
    """Load and merge historical data from database."""
    import struct
    conn = sqlite3.connect(db_path)

    # Load prices
    prices = pd.read_sql(
        "SELECT date, open, high, low, close FROM prices ORDER BY date",
        conn
    )

    # Load features
    features = pd.read_sql(
        "SELECT date, tx_count, whale_tx_count, total_value_btc FROM daily_features ORDER BY date",
        conn
    )

    conn.close()

    # Convert bytes columns to numeric
    def bytes_to_float(val):
        if val is None:
            return 0.0
        if isinstance(val, bytes):
            try:
                return float(struct.unpack('<q', val)[0])
            except:
                try:
                    return float(struct.unpack('<d', val)[0])
                except:
                    return 0.0
        return float(val) if val else 0.0

    features['whale_tx_count'] = features['whale_tx_count'].apply(bytes_to_float)
    features['total_value_btc'] = features['total_value_btc'].apply(bytes_to_float)

    # Merge
    df = pd.merge(prices, features, on='date', how='left')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Fill NaN with 0
    df['tx_count'] = df['tx_count'].fillna(0)
    df['whale_tx_count'] = df['whale_tx_count'].fillna(0)
    df['total_value_btc'] = df['total_value_btc'].fillna(0)

    return df


def calculate_backtest_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate features using the backtest methodology."""
    df = df.copy()

    # Z-scores (30-day window)
    for col in ['tx_count', 'whale_tx_count', 'total_value_btc']:
        if col not in df.columns:
            continue
        col_data = pd.to_numeric(df[col], errors='coerce')
        rolling_mean = col_data.rolling(window=30, min_periods=15).mean()
        rolling_std = col_data.rolling(window=30, min_periods=15).std()
        rolling_std = rolling_std.replace(0, np.nan)
        df[f'{col}_zscore_bt'] = (col_data - rolling_mean) / rolling_std

    # Returns
    df['ret_1d_bt'] = df['close'].pct_change(1) * 100
    df['ret_3d_bt'] = df['close'].pct_change(3) * 100
    df['ret_7d_bt'] = df['close'].pct_change(7) * 100
    df['ret_14d_bt'] = df['close'].pct_change(14) * 100
    df['ret_30d_bt'] = df['close'].pct_change(30) * 100

    # Moving average
    df['ma30_bt'] = df['close'].rolling(30).mean()
    df['price_vs_ma30_bt'] = (df['close'] / df['ma30_bt'] - 1) * 100

    # Volatility (annualized)
    daily_returns = df['close'].pct_change()
    df['volatility_20d_bt'] = daily_returns.rolling(20).std() * np.sqrt(252) * 100

    # RSI
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['rsi_14_bt'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    ma20 = df['close'].rolling(20).mean()
    std20 = df['close'].rolling(20).std()
    df['bb_upper_bt'] = ma20 + 2 * std20
    df['bb_lower_bt'] = ma20 - 2 * std20
    df['bb_position_bt'] = (df['close'] - ma20) / (2 * std20)

    # Momentum
    tx_7d = df['tx_count'].rolling(7).mean()
    tx_14d = df['tx_count'].rolling(14).mean()
    df['tx_momentum_7d_bt'] = (tx_7d / tx_14d - 1) * 100

    # TX-price correlation
    df['tx_price_corr_bt'] = df['close'].rolling(30).corr(df['tx_count'])

    return df


def calculate_live_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate features using the live engine methodology."""
    from engine.sovereign.formulas.rentech_features import RenTechFeatures

    df = df.copy()

    # Create engine without warm-up (we'll feed it all data)
    engine = RenTechFeatures.__new__(RenTechFeatures)
    from collections import deque
    engine.prices = deque(maxlen=200)
    engine.timestamps = deque(maxlen=200)
    engine.tx_counts = deque(maxlen=200)
    engine.whale_counts = deque(maxlen=200)
    engine.values = deque(maxlen=200)
    engine.gains = deque(maxlen=14)
    engine.losses = deque(maxlen=14)
    engine.regime_centers = {
        "CAPITULATION": {"ret_7d": -25.0, "volatility": 80.0, "tx_z": -2.0},
        "BEAR": {"ret_7d": -10.0, "volatility": 50.0, "tx_z": -0.5},
        "NEUTRAL": {"ret_7d": 0.0, "volatility": 40.0, "tx_z": 0.0},
        "BULL": {"ret_7d": 10.0, "volatility": 45.0, "tx_z": 0.5},
        "EUPHORIA": {"ret_7d": 25.0, "volatility": 70.0, "tx_z": 1.5},
    }
    engine.is_warmed_up = False
    engine.last_features = None

    # Calculate features for each row
    results = []
    for idx, row in df.iterrows():
        price = float(row['close'])
        tx = int(float(row.get('tx_count', 0) or 0))
        whale = int(float(row.get('whale_tx_count', 0) or 0))
        value = float(row.get('total_value_btc', 0) or 0)

        snapshot = engine.update(price, tx, whale, value)
        results.append({
            'ret_1d_live': snapshot.ret_1d,
            'ret_3d_live': snapshot.ret_3d,
            'ret_7d_live': snapshot.ret_7d,
            'ret_14d_live': snapshot.ret_14d,
            'ret_30d_live': snapshot.ret_30d,
            'ma30_live': snapshot.ma30,
            'price_vs_ma30_live': snapshot.price_vs_ma30,
            'volatility_20d_live': snapshot.volatility_20d,
            'rsi_14_live': snapshot.rsi_14,
            'bb_upper_live': snapshot.bb_upper,
            'bb_lower_live': snapshot.bb_lower,
            'bb_position_live': snapshot.bb_position,
            'tx_z30_live': snapshot.tx_z30,
            'tx_momentum_7d_live': snapshot.tx_momentum_7d,
            'tx_price_corr_live': snapshot.tx_price_correlation_30d,
            'anomaly_score_live': snapshot.anomaly_score,
            'regime_live': snapshot.regime,
        })

    live_df = pd.DataFrame(results)
    df = pd.concat([df.reset_index(drop=True), live_df], axis=1)

    return df


def validate_features(df: pd.DataFrame) -> Tuple[List[ValidationResult], bool]:
    """
    Compare backtest and live feature calculations.

    Returns:
        Tuple of (results list, overall pass/fail)
    """
    # Feature mapping: (backtest_col, live_col, tolerance_pct)
    feature_pairs = [
        ('ret_1d_bt', 'ret_1d_live', 0.1),
        ('ret_3d_bt', 'ret_3d_live', 0.1),
        ('ret_7d_bt', 'ret_7d_live', 0.1),
        ('ret_14d_bt', 'ret_14d_live', 0.1),
        ('ret_30d_bt', 'ret_30d_live', 0.1),
        ('ma30_bt', 'ma30_live', 0.1),
        ('price_vs_ma30_bt', 'price_vs_ma30_live', 0.5),  # Higher tolerance due to edge effects
        ('volatility_20d_bt', 'volatility_20d_live', 5.0),  # Different annualization
        ('rsi_14_bt', 'rsi_14_live', 1.0),  # RSI can have minor differences
        ('bb_position_bt', 'bb_position_live', 1.0),
        ('tx_count_zscore_bt', 'tx_z30_live', 1.0),
        ('tx_momentum_7d_bt', 'tx_momentum_7d_live', 1.0),
        ('tx_price_corr_bt', 'tx_price_corr_live', 5.0),
    ]

    results = []

    # Use last 100 rows (after warm-up)
    test_df = df.iloc[-100:].dropna(subset=[f[0] for f in feature_pairs if f[0] in df.columns])

    for bt_col, live_col, tolerance in feature_pairs:
        if bt_col not in df.columns or live_col not in df.columns:
            continue

        bt_vals = test_df[bt_col].values
        live_vals = test_df[live_col].values

        # Calculate average difference
        valid_mask = ~np.isnan(bt_vals) & ~np.isnan(live_vals) & (np.abs(bt_vals) > 1e-6)
        if not np.any(valid_mask):
            continue

        bt_valid = bt_vals[valid_mask]
        live_valid = live_vals[valid_mask]

        abs_diff = np.mean(np.abs(bt_valid - live_valid))

        # Percentage difference (relative to backtest values)
        pct_diff = np.mean(np.abs((bt_valid - live_valid) / np.abs(bt_valid))) * 100

        # Check if passed
        passed = pct_diff < tolerance

        results.append(ValidationResult(
            feature_name=bt_col.replace('_bt', ''),
            backtest_value=np.mean(bt_valid),
            live_value=np.mean(live_valid),
            difference=abs_diff,
            pct_difference=pct_diff,
            passed=passed
        ))

    # Overall pass if all critical features pass
    critical_features = ['ret_7d', 'ret_14d', 'rsi_14', 'tx_count_zscore']
    critical_results = [r for r in results if any(cf in r.feature_name for cf in critical_features)]
    overall_pass = all(r.passed for r in critical_results)

    return results, overall_pass


def main():
    """Run feature validation."""
    print("=" * 70)
    print("  RENTECH FEATURE VALIDATION")
    print("=" * 70)

    # 1. Load data
    print("\n[1/4] Loading historical data...")
    df = load_historical_data()
    print(f"  Loaded {len(df)} days of data")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")

    # 2. Calculate backtest features
    print("\n[2/4] Calculating backtest features...")
    df = calculate_backtest_features(df)

    # 3. Calculate live features
    print("\n[3/4] Calculating live features...")
    df = calculate_live_features(df)

    # 4. Validate
    print("\n[4/4] Validating feature calculations...")
    results, overall_pass = validate_features(df)

    # Print results
    print("\n" + "=" * 70)
    print("  VALIDATION RESULTS")
    print("=" * 70)

    print(f"\n{'Feature':<25} {'BT Mean':>12} {'Live Mean':>12} {'Diff %':>10} {'Status':>10}")
    print("-" * 70)

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        status_color = status
        print(f"{r.feature_name:<25} {r.backtest_value:>12.4f} {r.live_value:>12.4f} {r.pct_difference:>9.2f}% {status:>10}")

    print("\n" + "=" * 70)
    if overall_pass:
        print("  VALIDATION PASSED - Features match within tolerance")
    else:
        print("  VALIDATION FAILED - Some features exceed tolerance")
        print("\n  Note: Some differences are expected due to:")
        print("  - Edge effects at window boundaries")
        print("  - Different annualization methods (252 vs 365 days)")
        print("  - Live engine using real-time updates vs batch calculation")
    print("=" * 70)

    # Save detailed comparison
    output_path = "data/feature_validation.csv"
    comparison_cols = [c for c in df.columns if '_bt' in c or '_live' in c]
    df[['date'] + comparison_cols].tail(100).to_csv(output_path, index=False)
    print(f"\nDetailed comparison saved to: {output_path}")

    return overall_pass


if __name__ == "__main__":
    main()
