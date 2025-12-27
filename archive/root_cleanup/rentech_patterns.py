#!/usr/bin/env python3
"""
RENAISSANCE-STYLE PATTERN RECOGNITION
======================================
Advanced techniques:
1. Kernel Methods - higher dimensional feature space
2. Polynomial Features - capture non-linear relationships
3. Anomaly Detection - statistical outliers
4. Multi-Scale Analysis - patterns at different timeframes
5. Cross-Correlation - relationships between metrics
6. Hidden Markov Models - regime detection
7. Mean Reversion + Momentum - convergence trades
"""

import sqlite3
import numpy as np
import struct
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cluster import KMeans
from itertools import combinations

print("="*80)
print("  RENAISSANCE-STYLE PATTERN RECOGNITION")
print("  Advanced Multi-Dimensional Analysis")
print("="*80)

# Load data
conn = sqlite3.connect('data/unified_bitcoin.db')
cursor = conn.execute('SELECT date, close FROM prices ORDER BY date')
prices = {row[0]: row[1] for row in cursor.fetchall()}

cursor = conn.execute('SELECT date, tx_count, whale_tx_count, total_value_btc FROM daily_features ORDER BY date')
raw = []
for row in cursor.fetchall():
    whale = row[2]
    if isinstance(whale, bytes):
        try: whale = struct.unpack('<q', whale)[0]
        except: whale = 0
    elif whale is None: whale = 0
    raw.append({'date': row[0], 'tx': row[1] or 0, 'whale': whale, 'value': row[3] or 0})
conn.close()

data = []
for f in raw:
    if f['date'] in prices and prices[f['date']] > 0:
        data.append({**f, 'price': prices[f['date']]})

print(f"Loaded {len(data)} records")

# =============================================================================
# FEATURE ENGINEERING - RenTech Style
# =============================================================================
print("\n[1] FEATURE ENGINEERING...")

for i, d in enumerate(data):
    # Multi-scale z-scores (7, 14, 30, 60 day windows)
    for window in [7, 14, 30, 60]:
        start = max(0, i - window)
        for metric in ['tx', 'whale', 'value', 'price']:
            vals = [data[j][metric] for j in range(start, i+1)]
            mean, std = np.mean(vals), np.std(vals)
            d[f'{metric}_z{window}'] = (d[metric] - mean) / std if std > 0 else 0

    # Returns at multiple scales
    for lb in [1, 3, 5, 7, 14, 21, 30, 60]:
        d[f'ret_{lb}d'] = (d['price'] / data[i-lb]['price'] - 1) * 100 if i >= lb else 0

    # Volatility (rolling std of returns)
    if i >= 20:
        rets = [(data[j]['price'] / data[j-1]['price'] - 1) for j in range(i-19, i+1)]
        d['volatility_20d'] = np.std(rets) * np.sqrt(252) * 100  # Annualized
    else:
        d['volatility_20d'] = 0

    # RSI (14-day)
    if i >= 14:
        gains, losses = [], []
        for j in range(i-13, i+1):
            change = data[j]['price'] - data[j-1]['price']
            gains.append(max(0, change))
            losses.append(max(0, -change))
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        d['rsi_14'] = 100 - (100 / (1 + avg_gain/avg_loss)) if avg_loss > 0 else 100
    else:
        d['rsi_14'] = 50

    # Moving average crossovers
    for short, long in [(5, 20), (10, 30), (20, 50)]:
        if i >= long:
            ma_short = np.mean([data[j]['price'] for j in range(i-short, i)])
            ma_long = np.mean([data[j]['price'] for j in range(i-long, i)])
            d[f'ma_{short}_{long}_cross'] = (ma_short / ma_long - 1) * 100
        else:
            d[f'ma_{short}_{long}_cross'] = 0

    # Price vs various MAs
    for ma_len in [10, 20, 30, 50, 100, 200]:
        if i >= ma_len:
            ma = np.mean([data[j]['price'] for j in range(i-ma_len, i)])
            d[f'price_vs_ma{ma_len}'] = (d['price'] / ma - 1) * 100
        else:
            d[f'price_vs_ma{ma_len}'] = 0

    # On-chain momentum (tx change rate)
    if i >= 7:
        tx_now = np.mean([data[j]['tx'] for j in range(i-6, i+1)])
        tx_prev = np.mean([data[j]['tx'] for j in range(i-13, i-6)]) if i >= 14 else tx_now
        d['tx_momentum'] = (tx_now / tx_prev - 1) * 100 if tx_prev > 0 else 0
    else:
        d['tx_momentum'] = 0

    # Whale activity momentum
    if i >= 7:
        w_now = np.mean([data[j]['whale'] for j in range(i-6, i+1)])
        w_prev = np.mean([data[j]['whale'] for j in range(i-13, i-6)]) if i >= 14 else w_now
        d['whale_momentum'] = (w_now / w_prev - 1) * 100 if w_prev > 0 else 0
    else:
        d['whale_momentum'] = 0

    # Cross-correlations (RenTech style - relationships between metrics)
    if i >= 30:
        tx_series = [data[j]['tx'] for j in range(i-29, i+1)]
        price_series = [data[j]['price'] for j in range(i-29, i+1)]
        whale_series = [data[j]['whale'] for j in range(i-29, i+1)]

        # Correlation coefficients
        d['tx_price_corr'] = np.corrcoef(tx_series, price_series)[0, 1] if np.std(tx_series) > 0 else 0
        d['whale_price_corr'] = np.corrcoef(whale_series, price_series)[0, 1] if np.std(whale_series) > 0 else 0
        d['tx_whale_corr'] = np.corrcoef(tx_series, whale_series)[0, 1] if np.std(tx_series) > 0 else 0
    else:
        d['tx_price_corr'] = 0
        d['whale_price_corr'] = 0
        d['tx_whale_corr'] = 0

    # Bollinger Band position
    if i >= 20:
        prices_20 = [data[j]['price'] for j in range(i-19, i+1)]
        bb_mid = np.mean(prices_20)
        bb_std = np.std(prices_20)
        d['bb_position'] = (d['price'] - bb_mid) / (2 * bb_std) if bb_std > 0 else 0  # -1 to 1 scale
    else:
        d['bb_position'] = 0

    # Rate of change acceleration (2nd derivative)
    if i >= 14:
        roc_7 = d['ret_7d']
        roc_14 = d['ret_14d']
        d['roc_acceleration'] = roc_7 - (roc_14 - roc_7)  # Change in momentum
    else:
        d['roc_acceleration'] = 0

print(f"  Created {len([k for k in data[100].keys() if k not in ['date', 'tx', 'whale', 'value', 'price']])} features")

# =============================================================================
# ANOMALY DETECTION - Statistical Outliers
# =============================================================================
print("\n[2] ANOMALY DETECTION...")

def mahalanobis_distance(point, mean, cov_inv):
    """Multivariate anomaly detection"""
    diff = point - mean
    return np.sqrt(diff.T @ cov_inv @ diff)

# Calculate composite anomaly score
anomaly_features = ['tx_z30', 'whale_z30', 'value_z30', 'ret_7d', 'volatility_20d']
feature_matrix = []
for d in data[100:]:  # Skip first 100 for stability
    feature_matrix.append([d.get(f, 0) for f in anomaly_features])
feature_matrix = np.array(feature_matrix)

# Compute mean and covariance
mean_vec = np.mean(feature_matrix, axis=0)
try:
    cov_matrix = np.cov(feature_matrix.T)
    cov_inv = np.linalg.inv(cov_matrix)

    for i, d in enumerate(data[100:], 100):
        point = np.array([d.get(f, 0) for f in anomaly_features])
        d['anomaly_score'] = mahalanobis_distance(point, mean_vec, cov_inv)
except:
    for d in data:
        d['anomaly_score'] = 0

print(f"  Calculated Mahalanobis distance for anomaly detection")

# =============================================================================
# REGIME DETECTION - Clustering Market States
# =============================================================================
print("\n[3] REGIME DETECTION (K-Means Clustering)...")

regime_features = ['ret_30d', 'volatility_20d', 'tx_z30', 'whale_z30']
regime_matrix = []
for d in data[100:]:
    regime_matrix.append([d.get(f, 0) for f in regime_features])
regime_matrix = np.array(regime_matrix)

# Normalize
regime_matrix = (regime_matrix - regime_matrix.mean(axis=0)) / (regime_matrix.std(axis=0) + 1e-8)

# Cluster into 5 regimes
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
regimes = kmeans.fit_predict(regime_matrix)

regime_names = {0: 'NEUTRAL', 1: 'BULL', 2: 'BEAR', 3: 'VOLATILE', 4: 'QUIET'}
# Map based on characteristics
cluster_means = {}
for r in range(5):
    mask = regimes == r
    cluster_means[r] = regime_matrix[mask].mean(axis=0)

# Assign names based on return and volatility
sorted_by_ret = sorted(cluster_means.items(), key=lambda x: x[1][0])
regime_map = {}
regime_map[sorted_by_ret[0][0]] = 'CAPITULATION'  # Lowest return
regime_map[sorted_by_ret[1][0]] = 'BEAR'
regime_map[sorted_by_ret[2][0]] = 'NEUTRAL'
regime_map[sorted_by_ret[3][0]] = 'BULL'
regime_map[sorted_by_ret[4][0]] = 'EUPHORIA'  # Highest return

for i, d in enumerate(data[:100]):
    d['regime'] = 'NEUTRAL'
    d['regime_id'] = 2

for i, d in enumerate(data[100:], 100):
    cluster = regimes[i - 100]
    d['regime'] = regime_map[cluster]
    d['regime_id'] = cluster

# Count regimes
regime_counts = {}
for d in data[100:]:
    r = d['regime']
    regime_counts[r] = regime_counts.get(r, 0) + 1
print(f"  Regime distribution: {regime_counts}")

# =============================================================================
# PATTERN DISCOVERY - RenTech Multi-Dimensional
# =============================================================================
print("\n[4] ADVANCED PATTERN DISCOVERY...")

# Define complex patterns (combining multiple signals)
patterns = [
    # === CAPITULATION PATTERNS (contrarian) ===
    {
        "name": "DEEP_CAPITULATION",
        "desc": "Extreme multi-dimensional oversold",
        "condition": lambda d: (
            d.get('price_vs_ma30', 0) < -25 and
            d.get('tx_z30', 0) < -1.5 and
            d.get('rsi_14', 50) < 25 and
            d.get('bb_position', 0) < -1
        ),
        "direction": 1
    },
    {
        "name": "VOLATILITY_CAPITULATION",
        "desc": "Low volatility + deep oversold = spring loading",
        "condition": lambda d: (
            d.get('price_vs_ma30', 0) < -20 and
            d.get('volatility_20d', 20) < 50 and
            d.get('anomaly_score', 0) > 3
        ),
        "direction": 1
    },
    {
        "name": "WHALE_ACCUMULATION",
        "desc": "Whales buying while price drops",
        "condition": lambda d: (
            d.get('ret_7d', 0) < -10 and
            d.get('whale_momentum', 0) > 20 and
            d.get('whale_z30', 0) > 1
        ),
        "direction": 1
    },
    {
        "name": "CORRELATION_BREAK_BULL",
        "desc": "TX-price correlation breaks down in oversold",
        "condition": lambda d: (
            d.get('tx_price_corr', 0) < -0.3 and
            d.get('price_vs_ma30', 0) < -15 and
            d.get('tx_z30', 0) > 0
        ),
        "direction": 1
    },

    # === MOMENTUM PATTERNS ===
    {
        "name": "ACCELERATION_BREAKOUT",
        "desc": "Positive acceleration from oversold",
        "condition": lambda d: (
            d.get('roc_acceleration', 0) > 5 and
            d.get('price_vs_ma30', 0) < -10 and
            d.get('ret_3d', 0) > 3
        ),
        "direction": 1
    },
    {
        "name": "MA_CASCADE_UP",
        "desc": "All short MAs crossing above long MAs",
        "condition": lambda d: (
            d.get('ma_5_20_cross', 0) > 2 and
            d.get('ma_10_30_cross', 0) > 0 and
            d.get('ret_7d', 0) > 5
        ),
        "direction": 1
    },
    {
        "name": "VOLUME_MOMENTUM_CONFLUENCE",
        "desc": "TX and whale momentum aligned with price",
        "condition": lambda d: (
            d.get('tx_momentum', 0) > 10 and
            d.get('whale_momentum', 0) > 10 and
            d.get('ret_7d', 0) > 5
        ),
        "direction": 1
    },

    # === REGIME-BASED PATTERNS ===
    {
        "name": "CAPITULATION_REGIME_ENTRY",
        "desc": "Enter long when regime = CAPITULATION",
        "condition": lambda d: d.get('regime') == 'CAPITULATION',
        "direction": 1
    },
    {
        "name": "EUPHORIA_EXIT_SHORT",
        "desc": "Short when euphoria + overbought",
        "condition": lambda d: (
            d.get('regime') == 'EUPHORIA' and
            d.get('rsi_14', 50) > 80 and
            d.get('price_vs_ma30', 0) > 30
        ),
        "direction": -1
    },

    # === ANOMALY PATTERNS ===
    {
        "name": "EXTREME_ANOMALY_LONG",
        "desc": "Statistical anomaly in oversold territory",
        "condition": lambda d: (
            d.get('anomaly_score', 0) > 4 and
            d.get('ret_7d', 0) < -15
        ),
        "direction": 1
    },
    {
        "name": "MULTI_Z_EXTREME",
        "desc": "All z-scores extremely negative",
        "condition": lambda d: (
            d.get('tx_z30', 0) < -2 and
            d.get('whale_z30', 0) < -2 and
            d.get('value_z30', 0) < -2
        ),
        "direction": 1
    },

    # === MEAN REVERSION PATTERNS ===
    {
        "name": "BOLLINGER_BOUNCE",
        "desc": "Price at lower Bollinger band with declining volatility",
        "condition": lambda d: (
            d.get('bb_position', 0) < -0.9 and
            d.get('volatility_20d', 50) < 80 and
            d.get('ret_3d', 0) > 0  # Starting to bounce
        ),
        "direction": 1
    },
    {
        "name": "RSI_DIVERGENCE",
        "desc": "Price making new lows but RSI not confirming",
        "condition": lambda d: (
            d.get('ret_14d', 0) < -15 and
            d.get('rsi_14', 50) > 35 and  # RSI not as low
            d.get('ret_3d', 0) > 0
        ),
        "direction": 1
    },
]

# =============================================================================
# BACKTEST ALL PATTERNS
# =============================================================================
print("\n[5] BACKTESTING PATTERNS...")

def backtest_pattern(pattern, hold_days=7, max_dd_threshold=2.0):
    """Backtest a pattern with drawdown analysis"""
    trades = []

    for i in range(100, len(data) - hold_days):
        try:
            if not pattern["condition"](data[i]):
                continue
        except:
            continue

        entry = data[i]['price']
        exit_price = data[i + hold_days]['price']
        direction = pattern["direction"]
        ret = (exit_price / entry - 1) * 100 * direction

        # Max drawdown
        max_dd = 0
        for j in range(i, i + hold_days + 1):
            if j < len(data):
                if direction == 1:  # Long
                    dd = (entry - data[j]['price']) / entry * 100
                else:  # Short
                    dd = (data[j]['price'] - entry) / entry * 100
                max_dd = max(max_dd, dd)

        trades.append({
            'date': data[i]['date'],
            'entry': entry,
            'exit': exit_price,
            'return': ret,
            'max_dd': max_dd,
            'safe_50x': max_dd < max_dd_threshold,
            'regime': data[i].get('regime', 'UNKNOWN')
        })

    return trades

results = []
for pattern in patterns:
    for hold in [3, 7, 14, 30]:
        trades = backtest_pattern(pattern, hold)

        if len(trades) < 3:
            continue

        wins = sum(1 for t in trades if t['return'] > 0)
        win_rate = wins / len(trades)
        avg_ret = np.mean([t['return'] for t in trades])

        safe = [t for t in trades if t['safe_50x']]
        safe_wins = sum(1 for t in safe if t['return'] > 0)
        safe_wr = safe_wins / len(safe) if safe else 0
        safe_avg = np.mean([t['return'] for t in safe]) if safe else 0

        results.append({
            'name': pattern['name'],
            'hold': hold,
            'n': len(trades),
            'win_rate': win_rate,
            'avg_ret': avg_ret,
            'n_safe': len(safe),
            'safe_wr': safe_wr,
            'safe_avg': safe_avg,
            'trades': trades
        })

# Sort by safe win rate, then by safe avg return
results.sort(key=lambda x: (-x['safe_wr'], -x['safe_avg'], -x['n_safe']))

print("\n" + "="*100)
print("  TOP PATTERNS BY 50x-SAFE WIN RATE")
print("="*100)
print(f"{'Pattern':<30} {'Hold':>5} {'N':>4} {'WR':>6} {'Avg':>8} {'Safe':>5} {'SafeWR':>7} {'SafeAvg':>8}")
print("-"*100)

for r in results[:25]:
    marker = " ***" if r['safe_wr'] >= 0.95 and r['n_safe'] >= 3 else ""
    print(f"{r['name']:<30} {r['hold']:>4}d {r['n']:>4} {r['win_rate']*100:>5.0f}% {r['avg_ret']:>+7.1f}% {r['n_safe']:>5} {r['safe_wr']*100:>6.0f}% {r['safe_avg']:>+7.1f}%{marker}")

# =============================================================================
# SAVE TOP FORMULAS
# =============================================================================
print("\n" + "="*80)
print("  TOP RENTECH-STYLE FORMULAS (100% Safe Win Rate)")
print("="*80)

perfect = [r for r in results if r['safe_wr'] == 1.0 and r['n_safe'] >= 2]
for r in perfect:
    print(f"\n{r['name']} @ {r['hold']}d hold")
    print(f"  Total trades: {r['n']} | 50x-safe: {r['n_safe']}")
    print(f"  Safe WR: 100% | Safe Avg: {r['safe_avg']:+.2f}%")
    print(f"  At 50x leverage: {r['safe_avg']*50:+.0f}% per trade")

    safe_trades = [t for t in r['trades'] if t['safe_50x']]
    print(f"  Trades:")
    for t in safe_trades[:5]:
        print(f"    {t['date']}: ${t['entry']:,.0f} -> ${t['exit']:,.0f} | DD: {t['max_dd']:.1f}% | Ret: {t['return']:+.1f}%")
