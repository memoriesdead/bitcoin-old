#!/usr/bin/env python3
"""
ROI OPTIMIZER - Find fastest path from $100 to $1000

Tests all parameter combinations to find highest daily ROI.
"""

import os
import sys
import json
import sqlite3
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import List, Dict, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

BASE_DIR = Path(__file__).parent.parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = DATA_DIR / "simulation_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# PARAMETER GRID TO TEST
# =============================================================================

PARAM_GRID = {
    'kelly_fraction': [0.5, 0.75, 1.0, 1.5],
    'leverage': [5, 10, 15, 20, 25],
    'tp_pct': [0.005, 0.008, 0.01, 0.015, 0.02],  # 0.5% to 2%
    'sl_pct': [0.003, 0.004, 0.005, 0.008],        # 0.3% to 0.8%
    'signal_threshold': [1.1, 1.2, 1.3, 1.5],      # How aggressive
}

# Fees
TAKER_FEE = 0.00035
SLIPPAGE_BPS = 3


def load_data() -> List[Dict]:
    """Load price and feature data."""
    features = {}
    prices = {}

    # Load features
    for path in [DATA_DIR / "bitcoin_features.db", DATA_DIR / "historical_flows.db"]:
        if path.exists():
            conn = sqlite3.connect(path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [t[0] for t in cursor.fetchall()]

            if 'daily_features' in tables:
                cursor.execute("""
                    SELECT timestamp, tx_count, total_value_btc, whale_tx_count,
                           unique_senders, unique_receivers
                    FROM daily_features WHERE tx_count IS NOT NULL
                """)
                for row in cursor.fetchall():
                    ts = row[0]
                    date_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y-%m-%d')
                    features[date_str] = {
                        'timestamp': ts,
                        'tx_count': row[1] or 0,
                        'total_value_btc': row[2] or 0,
                        'whale_tx_count': row[3] or 0,
                        'unique_senders': row[4] or 0,
                        'unique_receivers': row[5] or 0,
                    }
            conn.close()
            if features:
                break

    # Load prices
    for path in [DATA_DIR / "bitcoin_2021_2025.db", DATA_DIR / "historical_flows.db"]:
        if path.exists():
            conn = sqlite3.connect(path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [t[0] for t in cursor.fetchall()]

            for table in ['prices', 'ohlcv']:
                if table in tables:
                    cursor.execute(f"SELECT timestamp, open, high, low, close FROM {table}")
                    for row in cursor.fetchall():
                        ts = row[0]
                        date_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y-%m-%d')
                        prices[date_str] = {
                            'timestamp': ts,
                            'open': row[1],
                            'high': row[2],
                            'low': row[3],
                            'close': row[4],
                        }
                    break
            conn.close()
            if prices:
                break

    # Merge
    data = []
    for date_str in sorted(features.keys()):
        if date_str in prices:
            data.append({'date': date_str, **features[date_str], **prices[date_str]})

    return data


def generate_signals(data: List[Dict], idx: int, threshold: float) -> List[Dict]:
    """Generate signals with given threshold."""
    if idx < 5:
        return []

    signals = []
    window = data[idx - 5:idx]
    current = data[idx]

    avg_whale = sum(f['whale_tx_count'] for f in window) / len(window)
    avg_value = sum(f['total_value_btc'] for f in window) / len(window)
    avg_tx = sum(f['tx_count'] for f in window) / len(window)

    # Whale signal
    if avg_whale > 0:
        ratio = current['whale_tx_count'] / avg_whale
        if ratio > threshold:
            receivers = current.get('unique_receivers', 1) or 1
            senders = current.get('unique_senders', 1) or 1
            direction = 1 if receivers > senders else -1
            confidence = min(0.70, 0.55 + (ratio - 1) * 0.1)
            signals.append({'direction': direction, 'confidence': confidence, 'type': 'WHALE'})

    # Value signal
    if avg_value > 0:
        ratio = current['total_value_btc'] / avg_value
        if ratio > threshold:
            receivers = current.get('unique_receivers', 1) or 1
            senders = current.get('unique_senders', 1) or 1
            direction = 1 if receivers > senders else -1
            confidence = min(0.65, 0.52 + (ratio - 1) * 0.08)
            signals.append({'direction': direction, 'confidence': confidence, 'type': 'VALUE'})

    # TX signal
    if avg_tx > 0:
        ratio = current['tx_count'] / avg_tx
        if ratio > threshold * 0.9:  # Slightly lower for TX
            prev_value = window[-1]['total_value_btc']
            direction = 1 if current['total_value_btc'] > prev_value else -1
            confidence = min(0.58, 0.51 + (ratio - 1) * 0.05)
            signals.append({'direction': direction, 'confidence': confidence, 'type': 'TX'})

    # Momentum signal
    closes = [d['close'] for d in window]
    ma = sum(closes) / len(closes)
    if current['close'] > ma * 1.015:
        signals.append({'direction': 1, 'confidence': 0.54, 'type': 'MOM_UP'})
    elif current['close'] < ma * 0.985:
        signals.append({'direction': -1, 'confidence': 0.52, 'type': 'MOM_DOWN'})

    return signals


def run_single_test(params: Dict, data: List[Dict], test_days: int = 30) -> Dict:
    """Run simulation with specific parameters for N days."""
    kelly = params['kelly_fraction']
    leverage = params['leverage']
    tp_pct = params['tp_pct']
    sl_pct = params['sl_pct']
    threshold = params['signal_threshold']

    # Use last N days of data for testing
    if len(data) < test_days + 10:
        return {'daily_roi': 0, 'params': params}

    test_data = data[-(test_days + 10):]

    equity = 100.0
    start_equity = equity
    trades = []
    wins = 0
    losses = 0
    current_trade = None
    max_dd = 0
    peak = equity

    for i in range(5, len(test_data) - 1):
        current = test_data[i]
        next_candle = test_data[i + 1]

        # Check exit
        if current_trade:
            high = current['high']
            low = current['low']
            exit_price = None

            if current_trade['direction'] == 1:
                if high >= current_trade['tp']:
                    exit_price = current_trade['tp']
                    wins += 1
                elif low <= current_trade['sl']:
                    exit_price = current_trade['sl']
                    losses += 1
            else:
                if low <= current_trade['tp']:
                    exit_price = current_trade['tp']
                    wins += 1
                elif high >= current_trade['sl']:
                    exit_price = current_trade['sl']
                    losses += 1

            if exit_price:
                if current_trade['direction'] == 1:
                    pnl_pct = (exit_price - current_trade['entry']) / current_trade['entry']
                else:
                    pnl_pct = (current_trade['entry'] - exit_price) / current_trade['entry']

                pnl = current_trade['size'] * pnl_pct * leverage
                fees = current_trade['size'] * TAKER_FEE * 2
                equity += pnl - fees
                trades.append(pnl - fees)
                current_trade = None

                if equity > peak:
                    peak = equity
                dd = (peak - equity) / peak
                if dd > max_dd:
                    max_dd = dd

        # New signal
        if not current_trade and equity > 1:
            signals = generate_signals(test_data, i, threshold)
            if signals:
                best = max(signals, key=lambda s: s['confidence'])
                if best['confidence'] >= 0.50:
                    entry = next_candle['open']
                    if best['direction'] == 1:
                        entry *= (1 + SLIPPAGE_BPS / 10000)
                    else:
                        entry *= (1 - SLIPPAGE_BPS / 10000)

                    # Kelly size
                    w = best['confidence']
                    r = tp_pct / sl_pct  # Win/loss ratio
                    kelly_pct = max(0, min(w - (1 - w) / r, 0.5)) * kelly
                    size = equity * kelly_pct

                    if best['direction'] == 1:
                        sl = entry * (1 - sl_pct)
                        tp = entry * (1 + tp_pct)
                    else:
                        sl = entry * (1 + sl_pct)
                        tp = entry * (1 - tp_pct)

                    current_trade = {
                        'entry': entry,
                        'direction': best['direction'],
                        'size': size,
                        'sl': sl,
                        'tp': tp,
                    }

        if equity <= 0:
            break

    # Calculate metrics
    total_trades = wins + losses
    win_rate = wins / total_trades if total_trades > 0 else 0
    total_return = (equity / start_equity) - 1
    daily_roi = (equity / start_equity) ** (1 / test_days) - 1 if test_days > 0 else 0

    # Days to 10x
    if daily_roi > 0:
        import math
        days_to_10x = math.log(10) / math.log(1 + daily_roi)
    else:
        days_to_10x = float('inf')

    return {
        'params': params,
        'final_equity': equity,
        'total_return': total_return * 100,
        'daily_roi': daily_roi * 100,
        'days_to_10x': days_to_10x,
        'total_trades': total_trades,
        'win_rate': win_rate * 100,
        'max_drawdown': max_dd * 100,
        'wins': wins,
        'losses': losses,
    }


def run_optimization():
    """Run full parameter optimization."""
    print("=" * 70)
    print("ROI OPTIMIZER - $100 to $1000 FASTEST PATH")
    print("=" * 70)

    # Load data
    print("\n[1] Loading data...")
    data = load_data()
    print(f"    Loaded {len(data)} data points")

    # Generate parameter combinations
    print("\n[2] Generating parameter combinations...")
    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    combinations = list(product(*values))
    print(f"    Testing {len(combinations)} combinations")

    # Run tests
    print("\n[3] Running simulations...")
    results = []

    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        result = run_single_test(params, data, test_days=60)  # Test on 60 days
        results.append(result)

        if (i + 1) % 100 == 0:
            print(f"    Completed {i + 1}/{len(combinations)}")

    # Sort by daily ROI
    results.sort(key=lambda x: x['daily_roi'], reverse=True)

    # Print top 10
    print("\n" + "=" * 70)
    print("TOP 10 CONFIGURATIONS BY DAILY ROI")
    print("=" * 70)

    print(f"\n{'Rank':<5} {'Daily ROI':<12} {'Days-10x':<10} {'WinRate':<10} {'MaxDD':<10} {'Trades':<8} {'Config'}")
    print("-" * 100)

    for i, r in enumerate(results[:10]):
        p = r['params']
        config = f"K={p['kelly_fraction']} L={p['leverage']}x TP={p['tp_pct']*100:.1f}% SL={p['sl_pct']*100:.1f}%"
        print(f"{i+1:<5} {r['daily_roi']:>8.2f}%    {r['days_to_10x']:>8.1f}d   {r['win_rate']:>7.1f}%   {r['max_drawdown']:>7.1f}%   {r['total_trades']:<8} {config}")

    # Best result
    best = results[0]
    print("\n" + "=" * 70)
    print("OPTIMAL CONFIGURATION")
    print("=" * 70)
    print(f"\nParameters:")
    for k, v in best['params'].items():
        print(f"  {k}: {v}")

    print(f"\nPerformance:")
    print(f"  Daily ROI:     {best['daily_roi']:.2f}%")
    print(f"  Days to 10x:   {best['days_to_10x']:.1f} days")
    print(f"  Win Rate:      {best['win_rate']:.1f}%")
    print(f"  Max Drawdown:  {best['max_drawdown']:.1f}%")
    print(f"  Total Trades:  {best['total_trades']}")
    print(f"  Final Equity:  ${best['final_equity']:.2f} (from $100)")

    print("\n" + "=" * 70)
    print("PROJECTION: $100 to $1000")
    print("=" * 70)
    daily = best['daily_roi'] / 100

    projections = []
    equity = 100
    day = 0
    while equity < 1000 and day < 365:
        equity *= (1 + daily)
        day += 1
        if day in [7, 14, 30, 60, 90] or equity >= 1000:
            projections.append((day, equity))

    print(f"\n{'Day':<8} {'Equity':<12} {'Return'}")
    print("-" * 35)
    for d, e in projections:
        print(f"{d:<8} ${e:>10.2f}   {(e/100-1)*100:>+.1f}%")

    if best['days_to_10x'] < 365:
        print(f"\n>>> $100 to $1000 in {best['days_to_10x']:.0f} DAYS <<<")
    else:
        print(f"\n>>> Would take {best['days_to_10x']:.0f} days (too slow)")

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = RESULTS_DIR / f"roi_optimization_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'best_config': best,
            'top_10': results[:10],
            'total_tested': len(combinations),
        }, f, indent=2, default=str)
    print(f"\n[SAVED] {results_file}")

    return best


if __name__ == '__main__':
    run_optimization()
