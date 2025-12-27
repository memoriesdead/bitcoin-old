#!/usr/bin/env python3
"""
FIND CONDITIONS SAFE FOR 50x LEVERAGE
=====================================
Must have:
1. 100% win rate
2. Max intra-period drawdown < 2%
"""

import sqlite3
import numpy as np
import struct
from datetime import datetime

print("="*80)
print("  FINDING 50x-SAFE CONDITIONS")
print("  Requirement: 100% win rate AND max drawdown < 2%")
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

# Compute features
for i, d in enumerate(data):
    start = max(0, i - 30)

    # Z-scores
    for metric in ['tx', 'whale', 'value']:
        window = [data[j][metric] for j in range(start, i+1)]
        mean, std = np.mean(window), np.std(window)
        d[f'{metric}_z'] = (d[metric] - mean) / std if len(window) > 1 and std > 0 else 0

    # Returns
    for lb in [1, 3, 7, 14, 30]:
        d[f'ret_{lb}d'] = (d['price'] / data[i-lb]['price'] - 1) * 100 if i >= lb else 0

    # Streaks
    down_streak = 0
    for j in range(i-1, max(0, i-15), -1):
        if data[j+1]['price'] < data[j]['price']:
            down_streak += 1
        else:
            break
    d['down_streak'] = down_streak

    # MA deviation
    if i >= 30:
        ma = np.mean([data[j]['price'] for j in range(i-30, i)])
        d['price_vs_ma30'] = (d['price'] / ma - 1) * 100
    else:
        d['price_vs_ma30'] = 0

date_idx = {d['date']: i for i, d in enumerate(data)}

def test_condition(name, cond_fn, hold_days, max_dd_threshold=2.0):
    """Test a condition with drawdown check."""
    trades = []

    for i in range(len(data) - hold_days):
        try:
            if not cond_fn(data[i]):
                continue
        except:
            continue

        entry_price = data[i]['price']
        exit_price = data[i + hold_days]['price']
        ret = (exit_price / entry_price - 1) * 100

        # Calculate max drawdown during hold
        min_price = entry_price
        max_dd = 0
        for j in range(i, i + hold_days + 1):
            if j < len(data):
                if data[j]['price'] < min_price:
                    min_price = data[j]['price']
                    dd = (entry_price - min_price) / entry_price * 100
                    max_dd = max(max_dd, dd)

        trades.append({
            'date': data[i]['date'],
            'ret': ret,
            'max_dd': max_dd,
            'entry': entry_price,
            'exit': exit_price,
            'safe_50x': max_dd < max_dd_threshold
        })

    return trades

# Test many conditions at multiple hold periods
conditions = [
    # Extreme capitulation
    ('tx_z < -2 AND whale_z < -2', lambda d: d['tx_z'] < -2 and d['whale_z'] < -2),
    ('tx_z < -2.5 AND whale_z < -2.5', lambda d: d['tx_z'] < -2.5 and d['whale_z'] < -2.5),
    ('tx_z < -3 AND whale_z < -2', lambda d: d['tx_z'] < -3 and d['whale_z'] < -2),
    ('ALL z < -2', lambda d: d['tx_z'] < -2 and d['whale_z'] < -2 and d['value_z'] < -2),
    ('ALL z < -2.5', lambda d: d['tx_z'] < -2.5 and d['whale_z'] < -2.5 and d['value_z'] < -2.5),

    # With price conditions
    ('tx_z < -2 AND whale_z < -2 AND ret_7d < -5', lambda d: d['tx_z'] < -2 and d['whale_z'] < -2 and d['ret_7d'] < -5),
    ('tx_z < -2 AND whale_z < -2 AND ret_7d < -10', lambda d: d['tx_z'] < -2 and d['whale_z'] < -2 and d['ret_7d'] < -10),
    ('ALL z < -2 AND ret_7d < -5', lambda d: d['tx_z'] < -2 and d['whale_z'] < -2 and d['value_z'] < -2 and d['ret_7d'] < -5),

    # Streak-based
    ('down_streak >= 5 AND tx_z < -1', lambda d: d['down_streak'] >= 5 and d['tx_z'] < -1),
    ('down_streak >= 6 AND tx_z < -1', lambda d: d['down_streak'] >= 6 and d['tx_z'] < -1),
    ('down_streak >= 7', lambda d: d['down_streak'] >= 7),

    # Extreme oversold
    ('ret_7d < -20 AND tx_z < -1', lambda d: d['ret_7d'] < -20 and d['tx_z'] < -1),
    ('ret_14d < -25 AND tx_z < -1', lambda d: d['ret_14d'] < -25 and d['tx_z'] < -1),
    ('price_vs_ma30 < -25', lambda d: d['price_vs_ma30'] < -25),
    ('price_vs_ma30 < -30', lambda d: d['price_vs_ma30'] < -30),

    # Combined extreme
    ('tx_z < -2 AND ret_7d < -15', lambda d: d['tx_z'] < -2 and d['ret_7d'] < -15),
    ('whale_z < -2 AND ret_7d < -15', lambda d: d['whale_z'] < -2 and d['ret_7d'] < -15),
    ('tx_z < -2 AND whale_z < -1.5 AND ret_7d < -10', lambda d: d['tx_z'] < -2 and d['whale_z'] < -1.5 and d['ret_7d'] < -10),
]

print("\n" + "="*80)
print("  TESTING ALL CONDITIONS (looking for 100% win + low drawdown)")
print("="*80)

safe_50x_results = []

for hold in [1, 3, 7, 14, 30]:
    for cond_name, cond_fn in conditions:
        trades = test_condition(cond_name, cond_fn, hold)

        if len(trades) < 3:
            continue

        wins = sum(1 for t in trades if t['ret'] > 0)
        win_rate = wins / len(trades)

        safe_trades = [t for t in trades if t['safe_50x']]
        safe_wins = sum(1 for t in safe_trades if t['ret'] > 0)

        if win_rate >= 0.85:
            safe_50x_results.append({
                'cond': cond_name,
                'hold': hold,
                'n': len(trades),
                'wr': win_rate,
                'avg_ret': np.mean([t['ret'] for t in trades]),
                'n_safe': len(safe_trades),
                'safe_wr': safe_wins / len(safe_trades) if safe_trades else 0,
                'safe_avg': np.mean([t['ret'] for t in safe_trades]) if safe_trades else 0,
                'trades': trades
            })

safe_50x_results.sort(key=lambda x: (-x['safe_wr'], -x['n_safe'], -x['safe_avg']))

print(f"\n{'Condition':<55} {'Hold':>4} {'N':>3} {'WR':>5} {'SafeN':>5} {'SafeWR':>6} {'SafeAvg':>8}")
print("-"*100)

for r in safe_50x_results[:30]:
    marker = " ***" if r['safe_wr'] == 1.0 and r['n_safe'] >= 2 else ""
    print(f"{r['cond']:<55} {r['hold']:>3}d {r['n']:>3} {r['wr']*100:>4.0f}% {r['n_safe']:>5} {r['safe_wr']*100:>5.0f}% {r['safe_avg']:>+7.1f}%{marker}")

# Focus on 100% win rate with 50x-safe trades
print("\n" + "="*80)
print("  100% WIN RATE CONDITIONS THAT ARE SAFE AT 50x")
print("="*80)

perfect_safe = [r for r in safe_50x_results if r['safe_wr'] == 1.0 and r['n_safe'] >= 2]

for r in perfect_safe:
    print(f"\n{r['cond']} @ {r['hold']}d hold")
    print(f"  Total trades: {r['n']} | 50x-safe trades: {r['n_safe']}")
    print(f"  Safe trade win rate: 100%")
    print(f"  Safe trade avg return: {r['safe_avg']:+.2f}%")
    print(f"  At 50x: {r['safe_avg']*50:+.0f}% per trade")

    safe_trades = [t for t in r['trades'] if t['safe_50x']]
    print(f"  Trades:")
    for t in safe_trades:
        print(f"    {t['date']}: Entry ${t['entry']:,.0f} -> ${t['exit']:,.0f} | DD: {t['max_dd']:.2f}% | Ret: {t['ret']:+.1f}%")

# Compounding the SAFE trades only
print("\n" + "="*80)
print("  COMPOUNDING 50x-SAFE TRADES ONLY")
print("="*80)

if perfect_safe:
    best = max(perfect_safe, key=lambda x: x['n_safe'])
    safe_trades = [t for t in best['trades'] if t['safe_50x']]

    print(f"\nBest condition: {best['cond']} @ {best['hold']}d")
    print(f"Safe trades: {len(safe_trades)}")

    for lev in [10, 25, 50]:
        capital = 100
        print(f"\n{lev}x leverage:")
        for t in safe_trades:
            ret = t['ret'] / 100 * lev
            prev_cap = capital
            capital *= (1 + ret)
            print(f"  {t['date']}: {t['ret']:+.1f}% -> {t['ret']*lev:+.0f}% | ${prev_cap:,.0f} -> ${capital:,.0f}")
        print(f"  FINAL: ${100:,.0f} -> ${capital:,.0f} ({capital/100:,.0f}x)")
