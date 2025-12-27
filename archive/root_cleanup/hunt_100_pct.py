#!/usr/bin/env python3
"""
HUNT FOR 100% WIN RATE CONDITIONS
=================================
Extreme leverage requires mathematical certainty.
"""

import sqlite3
import numpy as np
import struct
from datetime import datetime

# Load ALL data
conn = sqlite3.connect('data/unified_bitcoin.db')
cursor = conn.execute('SELECT date, close FROM prices ORDER BY date')
prices = {row[0]: row[1] for row in cursor.fetchall()}

cursor = conn.execute('SELECT date, tx_count, whale_tx_count, total_value_btc FROM daily_features ORDER BY date')
data = []
for row in cursor.fetchall():
    whale = row[2]
    if isinstance(whale, bytes):
        try: whale = struct.unpack('<q', whale)[0]
        except: whale = 0
    elif whale is None: whale = 0
    if row[0] in prices and prices[row[0]] > 0:
        data.append({'date': row[0], 'tx': row[1] or 0, 'whale': whale, 'value': row[3] or 0, 'price': prices[row[0]]})
conn.close()

# Compute ALL features
for i, d in enumerate(data):
    start = max(0, i - 30)
    for metric in ['tx', 'whale', 'value']:
        window = [data[j][metric] for j in range(start, i+1)]
        d[f'{metric}_z'] = (d[metric] - np.mean(window)) / np.std(window) if len(window) > 1 and np.std(window) > 0 else 0

    # Day of week
    d['dow'] = datetime.strptime(d['date'], '%Y-%m-%d').weekday()
    d['dom'] = int(d['date'].split('-')[2])

    # Streaks
    down_streak = 0
    for j in range(i-1, max(0, i-15), -1):
        if data[j+1]['price'] < data[j]['price']:
            down_streak += 1
        else:
            break
    d['down_streak'] = down_streak

    # Moving averages
    if i >= 30:
        d['ma_30'] = np.mean([data[j]['price'] for j in range(i-30, i)])
        d['price_vs_ma30'] = (d['price'] / d['ma_30'] - 1) * 100
    else:
        d['price_vs_ma30'] = 0

    # Returns
    for lb in [1, 3, 7, 14, 30]:
        if i >= lb:
            d[f'ret_{lb}d'] = (d['price'] / data[i-lb]['price'] - 1) * 100
        else:
            d[f'ret_{lb}d'] = 0

print('='*70)
print('  EXHAUSTIVE 100% WIN RATE SEARCH')
print('='*70)

results = []

conditions = [
    # Streak patterns
    ('down_streak >= 4', lambda d: d['down_streak'] >= 4),
    ('down_streak >= 5', lambda d: d['down_streak'] >= 5),
    ('down_streak >= 6', lambda d: d['down_streak'] >= 6),
    ('down_streak >= 7', lambda d: d['down_streak'] >= 7),
    ('down_streak >= 4 AND tx_z < -1', lambda d: d['down_streak'] >= 4 and d['tx_z'] < -1),
    ('down_streak >= 3 AND tx_z < -1.5', lambda d: d['down_streak'] >= 3 and d['tx_z'] < -1.5),
    ('down_streak >= 3 AND ret_7d < -10', lambda d: d['down_streak'] >= 3 and d['ret_7d'] < -10),

    # Extreme oversold
    ('ret_7d < -15', lambda d: d['ret_7d'] < -15),
    ('ret_7d < -20', lambda d: d['ret_7d'] < -20),
    ('ret_14d < -20', lambda d: d['ret_14d'] < -20),
    ('ret_14d < -25', lambda d: d['ret_14d'] < -25),
    ('ret_14d < -30', lambda d: d['ret_14d'] < -30),
    ('price_vs_ma30 < -20', lambda d: d['price_vs_ma30'] < -20),
    ('price_vs_ma30 < -25', lambda d: d['price_vs_ma30'] < -25),
    ('price_vs_ma30 < -30', lambda d: d['price_vs_ma30'] < -30),

    # Combined extreme
    ('ret_7d < -15 AND tx_z < -1', lambda d: d['ret_7d'] < -15 and d['tx_z'] < -1),
    ('ret_14d < -20 AND tx_z < -1.5', lambda d: d['ret_14d'] < -20 and d['tx_z'] < -1.5),
    ('price_vs_ma30 < -20 AND tx_z < -1.5', lambda d: d['price_vs_ma30'] < -20 and d['tx_z'] < -1.5),
    ('price_vs_ma30 < -25 AND tx_z < -1', lambda d: d['price_vs_ma30'] < -25 and d['tx_z'] < -1),

    # Capitulation
    ('tx_z < -2 AND whale_z < -2', lambda d: d['tx_z'] < -2 and d['whale_z'] < -2),
    ('tx_z < -2 AND whale_z < -1.5 AND ret_7d < -10', lambda d: d['tx_z'] < -2 and d['whale_z'] < -1.5 and d['ret_7d'] < -10),
    ('ALL z < -1.5', lambda d: d['tx_z'] < -1.5 and d['whale_z'] < -1.5 and d['value_z'] < -1.5),
    ('ALL z < -2', lambda d: d['tx_z'] < -2 and d['whale_z'] < -2 and d['value_z'] < -2),

    # Combined oversold + low activity
    ('ret_7d < -10 AND tx_z < -1.5', lambda d: d['ret_7d'] < -10 and d['tx_z'] < -1.5),
    ('ret_7d < -10 AND whale_z < -1.5', lambda d: d['ret_7d'] < -10 and d['whale_z'] < -1.5),
    ('ret_14d < -15 AND tx_z < -1.5', lambda d: d['ret_14d'] < -15 and d['tx_z'] < -1.5),
    ('ret_14d < -15 AND whale_z < -2', lambda d: d['ret_14d'] < -15 and d['whale_z'] < -2),

    # Extreme spikes (momentum continuation)
    ('tx_z > 2.5 AND ret_7d > 15', lambda d: d['tx_z'] > 2.5 and d['ret_7d'] > 15),
    ('tx_z > 3 AND ret_7d > 10', lambda d: d['tx_z'] > 3 and d['ret_7d'] > 10),
]

for cond_name, cond_fn in conditions:
    for hold in [1, 3, 7, 14, 30]:
        trades = []
        for i in range(len(data) - hold):
            try:
                if cond_fn(data[i]):
                    ret = (data[i+hold]['price'] / data[i]['price'] - 1) * 100
                    trades.append({'date': data[i]['date'], 'ret': ret})
            except:
                pass

        if len(trades) >= 3:
            wins = sum(1 for t in trades if t['ret'] > 0)
            win_rate = wins / len(trades)
            avg_ret = np.mean([t['ret'] for t in trades])

            if win_rate >= 0.85:
                results.append({
                    'cond': cond_name,
                    'hold': hold,
                    'n': len(trades),
                    'wr': win_rate,
                    'avg': avg_ret,
                    'trades': trades
                })

results.sort(key=lambda x: (-x['wr'], -x['n'], -x['avg']))

print()
print(f"{'Condition':<50} {'Hold':>4} {'N':>4} {'Win%':>6} {'Avg':>8}")
print('-'*80)

for r in results[:40]:
    marker = ' ***' if r['wr'] == 1.0 else (' **' if r['wr'] >= 0.95 else '')
    print(f"{r['cond']:<50} {r['hold']:>3}d {r['n']:>4} {r['wr']*100:>5.0f}% {r['avg']:>+7.1f}%{marker}")

# 100% Summary
print()
print('='*70)
print('  100% WIN RATE CONDITIONS')
print('='*70)

perfect = [r for r in results if r['wr'] == 1.0]
by_cond = {}
for p in perfect:
    if p['cond'] not in by_cond:
        by_cond[p['cond']] = []
    by_cond[p['cond']].append(p)

for cond, items in sorted(by_cond.items(), key=lambda x: max(i['n'] for i in x[1]), reverse=True):
    best = max(items, key=lambda x: x['avg'])
    total_trades = max(i['n'] for i in items)
    print(f"\n{cond}")
    print(f"  Trades: {total_trades} | Best: {best['hold']}d hold | Avg: +{best['avg']:.1f}%")
    print(f"  At 50x leverage: +{best['avg']*50:.0f}% per trade")
    print(f"  Dates: {', '.join(t['date'] for t in best['trades'])}")

# COMPOUNDING
print()
print('='*70)
print('  COMPOUNDING WITH 100% WIN RATE + 50x LEVERAGE')
print('='*70)

if perfect:
    # Use the condition with most trades
    best_cond = max(perfect, key=lambda x: x['n'])

    print(f"\nCondition: {best_cond['cond']}")
    print(f"Hold: {best_cond['hold']}d")
    print(f"Trades: {best_cond['n']}")
    print(f"Avg return per trade: +{best_cond['avg']:.1f}%")

    for lev in [10, 25, 50]:
        capital = 100
        for t in best_cond['trades']:
            ret = t['ret'] / 100 * lev
            capital *= (1 + ret)

        print(f"\n{lev}x leverage:")
        print(f"  $100 -> ${capital:,.0f} ({capital/100:,.0f}x return)")

        # Year by year
        print("  By trade:")
        cap = 100
        for t in best_cond['trades']:
            ret = t['ret'] / 100 * lev
            cap *= (1 + ret)
            print(f"    {t['date']}: +{t['ret']*lev:.0f}% -> ${cap:,.0f}")
