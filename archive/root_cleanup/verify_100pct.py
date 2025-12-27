#!/usr/bin/env python3
"""
RIGOROUS VERIFICATION OF 100% WIN RATE CONDITIONS
==================================================
Triple-check every trade before risking real money.
"""

import sqlite3
import numpy as np
import struct
from datetime import datetime, timedelta

print("="*80)
print("  RIGOROUS VERIFICATION - EVERY TRADE CHECKED MANUALLY")
print("="*80)

# Load ALL data fresh
conn = sqlite3.connect('data/unified_bitcoin.db')

# Get prices
cursor = conn.execute('SELECT date, close FROM prices ORDER BY date')
prices = {row[0]: row[1] for row in cursor.fetchall()}
print(f"\nLoaded {len(prices)} price records")
print(f"Price date range: {min(prices.keys())} to {max(prices.keys())}")

# Get features
cursor = conn.execute('SELECT date, tx_count, whale_tx_count, total_value_btc FROM daily_features ORDER BY date')
raw_features = []
for row in cursor.fetchall():
    whale = row[2]
    if isinstance(whale, bytes):
        try: whale = struct.unpack('<q', whale)[0]
        except: whale = 0
    elif whale is None: whale = 0
    raw_features.append({
        'date': row[0],
        'tx': row[1] or 0,
        'whale': whale,
        'value': row[3] or 0
    })
conn.close()

print(f"Loaded {len(raw_features)} feature records")
print(f"Feature date range: {raw_features[0]['date']} to {raw_features[-1]['date']}")

# Build complete dataset with prices
data = []
for f in raw_features:
    if f['date'] in prices and prices[f['date']] > 0:
        data.append({
            'date': f['date'],
            'tx': f['tx'],
            'whale': f['whale'],
            'value': f['value'],
            'price': prices[f['date']]
        })

print(f"Combined records with valid prices: {len(data)}")

# Compute z-scores with 30-day rolling window
for i, d in enumerate(data):
    start = max(0, i - 30)

    # Transaction z-score
    tx_window = [data[j]['tx'] for j in range(start, i+1)]
    tx_mean = np.mean(tx_window)
    tx_std = np.std(tx_window)
    d['tx_z'] = (d['tx'] - tx_mean) / tx_std if len(tx_window) > 1 and tx_std > 0 else 0

    # Whale z-score
    whale_window = [data[j]['whale'] for j in range(start, i+1)]
    whale_mean = np.mean(whale_window)
    whale_std = np.std(whale_window)
    d['whale_z'] = (d['whale'] - whale_mean) / whale_std if len(whale_window) > 1 and whale_std > 0 else 0

    # Value z-score
    value_window = [data[j]['value'] for j in range(start, i+1)]
    value_mean = np.mean(value_window)
    value_std = np.std(value_window)
    d['value_z'] = (d['value'] - value_mean) / value_std if len(value_window) > 1 and value_std > 0 else 0

    # 7-day return (looking backwards)
    if i >= 7:
        d['ret_7d'] = (d['price'] / data[i-7]['price'] - 1) * 100
    else:
        d['ret_7d'] = 0

# Create date index for fast lookup
date_idx = {d['date']: i for i, d in enumerate(data)}

print("\n" + "="*80)
print("  CONDITION: tx_z < -2 AND whale_z < -2 (30-day hold)")
print("="*80)

condition_1_trades = []
for i in range(len(data) - 30):
    d = data[i]
    if d['tx_z'] < -2 and d['whale_z'] < -2:
        entry_price = d['price']
        exit_price = data[i + 30]['price']
        ret = (exit_price / entry_price - 1) * 100
        condition_1_trades.append({
            'date': d['date'],
            'exit_date': data[i + 30]['date'],
            'tx_z': d['tx_z'],
            'whale_z': d['whale_z'],
            'value_z': d['value_z'],
            'entry': entry_price,
            'exit': exit_price,
            'return': ret
        })

print(f"\nTotal trades found: {len(condition_1_trades)}")
print(f"Winning trades: {sum(1 for t in condition_1_trades if t['return'] > 0)}")
print(f"Losing trades: {sum(1 for t in condition_1_trades if t['return'] <= 0)}")

print("\nDETAILED TRADE BREAKDOWN:")
print("-"*80)
for t in condition_1_trades:
    win = "WIN" if t['return'] > 0 else "LOSS"
    print(f"\n{t['date']} -> {t['exit_date']}")
    print(f"  tx_z: {t['tx_z']:.3f}, whale_z: {t['whale_z']:.3f}, value_z: {t['value_z']:.3f}")
    print(f"  Entry: ${t['entry']:,.2f} -> Exit: ${t['exit']:,.2f}")
    print(f"  Return: {t['return']:+.2f}% [{win}]")

# Calculate win rate
if condition_1_trades:
    wins = sum(1 for t in condition_1_trades if t['return'] > 0)
    win_rate = wins / len(condition_1_trades) * 100
    avg_ret = np.mean([t['return'] for t in condition_1_trades])
    print(f"\n*** WIN RATE: {win_rate:.1f}% ({wins}/{len(condition_1_trades)}) ***")
    print(f"*** AVG RETURN: {avg_ret:+.2f}% ***")

print("\n" + "="*80)
print("  CONDITION: tx_z < -2 AND whale_z < -1.5 AND ret_7d < -10 (30-day hold)")
print("="*80)

condition_2_trades = []
for i in range(len(data) - 30):
    d = data[i]
    if d['tx_z'] < -2 and d['whale_z'] < -1.5 and d['ret_7d'] < -10:
        entry_price = d['price']
        exit_price = data[i + 30]['price']
        ret = (exit_price / entry_price - 1) * 100
        condition_2_trades.append({
            'date': d['date'],
            'exit_date': data[i + 30]['date'],
            'tx_z': d['tx_z'],
            'whale_z': d['whale_z'],
            'ret_7d': d['ret_7d'],
            'entry': entry_price,
            'exit': exit_price,
            'return': ret
        })

print(f"\nTotal trades found: {len(condition_2_trades)}")
print(f"Winning trades: {sum(1 for t in condition_2_trades if t['return'] > 0)}")
print(f"Losing trades: {sum(1 for t in condition_2_trades if t['return'] <= 0)}")

print("\nDETAILED TRADE BREAKDOWN:")
print("-"*80)
for t in condition_2_trades:
    win = "WIN" if t['return'] > 0 else "LOSS"
    print(f"\n{t['date']} -> {t['exit_date']}")
    print(f"  tx_z: {t['tx_z']:.3f}, whale_z: {t['whale_z']:.3f}, ret_7d: {t['ret_7d']:.1f}%")
    print(f"  Entry: ${t['entry']:,.2f} -> Exit: ${t['exit']:,.2f}")
    print(f"  Return: {t['return']:+.2f}% [{win}]")

if condition_2_trades:
    wins = sum(1 for t in condition_2_trades if t['return'] > 0)
    win_rate = wins / len(condition_2_trades) * 100
    avg_ret = np.mean([t['return'] for t in condition_2_trades])
    print(f"\n*** WIN RATE: {win_rate:.1f}% ({wins}/{len(condition_2_trades)}) ***")
    print(f"*** AVG RETURN: {avg_ret:+.2f}% ***")

print("\n" + "="*80)
print("  CONDITION: ALL z < -2 (30-day hold)")
print("="*80)

condition_3_trades = []
for i in range(len(data) - 30):
    d = data[i]
    if d['tx_z'] < -2 and d['whale_z'] < -2 and d['value_z'] < -2:
        entry_price = d['price']
        exit_price = data[i + 30]['price']
        ret = (exit_price / entry_price - 1) * 100
        condition_3_trades.append({
            'date': d['date'],
            'exit_date': data[i + 30]['date'],
            'tx_z': d['tx_z'],
            'whale_z': d['whale_z'],
            'value_z': d['value_z'],
            'entry': entry_price,
            'exit': exit_price,
            'return': ret
        })

print(f"\nTotal trades found: {len(condition_3_trades)}")
print(f"Winning trades: {sum(1 for t in condition_3_trades if t['return'] > 0)}")
print(f"Losing trades: {sum(1 for t in condition_3_trades if t['return'] <= 0)}")

print("\nDETAILED TRADE BREAKDOWN:")
print("-"*80)
for t in condition_3_trades:
    win = "WIN" if t['return'] > 0 else "LOSS"
    print(f"\n{t['date']} -> {t['exit_date']}")
    print(f"  tx_z: {t['tx_z']:.3f}, whale_z: {t['whale_z']:.3f}, value_z: {t['value_z']:.3f}")
    print(f"  Entry: ${t['entry']:,.2f} -> Exit: ${t['exit']:,.2f}")
    print(f"  Return: {t['return']:+.2f}% [{win}]")

if condition_3_trades:
    wins = sum(1 for t in condition_3_trades if t['return'] > 0)
    win_rate = wins / len(condition_3_trades) * 100
    avg_ret = np.mean([t['return'] for t in condition_3_trades])
    print(f"\n*** WIN RATE: {win_rate:.1f}% ({wins}/{len(condition_3_trades)}) ***")
    print(f"*** AVG RETURN: {avg_ret:+.2f}% ***")

# CRITICAL: Check for MAX DRAWDOWN during hold period
print("\n" + "="*80)
print("  CRITICAL: MAX DRAWDOWN DURING 30-DAY HOLD (LIQUIDATION CHECK)")
print("="*80)
print("\nAt 50x leverage, you get LIQUIDATED at 2% move against you!")

def check_max_drawdown(trades_list, condition_name):
    print(f"\n{condition_name}:")
    print("-"*60)

    safe_trades = 0
    risky_trades = 0

    for t in trades_list:
        entry_idx = date_idx[t['date']]
        entry_price = t['entry']

        # Find max drawdown during the 30-day period
        min_price = entry_price
        max_dd = 0
        worst_date = t['date']

        for j in range(entry_idx, entry_idx + 31):
            if j < len(data):
                curr_price = data[j]['price']
                if curr_price < min_price:
                    min_price = curr_price
                    dd = (entry_price - min_price) / entry_price * 100
                    if dd > max_dd:
                        max_dd = dd
                        worst_date = data[j]['date']

        # Would 50x get liquidated? (2% drawdown = liquidation)
        liq_50x = "LIQUIDATED!" if max_dd > 2 else "SAFE"
        liq_25x = "LIQUIDATED!" if max_dd > 4 else "SAFE"
        liq_10x = "LIQUIDATED!" if max_dd > 10 else "SAFE"

        print(f"\n{t['date']}: Entry ${entry_price:,.2f}")
        print(f"  Max drawdown: {max_dd:.2f}% (on {worst_date})")
        print(f"  50x: {liq_50x} | 25x: {liq_25x} | 10x: {liq_10x}")
        print(f"  Final return: {t['return']:+.2f}%")

        if max_dd <= 2:
            safe_trades += 1
        else:
            risky_trades += 1

    print(f"\nSUMMARY: {safe_trades} trades safe at 50x, {risky_trades} would get liquidated")

check_max_drawdown(condition_1_trades, "tx_z < -2 AND whale_z < -2")
check_max_drawdown(condition_2_trades, "tx_z < -2 AND whale_z < -1.5 AND ret_7d < -10")
check_max_drawdown(condition_3_trades, "ALL z < -2")

print("\n" + "="*80)
print("  FINAL VERDICT")
print("="*80)
