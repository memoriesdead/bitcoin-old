#!/usr/bin/env python3
"""
EXTRACT OPTIMAL LEVERAGE CONDITIONS
====================================
Find EXACTLY when 50x leverage is profitable.
RenTech approach: Only bet big when edge is proven.
"""

import sqlite3
import numpy as np
from pathlib import Path

DATA_DIR = Path('/root/validation/data')

def load_data():
    conn = sqlite3.connect(DATA_DIR / 'metrics.db')
    cursor = conn.execute('''
        SELECT timestamp, tx_count, total_volume_btc, tx_whale,
               tx_mega, consolidation_ratio, price
        FROM metrics WHERE price > 0
        ORDER BY timestamp
    ''')
    data = cursor.fetchall()
    conn.close()
    return np.array(data)

def analyze_conditions(data):
    n = len(data)
    ts = data[:, 0]
    whale = data[:, 3].astype(int)
    mega = data[:, 4].astype(int)
    price = data[:, 6].astype(float)

    results = []

    print('='*70)
    print('  EXTRACTING OPTIMAL LEVERAGE CONDITIONS')
    print('  RenTech Approach: Only use 50x when confidence is MAXIMUM')
    print('='*70)
    print('Data points: {:,}'.format(n))
    print('Price range: ${:,.0f} - ${:,.0f}'.format(price.min(), price.max()))
    print()

    # Test different whale thresholds AND hold times
    for whale_thresh in [1, 2, 3, 5, 10, 20, 50]:
        for hold_secs in [60, 120, 300, 600]:  # 1, 2, 5, 10 min

            signal_mask = whale >= whale_thresh
            signal_indices = np.where(signal_mask)[0]

            if len(signal_indices) < 5:
                continue

            returns_long = []
            returns_short = []

            for idx in signal_indices:
                future_mask = ts > ts[idx] + hold_secs
                if not np.any(future_mask):
                    continue
                future_idx = np.where(future_mask)[0][0]

                entry_price = price[idx]
                exit_price = price[future_idx]

                ret_bps = (exit_price / entry_price - 1) * 10000
                returns_long.append(ret_bps)
                returns_short.append(-ret_bps)

            if len(returns_long) < 5:
                continue

            long_arr = np.array(returns_long)
            long_mean = np.mean(long_arr)
            long_std = np.std(long_arr)
            long_wins = np.sum(long_arr > 0) / len(long_arr) * 100

            short_arr = np.array(returns_short)
            short_mean = np.mean(short_arr)
            short_std = np.std(short_arr)
            short_wins = np.sum(short_arr > 0) / len(short_arr) * 100

            # Net after 8 bps fees (maker round trip)
            long_net = long_mean - 8
            short_net = short_mean - 8

            # Kelly fraction: edge / variance
            long_kelly = long_net / (long_std**2) if long_std > 0 else 0
            short_kelly = short_net / (short_std**2) if short_std > 0 else 0

            # Optimal leverage (capped at 50)
            long_opt_lev = min(50, max(0, long_kelly * 100))
            short_opt_lev = min(50, max(0, short_kelly * 100))

            results.append({
                'whale': whale_thresh,
                'hold': hold_secs,
                'n': len(returns_long),
                'long_net': long_net,
                'long_win': long_wins,
                'long_std': long_std,
                'long_lev': long_opt_lev,
                'short_net': short_net,
                'short_win': short_wins,
                'short_std': short_std,
                'short_lev': short_opt_lev
            })

    # Sort by best net return
    results.sort(key=lambda x: max(x['long_net'], x['short_net']), reverse=True)

    print('TOP CONDITIONS FOR LEVERAGE:')
    print('-'*70)
    print('{:>6} {:>6} {:>6} | {:>10} {:>6} {:>5} | {:>10} {:>6} {:>5}'.format(
        'Whale', 'Hold', 'N', 'LONG', 'Win%', 'Lev', 'SHORT', 'Win%', 'Lev'))
    print('-'*70)

    for r in results[:20]:
        hold_str = '{}m'.format(r['hold']//60)
        print('{:>6} {:>6} {:>6} | {:>+10.1f} {:>5.0f}% {:>5.1f} | {:>+10.1f} {:>5.0f}% {:>5.1f}'.format(
            r['whale'], hold_str, r['n'],
            r['long_net'], r['long_win'], r['long_lev'],
            r['short_net'], r['short_win'], r['short_lev']))

    # Find THE BEST conditions
    best_long = max(results, key=lambda x: x['long_net'])
    best_short = max(results, key=lambda x: x['short_net'])

    print()
    print('='*70)
    print('  OPTIMAL LEVERAGE CONDITIONS')
    print('='*70)

    if best_long['long_net'] > 0:
        print('LONG: whale >= {}, hold {}m'.format(best_long['whale'], best_long['hold']//60))
        print('  Edge: {:+.1f} bps | Win: {:.0f}% | Std: {:.1f} bps'.format(
            best_long['long_net'], best_long['long_win'], best_long['long_std']))
        print('  Optimal Kelly Leverage: {:.1f}x'.format(best_long['long_lev']))
        print('  Trades in dataset: {}'.format(best_long['n']))
    else:
        print('LONG: No profitable condition found')

    print()

    if best_short['short_net'] > 0:
        print('SHORT: whale >= {}, hold {}m'.format(best_short['whale'], best_short['hold']//60))
        print('  Edge: {:+.1f} bps | Win: {:.0f}% | Std: {:.1f} bps'.format(
            best_short['short_net'], best_short['short_win'], best_short['short_std']))
        print('  Optimal Kelly Leverage: {:.1f}x'.format(best_short['short_lev']))
        print('  Trades in dataset: {}'.format(best_short['n']))
    else:
        print('SHORT: No profitable condition found')

    # Compound simulation
    print()
    print('='*70)
    print('  COMPOUNDING SIMULATION ($100 -> ???)')
    print('  Using HALF Kelly for safety')
    print('='*70)

    # Pick best overall
    if best_long['long_net'] > best_short['short_net'] and best_long['long_net'] > 0:
        best = best_long
        direction = 'LONG'
        edge = best['long_net']
        opt_lev = best['long_lev']
    elif best_short['short_net'] > 0:
        best = best_short
        direction = 'SHORT'
        edge = best['short_net']
        opt_lev = best['short_lev']
    else:
        print('No profitable edge found.')
        return

    # Use half Kelly for safety
    safe_lev = min(opt_lev / 2, 25)  # Cap at 25x for half Kelly
    ret_per_trade = edge * safe_lev / 10000

    # Estimate trades per day
    hours_of_data = (ts[-1] - ts[0]) / 3600
    trades_per_hour = best['n'] / hours_of_data
    trades_per_day = trades_per_hour * 24

    print('Strategy: {} when whale >= {}, hold {}m'.format(direction, best['whale'], best['hold']//60))
    print('Edge: {:+.1f} bps per trade'.format(edge))
    print('Half-Kelly leverage: {:.1f}x'.format(safe_lev))
    print('Return per trade: {:+.3f}%'.format(ret_per_trade * 100))
    print('Expected trades/day: {:.1f}'.format(trades_per_day))
    print()

    # Simulate compounding
    for days in [1, 7, 30, 90, 365]:
        capital = 100.0
        total_trades = int(trades_per_day * days)
        for _ in range(total_trades):
            capital *= (1 + ret_per_trade)
        print('After {:>3} days ({:>5} trades): ${:>20,.2f}'.format(days, total_trades, capital))

    print()
    print('='*70)
    print('  CRITICAL: This assumes edge persists. Real trading has:')
    print('  - Slippage, execution delays, API failures')
    print('  - Edge decay as others discover same signal')
    print('  - Black swan events that wipe leveraged positions')
    print('='*70)

if __name__ == '__main__':
    data = load_data()
    analyze_conditions(data)
