#!/usr/bin/env python3
"""
VPS SHORT TRADER TEST - Run on Hostinger
=========================================
Tests SHORT strategy against real correlation.db data.

Usage: python3 vps_test.py
"""

import sqlite3
import time

# VPS paths
CORRELATION_DB = "/root/sovereign/correlation.db"
TRADES_DB = "/root/sovereign/trades.db"


def get_inflow_signals():
    """Get all INFLOW signals from correlation.db."""
    try:
        conn = sqlite3.connect(CORRELATION_DB)
        cursor = conn.execute("""
            SELECT exchange, direction, flow_btc, price, timestamp
            FROM flows
            WHERE direction = 'INFLOW' OR direction = -1
            ORDER BY timestamp DESC
            LIMIT 100
        """)
        signals = cursor.fetchall()
        conn.close()
        return signals
    except Exception as e:
        print(f"Error reading correlation.db: {e}")
        return []


def get_short_trades():
    """Get all SHORT trades from trades.db."""
    try:
        conn = sqlite3.connect(TRADES_DB)
        cursor = conn.execute("""
            SELECT id, exchange, direction, entry_price, exit_price,
                   pnl_usd, status, exit_reason
            FROM trades
            WHERE direction = 'SHORT' OR direction = -1
            ORDER BY id DESC
        """)
        trades = cursor.fetchall()
        conn.close()
        return trades
    except Exception as e:
        print(f"Error reading trades.db: {e}")
        return []


def get_pattern_stats():
    """Get pattern statistics for SHORT/INFLOW."""
    try:
        conn = sqlite3.connect(CORRELATION_DB)
        cursor = conn.execute("""
            SELECT exchange, direction, correlation, win_rate, samples
            FROM patterns
            WHERE direction = 'INFLOW' OR direction = -1
            ORDER BY samples DESC
        """)
        patterns = cursor.fetchall()
        conn.close()
        return patterns
    except Exception as e:
        print(f"Error reading patterns: {e}")
        return []


def main():
    print("=" * 60)
    print("VPS SHORT TRADER TEST")
    print("=" * 60)
    print()

    # Check INFLOW signals
    print("INFLOW SIGNALS (last 100):")
    print("-" * 40)
    signals = get_inflow_signals()
    if signals:
        for s in signals[:10]:
            print(f"  {s[0]:12} | {s[2]:8.2f} BTC | ${s[3]:,.2f}")
        print(f"  ... {len(signals)} total INFLOW signals")
    else:
        print("  No signals found")
    print()

    # Check SHORT trades
    print("SHORT TRADES:")
    print("-" * 40)
    trades = get_short_trades()
    wins = 0
    losses = 0
    total_pnl = 0

    if trades:
        for t in trades:
            pnl = t[5] or 0
            total_pnl += pnl
            if pnl > 0:
                wins += 1
            else:
                losses += 1
            status = "WIN" if pnl > 0 else "LOSS"
            print(f"  #{t[0]:4} | {t[1]:12} | ${t[3]:,.2f} -> ${t[4]:,.2f} | ${pnl:+.2f} | {status}")

        total = wins + losses
        win_rate = wins / total * 100 if total > 0 else 0

        print()
        print(f"  Total: {total} trades")
        print(f"  Wins: {wins} | Losses: {losses}")
        print(f"  Win Rate: {win_rate:.1f}%")
        print(f"  Total P&L: ${total_pnl:+,.2f}")
    else:
        print("  No SHORT trades found")
    print()

    # Check pattern statistics
    print("PATTERN STATISTICS:")
    print("-" * 40)
    patterns = get_pattern_stats()
    if patterns:
        for p in patterns[:10]:
            print(f"  {p[0]:12} | corr: {p[2]:.2f} | win: {p[3]*100:.1f}% | samples: {p[4]}")
    else:
        print("  No patterns found")
    print()

    print("=" * 60)
    if trades and win_rate == 100:
        print("SUCCESS: 100% WIN RATE ON SHORT TRADES")
    elif trades:
        print(f"Win Rate: {win_rate:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
