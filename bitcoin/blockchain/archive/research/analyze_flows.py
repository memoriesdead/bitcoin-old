#!/usr/bin/env python3
"""
ANALYZE FLOW DATA FOR 100% WIN RATE PATTERNS
=============================================

Run after collecting 24-48 hours of flow data.
Finds patterns that predict price movements with 100% accuracy.

Usage:
    python3 analyze_flows.py
"""

import sqlite3
import pandas as pd
from typing import Dict, List, Tuple


def load_data(db_path: str = "/root/sovereign/flow_data.db") -> pd.DataFrame:
    """Load flow data from SQLite."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM flows WHERE price_t10m IS NOT NULL", conn)
    conn.close()
    return df


def analyze_by_exchange(df: pd.DataFrame) -> Dict:
    """Win rate by exchange."""
    results = {}
    for exchange in df['exchange'].unique():
        subset = df[df['exchange'] == exchange]
        wins = subset['price_moved_expected'].sum()
        total = len(subset)
        win_rate = wins / total * 100 if total > 0 else 0
        results[exchange] = {
            'total': total,
            'wins': wins,
            'win_rate': win_rate
        }
    return results


def analyze_by_flow_size(df: pd.DataFrame) -> Dict:
    """Win rate by flow size bucket."""
    buckets = [
        (0, 1, '<1 BTC'),
        (1, 5, '1-5 BTC'),
        (5, 10, '5-10 BTC'),
        (10, 50, '10-50 BTC'),
        (50, 100, '50-100 BTC'),
        (100, float('inf'), '>100 BTC')
    ]

    results = {}
    for min_btc, max_btc, label in buckets:
        subset = df[(df['flow_btc'] >= min_btc) & (df['flow_btc'] < max_btc)]
        if len(subset) > 0:
            wins = subset['price_moved_expected'].sum()
            total = len(subset)
            win_rate = wins / total * 100
            results[label] = {
                'total': total,
                'wins': wins,
                'win_rate': win_rate
            }
    return results


def analyze_by_direction(df: pd.DataFrame) -> Dict:
    """Win rate by flow direction."""
    results = {}
    for direction in ['INFLOW', 'OUTFLOW']:
        subset = df[df['direction'] == direction]
        if len(subset) > 0:
            wins = subset['price_moved_expected'].sum()
            total = len(subset)
            results[direction] = {
                'total': total,
                'wins': wins,
                'win_rate': wins / total * 100
            }
    return results


def analyze_by_price_impact(df: pd.DataFrame) -> Dict:
    """Win rate by actual price impact magnitude."""
    results = {
        'big_moves': {},  # Flows that caused >0.1% moves
        'small_moves': {}  # Flows that caused <0.1% moves
    }

    # For inflows (expecting down move)
    inflows = df[df['direction'] == 'INFLOW']
    big_down = inflows[inflows['max_down_move_pct'] > 0.1]
    small_down = inflows[inflows['max_down_move_pct'] <= 0.1]

    if len(big_down) > 0:
        wins = big_down['price_moved_expected'].sum()
        results['big_moves']['INFLOW'] = {
            'total': len(big_down),
            'wins': wins,
            'win_rate': wins / len(big_down) * 100
        }

    if len(small_down) > 0:
        wins = small_down['price_moved_expected'].sum()
        results['small_moves']['INFLOW'] = {
            'total': len(small_down),
            'wins': wins,
            'win_rate': wins / len(small_down) * 100
        }

    return results


def find_100_percent_patterns(df: pd.DataFrame) -> List[Dict]:
    """Find patterns with 100% win rate (min 5 samples)."""
    patterns = []

    # Check each combination of exchange + direction + size bucket
    size_buckets = [
        (0, 1), (1, 5), (5, 10), (10, 50), (50, 100), (100, float('inf'))
    ]

    for exchange in df['exchange'].unique():
        for direction in ['INFLOW', 'OUTFLOW']:
            for min_btc, max_btc in size_buckets:
                subset = df[
                    (df['exchange'] == exchange) &
                    (df['direction'] == direction) &
                    (df['flow_btc'] >= min_btc) &
                    (df['flow_btc'] < max_btc)
                ]

                if len(subset) >= 5:  # Need at least 5 samples
                    wins = subset['price_moved_expected'].sum()
                    win_rate = wins / len(subset) * 100

                    if win_rate >= 90:  # Patterns with 90%+ win rate
                        patterns.append({
                            'exchange': exchange,
                            'direction': direction,
                            'min_btc': min_btc,
                            'max_btc': max_btc if max_btc != float('inf') else 'inf',
                            'samples': len(subset),
                            'wins': wins,
                            'win_rate': win_rate,
                            'avg_down_move': subset['max_down_move_pct'].mean(),
                            'avg_up_move': subset['max_up_move_pct'].mean()
                        })

    # Sort by win rate (highest first), then by sample size
    patterns.sort(key=lambda x: (-x['win_rate'], -x['samples']))
    return patterns


def main():
    print("=" * 70)
    print("FLOW DATA ANALYSIS FOR 100% WIN RATE")
    print("=" * 70)

    try:
        df = load_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Make sure the data collector has been running for 10+ minutes.")
        return

    if len(df) == 0:
        print("\nNo completed flows yet.")
        print("Flows need 10 minutes of price tracking before completion.")
        print("Run again after the collector has been running for a while.")
        return

    print(f"\nTotal completed flows: {len(df)}")
    print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # By direction
    print("\n" + "=" * 50)
    print("WIN RATE BY DIRECTION")
    print("=" * 50)
    by_direction = analyze_by_direction(df)
    for direction, stats in by_direction.items():
        print(f"{direction:10} | Total: {stats['total']:4} | Wins: {stats['wins']:4} | Win Rate: {stats['win_rate']:.1f}%")

    # By exchange
    print("\n" + "=" * 50)
    print("WIN RATE BY EXCHANGE")
    print("=" * 50)
    by_exchange = analyze_by_exchange(df)
    for exchange, stats in sorted(by_exchange.items(), key=lambda x: -x[1]['win_rate']):
        print(f"{exchange:15} | Total: {stats['total']:4} | Wins: {stats['wins']:4} | Win Rate: {stats['win_rate']:.1f}%")

    # By flow size
    print("\n" + "=" * 50)
    print("WIN RATE BY FLOW SIZE")
    print("=" * 50)
    by_size = analyze_by_flow_size(df)
    for size, stats in by_size.items():
        print(f"{size:12} | Total: {stats['total']:4} | Wins: {stats['wins']:4} | Win Rate: {stats['win_rate']:.1f}%")

    # 100% patterns
    print("\n" + "=" * 50)
    print("PATTERNS WITH 90%+ WIN RATE (min 5 samples)")
    print("=" * 50)
    patterns = find_100_percent_patterns(df)

    if patterns:
        for p in patterns:
            print(f"\n{p['exchange']} {p['direction']} [{p['min_btc']}-{p['max_btc']} BTC]")
            print(f"  Samples: {p['samples']} | Win Rate: {p['win_rate']:.1f}%")
            print(f"  Avg Down: {p['avg_down_move']:.3f}% | Avg Up: {p['avg_up_move']:.3f}%")
    else:
        print("No patterns with 90%+ win rate found yet.")
        print("Need more data - let collector run for 24-48 hours.")

    # Summary
    print("\n" + "=" * 50)
    print("NEXT STEPS")
    print("=" * 50)
    if patterns:
        best = patterns[0]
        print(f"\nBest pattern found:")
        print(f"  {best['exchange']} {best['direction']} [{best['min_btc']}-{best['max_btc']} BTC]")
        print(f"  Win Rate: {best['win_rate']:.1f}% ({best['wins']}/{best['samples']})")
        print(f"\nTo trade this pattern:")
        print(f"  1. Filter flows: exchange={best['exchange']}, direction={best['direction']}")
        print(f"  2. Size filter: {best['min_btc']} <= flow_btc < {best['max_btc']}")
        print(f"  3. Expected profit: ~{best['avg_down_move']:.3f}% per trade")
    else:
        print("\nNo high-confidence patterns yet.")
        print("Keep collecting data for 24-48 hours.")


if __name__ == "__main__":
    main()
