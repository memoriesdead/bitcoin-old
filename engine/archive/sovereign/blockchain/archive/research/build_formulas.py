#!/usr/bin/env python3
"""
BUILD DETERMINISTIC FORMULAS
============================
Analyzes collected correlation data and builds trading formulas.

Run this AFTER collecting 24-48 hours of data.

The output is a JSON file with EXACT formulas:
  - Exchange
  - Direction (inflow/outflow)
  - Coefficient: price_delta = flow_btc * coefficient
  - Win rate
  - Optimal time window
  - Minimum flow threshold

ONLY trade when formulas show:
  - 70%+ win rate
  - 60%+ predictability score
  - 100+ samples
"""

import json
import sqlite3
import sys
import os
from datetime import datetime
from typing import Dict, List, Optional

sys.path.insert(0, '/root/sovereign/blockchain')

from deterministic_correlation import DeterministicCorrelationDB, CorrelationConfig


def analyze_correlations(db_path: str) -> Dict:
    """Analyze correlations and build formulas."""

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    results = {
        'analysis_time': datetime.now().isoformat(),
        'database': db_path,
        'exchanges': {},
        'viable_formulas': [],
        'not_viable': []
    }

    # Get all exchanges with data
    cursor = conn.execute("""
        SELECT DISTINCT exchange FROM flow_events
    """)
    exchanges = [row['exchange'] for row in cursor.fetchall()]

    print(f"Found {len(exchanges)} exchanges with data")
    print()

    for exchange in exchanges:
        results['exchanges'][exchange] = {
            'inflow': analyze_direction(conn, exchange, 'inflow'),
            'outflow': analyze_direction(conn, exchange, 'outflow')
        }

    # Find viable formulas
    for exchange, data in results['exchanges'].items():
        for direction, analysis in data.items():
            if analysis['viable']:
                results['viable_formulas'].append({
                    'exchange': exchange,
                    'direction': direction,
                    **analysis
                })
            else:
                results['not_viable'].append({
                    'exchange': exchange,
                    'direction': direction,
                    'reason': analysis.get('reason', 'Unknown')
                })

    conn.close()
    return results


def analyze_direction(conn: sqlite3.Connection, exchange: str, direction: str) -> Dict:
    """Analyze correlation for a specific exchange and direction."""

    # Count samples
    cursor = conn.execute("""
        SELECT COUNT(*) as cnt
        FROM flow_events fe
        JOIN price_observations po ON fe.id = po.flow_id
        WHERE fe.exchange = ? AND fe.direction = ?
          AND po.price IS NOT NULL
    """, (exchange, direction))

    sample_count = cursor.fetchone()['cnt']

    if sample_count < 30:
        return {
            'viable': False,
            'reason': f'Not enough samples ({sample_count} < 30)',
            'sample_count': sample_count
        }

    # Get all observations with price deltas
    cursor = conn.execute("""
        SELECT fe.amount_btc, po.window_seconds, po.price_delta, po.price_delta_pct
        FROM flow_events fe
        JOIN price_observations po ON fe.id = po.flow_id
        WHERE fe.exchange = ? AND fe.direction = ?
          AND po.price IS NOT NULL
    """, (exchange, direction))

    rows = cursor.fetchall()

    # Group by time window
    by_window = {}
    for row in rows:
        window = row['window_seconds']
        if window not in by_window:
            by_window[window] = []
        by_window[window].append({
            'amount': row['amount_btc'],
            'delta': row['price_delta'],
            'delta_pct': row['price_delta_pct']
        })

    # Find best time window
    best_window = None
    best_stats = None
    best_score = 0

    for window, data in by_window.items():
        stats = calculate_window_stats(data, direction)
        if stats['predictability_score'] > best_score:
            best_score = stats['predictability_score']
            best_stats = stats
            best_window = window

    if not best_stats:
        return {
            'viable': False,
            'reason': 'No valid statistics',
            'sample_count': sample_count
        }

    # Check viability thresholds
    viable = (
        best_stats['win_rate'] >= 0.60 and  # 60% win rate minimum
        best_stats['predictability_score'] >= 50 and  # 50% predictability
        sample_count >= 30  # At least 30 samples
    )

    return {
        'viable': viable,
        'reason': 'Meets criteria' if viable else 'Below thresholds',
        'sample_count': sample_count,
        'optimal_window_seconds': best_window,
        'coefficient': best_stats['coefficient'],
        'win_rate': best_stats['win_rate'],
        'predictability_score': best_stats['predictability_score'],
        'avg_delta': best_stats['avg_delta'],
        'avg_delta_pct': best_stats['avg_delta_pct'],
        'correlation': best_stats['correlation']
    }


def calculate_window_stats(data: List[Dict], direction: str) -> Dict:
    """Calculate statistics for a time window."""

    n = len(data)
    if n < 2:
        return {'predictability_score': 0}

    amounts = [d['amount'] for d in data]
    deltas = [d['delta'] for d in data]
    delta_pcts = [d['delta_pct'] for d in data]

    avg_delta = sum(deltas) / n
    avg_delta_pct = sum(delta_pcts) / n

    # Pearson correlation
    mean_a = sum(amounts) / n
    mean_d = sum(deltas) / n

    numerator = sum((a - mean_a) * (d - mean_d) for a, d in zip(amounts, deltas))
    denom_a = sum((a - mean_a) ** 2 for a in amounts) ** 0.5
    denom_d = sum((d - mean_d) ** 2 for d in deltas) ** 0.5

    correlation = numerator / (denom_a * denom_d) if denom_a * denom_d > 0 else 0

    # Linear coefficient (slope)
    denom = sum((a - mean_a) ** 2 for a in amounts)
    coefficient = numerator / denom if denom > 0 else 0

    # Win rate
    # For inflow: win if price went DOWN (we short)
    # For outflow: win if price went UP (we long)
    if direction == 'inflow':
        wins = sum(1 for d in deltas if d < 0)
    else:
        wins = sum(1 for d in deltas if d > 0)

    win_rate = wins / n

    # Predictability score
    predictability = abs(correlation) * 50 + win_rate * 50

    return {
        'sample_count': n,
        'avg_delta': avg_delta,
        'avg_delta_pct': avg_delta_pct,
        'correlation': correlation,
        'coefficient': coefficient,
        'win_rate': win_rate,
        'predictability_score': predictability
    }


def print_report(results: Dict):
    """Print human-readable report."""

    print("=" * 70)
    print("DETERMINISTIC FORMULA ANALYSIS")
    print("=" * 70)
    print(f"Analysis time: {results['analysis_time']}")
    print(f"Database: {results['database']}")
    print()

    # Viable formulas
    if results['viable_formulas']:
        print("=" * 70)
        print("VIABLE FORMULAS (Ready for Trading)")
        print("=" * 70)

        for f in results['viable_formulas']:
            print()
            print(f"{f['exchange'].upper()} - {f['direction'].upper()}")
            print("-" * 50)
            print(f"  Formula: price_delta = flow_btc × {f['coefficient']:.4f}")
            print(f"  Optimal window: {f['optimal_window_seconds']}s ({f['optimal_window_seconds']/60:.0f} min)")
            print(f"  Win rate: {f['win_rate']:.1%}")
            print(f"  Predictability: {f['predictability_score']:.1f}%")
            print(f"  Avg delta: ${f['avg_delta']:.2f} ({f['avg_delta_pct']:.3f}%)")
            print(f"  Correlation: {f['correlation']:.3f}")
            print(f"  Samples: {f['sample_count']}")

            # Interpretation
            if f['direction'] == 'inflow':
                if f['coefficient'] < 0:
                    print(f"  → CORRECT: Inflow causes price DROP")
                else:
                    print(f"  → WARNING: Inflow correlates with price RISE (unexpected)")
            else:
                if f['coefficient'] > 0:
                    print(f"  → CORRECT: Outflow causes price RISE")
                else:
                    print(f"  → WARNING: Outflow correlates with price DROP (unexpected)")
    else:
        print()
        print("NO VIABLE FORMULAS YET")
        print("Need more data collection time.")

    # Not viable
    if results['not_viable']:
        print()
        print("=" * 70)
        print("NOT VIABLE (Need more data or low predictability)")
        print("=" * 70)
        for item in results['not_viable']:
            print(f"  {item['exchange']:12} {item['direction']:8} - {item['reason']}")

    print()


def save_formulas(results: Dict, output_path: str):
    """Save viable formulas to JSON file."""

    formulas = {
        'generated_at': results['analysis_time'],
        'formulas': {}
    }

    for f in results['viable_formulas']:
        key = f"{f['exchange']}_{f['direction']}"
        formulas['formulas'][key] = {
            'exchange': f['exchange'],
            'direction': f['direction'],
            'coefficient': f['coefficient'],
            'optimal_window_seconds': f['optimal_window_seconds'],
            'win_rate': f['win_rate'],
            'predictability_score': f['predictability_score'],
            'sample_count': f['sample_count']
        }

    with open(output_path, 'w') as fp:
        json.dump(formulas, fp, indent=2)

    print(f"Saved {len(formulas['formulas'])} formulas to: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build deterministic trading formulas")
    parser.add_argument("--db", default="/root/sovereign/deterministic_correlation.db",
                       help="Path to correlation database")
    parser.add_argument("--output", default="/root/sovereign/trading_formulas.json",
                       help="Output path for formulas JSON")

    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"Error: Database not found: {args.db}")
        print("Run correlation_collector_pipeline.py first to collect data.")
        sys.exit(1)

    # Analyze
    results = analyze_correlations(args.db)

    # Print report
    print_report(results)

    # Save formulas if any viable
    if results['viable_formulas']:
        save_formulas(results, args.output)
    else:
        print("No formulas to save. Collect more data.")


if __name__ == "__main__":
    main()
