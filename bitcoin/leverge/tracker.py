#!/usr/bin/env python3
"""
100% WIN RATE TRACKER

Tracks patterns and trades that meet the 100% threshold.
Only patterns with win_rate = 1.0 and sample_count >= 30 qualify.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict


class PerfectPatternTracker:
    """Track and validate 100% win rate patterns."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.trades_file = Path(__file__).parent / "trades.json"
        self._init_trades_file()

    def _init_trades_file(self):
        """Initialize trades JSON file if not exists."""
        if not self.trades_file.exists():
            self.trades_file.write_text(json.dumps({
                "trades": [],
                "total_pnl": 0.0,
                "win_count": 0,
                "loss_count": 0
            }, indent=2))

    def get_perfect_patterns(self, min_samples: int = 30) -> List[Dict]:
        """
        Query correlation.db for patterns with 100% win rate.

        Returns list of patterns that qualify for max leverage trading.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT exchange, direction, bucket_min, bucket_max,
                       sample_count, win_rate, correlation, avg_price_change
                FROM patterns
                WHERE win_rate = 1.0 AND sample_count >= ?
                ORDER BY sample_count DESC
            """, (min_samples,))

            rows = cursor.fetchall()
            conn.close()

            patterns = []
            for row in rows:
                patterns.append({
                    'exchange': row[0],
                    'direction': row[1],
                    'bucket': (row[2], row[3]),
                    'sample_count': row[4],
                    'win_rate': row[5],
                    'correlation': row[6],
                    'avg_move': row[7]
                })

            return patterns

        except Exception as e:
            print(f"Error querying patterns: {e}")
            return []

    def is_perfect_pattern(self, exchange: str, direction: str, bucket: tuple) -> bool:
        """Check if a specific pattern has 100% win rate."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT win_rate, sample_count
                FROM patterns
                WHERE exchange = ? AND direction = ?
                      AND bucket_min = ? AND bucket_max = ?
            """, (exchange.lower(), direction.upper(), bucket[0], bucket[1]))

            row = cursor.fetchone()
            conn.close()

            if row:
                win_rate, sample_count = row
                return win_rate == 1.0 and sample_count >= 30
            return False

        except Exception:
            return False

    def log_trade(self, trade: Dict):
        """Log a 100% pattern trade."""
        data = json.loads(self.trades_file.read_text())

        trade['timestamp'] = datetime.now().isoformat()
        data['trades'].append(trade)

        if trade.get('pnl', 0) > 0:
            data['win_count'] += 1
        else:
            data['loss_count'] += 1

        data['total_pnl'] += trade.get('pnl', 0)

        self.trades_file.write_text(json.dumps(data, indent=2))

    def get_stats(self) -> Dict:
        """Get trading statistics."""
        data = json.loads(self.trades_file.read_text())
        total = data['win_count'] + data['loss_count']

        return {
            'total_trades': total,
            'wins': data['win_count'],
            'losses': data['loss_count'],
            'win_rate': data['win_count'] / total if total > 0 else 0,
            'total_pnl': data['total_pnl']
        }

    def print_qualified_patterns(self):
        """Print all patterns that qualify for 100% trading."""
        patterns = self.get_perfect_patterns()

        if not patterns:
            print("\n" + "="*60)
            print("NO PATTERNS WITH 100% WIN RATE (>=30 samples)")
            print("="*60)
            print("\nData collection continues. Patterns will emerge.")
            print("The market will reveal certainty. We wait.\n")
            return

        print("\n" + "="*60)
        print("100% WIN RATE PATTERNS - QUALIFIED FOR MAX LEVERAGE")
        print("="*60)

        for p in patterns:
            print(f"\n{p['exchange'].upper()} {p['direction']} {p['bucket']}")
            print(f"  Samples:  {p['sample_count']}")
            print(f"  Win Rate: {p['win_rate']:.0%}")
            print(f"  Avg Move: {p['avg_move']:.2%}")

        print("\n" + "="*60)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import get_config

    config = get_config()
    tracker = PerfectPatternTracker(config.correlation_db_path)
    tracker.print_qualified_patterns()
