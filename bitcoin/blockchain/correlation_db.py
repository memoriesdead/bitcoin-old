#!/usr/bin/env python3
"""
CORRELATION DATABASE
====================
Phase 4: Track flow → price correlation per exchange.

For each detected flow, store:
- Exchange
- Direction (INFLOW/OUTFLOW)
- Amount (BTC)
- Price at T=0
- Price at T+30s
- Price at T+60s
- Price at T+5m
- Calculated correlation

This builds the data needed to discover causation patterns per exchange.
"""

import sys
import time
import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

sys.path.insert(0, '/root/sovereign')
sys.path.insert(0, '/root/sovereign/blockchain')

try:
    # VPS path
    from multi_price_feed import MultiExchangePriceFeed, get_exchange_price
except ImportError:
    try:
        # Local path via core/
        from bitcoin.core import MultiExchangePriceFeed, get_exchange_price
    except ImportError:
        # Legacy fallback
        from blockchain.multi_price_feed import MultiExchangePriceFeed, get_exchange_price


@dataclass
class FlowRecord:
    """A single flow event with price tracking."""
    flow_id: int
    timestamp: float
    exchange: str
    direction: str  # 'INFLOW' or 'OUTFLOW'
    amount_btc: float
    txid: str

    # Prices at different times
    price_t0: Optional[float] = None
    price_t30: Optional[float] = None
    price_t60: Optional[float] = None
    price_t300: Optional[float] = None  # 5 minutes

    # Calculated metrics
    change_30s: Optional[float] = None  # price change in 30s (%)
    change_60s: Optional[float] = None
    change_5m: Optional[float] = None
    direction_correct: Optional[bool] = None  # Did price move as expected?


class CorrelationDatabase:
    """
    SQLite database for flow → price correlation tracking.

    Schema:
        flows(
            id INTEGER PRIMARY KEY,
            timestamp REAL,
            exchange TEXT,
            direction TEXT,
            amount_btc REAL,
            txid TEXT,
            price_t0 REAL,
            price_t30 REAL,
            price_t60 REAL,
            price_t300 REAL,
            change_30s REAL,
            change_60s REAL,
            change_5m REAL,
            direction_correct INTEGER,
            verified_at TEXT
        )
    """

    def __init__(self, db_path: str = "/root/sovereign/correlation.db"):
        self.db_path = Path(db_path)
        self.lock = threading.Lock()
        self.price_feed = MultiExchangePriceFeed()

        # Pending verifications (flow_id, check_time, check_type)
        self.pending: List[Tuple[int, float, str]] = []

        self._init_db()

    def _init_db(self):
        """Initialize database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS flows (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                exchange TEXT,
                direction TEXT,
                amount_btc REAL,
                txid TEXT,
                price_t0 REAL,
                price_t30 REAL,
                price_t60 REAL,
                price_t300 REAL,
                change_30s REAL,
                change_60s REAL,
                change_5m REAL,
                direction_correct INTEGER,
                verified_at TEXT
            )
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_exchange ON flows(exchange)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_direction ON flows(direction)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON flows(timestamp)")

        # Aggregated stats table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS exchange_stats (
                exchange TEXT PRIMARY KEY,
                total_inflows INTEGER DEFAULT 0,
                total_outflows INTEGER DEFAULT 0,
                inflow_btc REAL DEFAULT 0,
                outflow_btc REAL DEFAULT 0,
                avg_change_30s_inflow REAL,
                avg_change_30s_outflow REAL,
                avg_change_5m_inflow REAL,
                avg_change_5m_outflow REAL,
                inflow_correct_pct REAL,
                outflow_correct_pct REAL,
                updated_at TEXT
            )
        """)

        conn.commit()
        conn.close()
        print(f"[CORR_DB] Initialized at {self.db_path}")

    def record_flow(self, exchange: str, direction: str, amount_btc: float,
                    txid: str, price_now: Optional[float] = None) -> int:
        """
        Record a new flow event.

        Args:
            exchange: Exchange name
            direction: 'INFLOW' or 'OUTFLOW'
            amount_btc: Flow amount in BTC
            txid: Transaction ID
            price_now: Current price (fetched if not provided)

        Returns:
            Flow ID
        """
        if price_now is None:
            price_now = self.price_feed.get_price(exchange)

        now = time.time()

        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO flows (timestamp, exchange, direction, amount_btc, txid, price_t0)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (now, exchange, direction, amount_btc, txid, price_now))

            flow_id = cursor.lastrowid
            conn.commit()
            conn.close()

        # Schedule price checks
        self.pending.append((flow_id, now + 30, 't30'))
        self.pending.append((flow_id, now + 60, 't60'))
        self.pending.append((flow_id, now + 300, 't300'))

        return flow_id

    def check_pending_verifications(self):
        """Check and update pending price verifications."""
        now = time.time()
        still_pending = []

        for flow_id, check_time, check_type in self.pending:
            if now >= check_time:
                self._verify_price(flow_id, check_type)
            else:
                still_pending.append((flow_id, check_time, check_type))

        self.pending = still_pending

    def _verify_price(self, flow_id: int, check_type: str):
        """Verify price at a specific time offset."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get flow record
            cursor.execute("""
                SELECT exchange, direction, price_t0 FROM flows WHERE id = ?
            """, (flow_id,))
            row = cursor.fetchone()

            if not row:
                conn.close()
                return

            exchange, direction, price_t0 = row

            # Get current price
            price_now = self.price_feed.get_price(exchange)

            if price_now is None or price_t0 is None:
                conn.close()
                return

            # Calculate change
            change_pct = ((price_now - price_t0) / price_t0) * 100

            # Update record
            if check_type == 't30':
                cursor.execute("""
                    UPDATE flows SET price_t30 = ?, change_30s = ? WHERE id = ?
                """, (price_now, change_pct, flow_id))
            elif check_type == 't60':
                cursor.execute("""
                    UPDATE flows SET price_t60 = ?, change_60s = ? WHERE id = ?
                """, (price_now, change_pct, flow_id))
            elif check_type == 't300':
                # Final verification - also check if direction was correct
                # INFLOW (deposit) → expect price DOWN → change should be negative
                # OUTFLOW (withdrawal) → expect price UP → change should be positive
                if direction == 'INFLOW':
                    correct = change_pct < 0
                else:  # OUTFLOW
                    correct = change_pct > 0

                cursor.execute("""
                    UPDATE flows SET price_t300 = ?, change_5m = ?,
                           direction_correct = ?, verified_at = ?
                    WHERE id = ?
                """, (price_now, change_pct, 1 if correct else 0,
                      datetime.now().isoformat(), flow_id))

            conn.commit()
            conn.close()

    def get_exchange_correlation(self, exchange: str) -> Dict:
        """
        Get correlation statistics for an exchange.

        Returns:
            {
                'total_flows': int,
                'inflows': {'count': int, 'btc': float, 'avg_change_30s': float, ...},
                'outflows': {'count': int, 'btc': float, 'avg_change_30s': float, ...},
                'accuracy_30s': float,
                'accuracy_5m': float,
            }
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        result = {
            'exchange': exchange,
            'inflows': {},
            'outflows': {},
        }

        # Inflows
        cursor.execute("""
            SELECT COUNT(*), SUM(amount_btc), AVG(change_30s), AVG(change_60s), AVG(change_5m),
                   SUM(CASE WHEN direction_correct = 1 THEN 1 ELSE 0 END),
                   SUM(CASE WHEN direction_correct IS NOT NULL THEN 1 ELSE 0 END)
            FROM flows WHERE exchange = ? AND direction = 'INFLOW'
        """, (exchange,))
        row = cursor.fetchone()
        if row and row[0]:
            verified = row[6] or 0
            correct = row[5] or 0
            result['inflows'] = {
                'count': row[0],
                'btc': row[1] or 0,
                'avg_change_30s': row[2],
                'avg_change_60s': row[3],
                'avg_change_5m': row[4],
                'accuracy': (correct / verified * 100) if verified > 0 else None,
                'verified': verified,
            }

        # Outflows
        cursor.execute("""
            SELECT COUNT(*), SUM(amount_btc), AVG(change_30s), AVG(change_60s), AVG(change_5m),
                   SUM(CASE WHEN direction_correct = 1 THEN 1 ELSE 0 END),
                   SUM(CASE WHEN direction_correct IS NOT NULL THEN 1 ELSE 0 END)
            FROM flows WHERE exchange = ? AND direction = 'OUTFLOW'
        """, (exchange,))
        row = cursor.fetchone()
        if row and row[0]:
            verified = row[6] or 0
            correct = row[5] or 0
            result['outflows'] = {
                'count': row[0],
                'btc': row[1] or 0,
                'avg_change_30s': row[2],
                'avg_change_60s': row[3],
                'avg_change_5m': row[4],
                'accuracy': (correct / verified * 100) if verified > 0 else None,
                'verified': verified,
            }

        conn.close()

        # Calculate overall
        total_verified = (result['inflows'].get('verified', 0) +
                          result['outflows'].get('verified', 0))
        total_correct = 0
        if result['inflows'].get('accuracy') is not None:
            total_correct += result['inflows']['verified'] * result['inflows']['accuracy'] / 100
        if result['outflows'].get('accuracy') is not None:
            total_correct += result['outflows']['verified'] * result['outflows']['accuracy'] / 100

        result['overall_accuracy'] = (total_correct / total_verified * 100) if total_verified > 0 else None
        result['total_verified'] = total_verified

        return result

    def get_all_correlations(self) -> Dict[str, Dict]:
        """Get correlation stats for all exchanges."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT DISTINCT exchange FROM flows")
        exchanges = [row[0] for row in cursor.fetchall()]
        conn.close()

        return {ex: self.get_exchange_correlation(ex) for ex in exchanges}

    def get_causation_patterns(self, exchange: str, min_samples: int = 10) -> Dict:
        """
        Analyze causation patterns for an exchange.

        Returns patterns like:
        - What flow size consistently moves price?
        - What is the typical time delay?
        - What is the average magnitude?
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        patterns = {
            'exchange': exchange,
            'inflow_patterns': [],
            'outflow_patterns': [],
        }

        # Analyze inflows by size buckets
        for direction in ['INFLOW', 'OUTFLOW']:
            cursor.execute("""
                SELECT
                    CASE
                        WHEN amount_btc < 1 THEN 'tiny_<1'
                        WHEN amount_btc < 10 THEN 'small_1-10'
                        WHEN amount_btc < 100 THEN 'medium_10-100'
                        WHEN amount_btc < 1000 THEN 'large_100-1000'
                        ELSE 'whale_1000+'
                    END as size_bucket,
                    COUNT(*) as count,
                    AVG(change_30s) as avg_30s,
                    AVG(change_60s) as avg_60s,
                    AVG(change_5m) as avg_5m,
                    SUM(CASE WHEN direction_correct = 1 THEN 1 ELSE 0 END) as correct,
                    SUM(CASE WHEN direction_correct IS NOT NULL THEN 1 ELSE 0 END) as verified
                FROM flows
                WHERE exchange = ? AND direction = ? AND change_5m IS NOT NULL
                GROUP BY size_bucket
                HAVING count >= ?
            """, (exchange, direction, min_samples))

            key = 'inflow_patterns' if direction == 'INFLOW' else 'outflow_patterns'
            for row in cursor.fetchall():
                patterns[key].append({
                    'size_bucket': row[0],
                    'count': row[1],
                    'avg_change_30s': row[2],
                    'avg_change_60s': row[3],
                    'avg_change_5m': row[4],
                    'accuracy': (row[5] / row[6] * 100) if row[6] > 0 else None,
                })

        conn.close()
        return patterns

    def print_report(self):
        """Print correlation report."""
        print()
        print("=" * 70)
        print("CORRELATION DATABASE REPORT")
        print("=" * 70)

        correlations = self.get_all_correlations()

        if not correlations:
            print("No data yet.")
            return

        print(f"\n{'Exchange':<15} {'Inflows':>8} {'Outflows':>8} {'Verified':>10} {'Accuracy':>10}")
        print("-" * 55)

        for ex, data in sorted(correlations.items()):
            inflows = data['inflows'].get('count', 0)
            outflows = data['outflows'].get('count', 0)
            verified = data['total_verified']
            accuracy = data['overall_accuracy']
            acc_str = f"{accuracy:.1f}%" if accuracy else "-"

            print(f"{ex:<15} {inflows:>8} {outflows:>8} {verified:>10} {acc_str:>10}")

        print()


class CorrelationTracker:
    """
    Track flow → price correlation in real-time.

    Integrates with:
    - ClusterRunner for address discovery
    - FlowDetectorWithCache for flow detection
    - MultiExchangePriceFeed for prices
    - CorrelationDatabase for storage
    """

    def __init__(self, db_path: str = "/root/sovereign/correlation.db"):
        self.db = CorrelationDatabase(db_path)
        self.db.price_feed.start()

    def on_flow_detected(self, exchange: str, direction: str,
                         amount_btc: float, txid: str):
        """Called when a flow is detected."""
        # Get price from this specific exchange
        price = self.db.price_feed.get_price(exchange)

        # Record the flow
        flow_id = self.db.record_flow(
            exchange=exchange,
            direction=direction,
            amount_btc=amount_btc,
            txid=txid,
            price_now=price
        )

        print(f"[CORR] {exchange} {direction} {amount_btc:.4f} BTC @ ${price:,.0f} (#{flow_id})")

    def run_verification_loop(self):
        """Run in background to verify pending prices."""
        while True:
            try:
                self.db.check_pending_verifications()
            except Exception as e:
                print(f"[CORR] Verification error: {e}")
            time.sleep(5)

    def stop(self):
        """Stop the tracker."""
        self.db.price_feed.stop()


if __name__ == '__main__':
    print("=" * 70)
    print("CORRELATION DATABASE")
    print("=" * 70)

    db = CorrelationDatabase()

    # Show existing data
    db.print_report()

    # Show causation patterns for each exchange
    correlations = db.get_all_correlations()
    for exchange in correlations:
        patterns = db.get_causation_patterns(exchange)
        if patterns['inflow_patterns'] or patterns['outflow_patterns']:
            print(f"\n{exchange.upper()} CAUSATION PATTERNS:")
            for p in patterns['inflow_patterns']:
                print(f"  INFLOW {p['size_bucket']}: {p['count']} samples, "
                      f"avg 5m change: {p['avg_change_5m']:.3f}%, "
                      f"accuracy: {p['accuracy']:.1f}%" if p['accuracy'] else "")
            for p in patterns['outflow_patterns']:
                print(f"  OUTFLOW {p['size_bucket']}: {p['count']} samples, "
                      f"avg 5m change: {p['avg_change_5m']:.3f}%, "
                      f"accuracy: {p['accuracy']:.1f}%" if p['accuracy'] else "")
