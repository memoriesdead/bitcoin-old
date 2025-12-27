#!/usr/bin/env python3
"""
DETERMINISTIC CORRELATION ENGINE
================================
The missing piece: Actually measure the mathematical relationship between
blockchain flows and price movements.

We CANNOT trade until we know:
  - "100 BTC into Coinbase = price drops $X in Y minutes"
  - "50 BTC out of Binance = price rises $Z in W minutes"

This engine:
1. Captures every flow event with exact price at T=0
2. Schedules price checks at T+1m, T+5m, T+15m, T+30m, T+60m
3. Calculates EXACT correlation per exchange
4. Builds deterministic formula: flow_btc * coefficient = price_delta

NO TRADING until we have the math. Data collection only.
"""

import sqlite3
import time
import threading
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from collections import defaultdict
import json
import statistics

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class CorrelationConfig:
    """Configuration for correlation tracking."""
    db_path: str = "deterministic_correlation.db"

    # Time windows to check price (seconds)
    check_windows: Tuple[int, ...] = (60, 300, 900, 1800, 3600)  # 1m, 5m, 15m, 30m, 60m

    # Minimum BTC to track (reduce noise)
    min_flow_btc: float = 0.5

    # Exchanges to track
    exchanges: Tuple[str, ...] = (
        'binance', 'coinbase', 'kraken', 'bitstamp', 'gemini',
        'bitfinex', 'okx', 'bybit', 'huobi', 'htx', 'kucoin'
    )


# =============================================================================
# DATABASE SCHEMA
# =============================================================================

SCHEMA = """
-- Flow events: Every blockchain flow we detect
CREATE TABLE IF NOT EXISTS flow_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,              -- Unix timestamp (float for precision)
    exchange TEXT NOT NULL,               -- Which exchange
    direction TEXT NOT NULL,              -- 'inflow' or 'outflow'
    amount_btc REAL NOT NULL,             -- How much BTC
    txid TEXT,                            -- Transaction ID
    block_height INTEGER,                 -- Block height

    -- Price at moment of flow detection
    price_t0 REAL,                        -- Price when flow detected

    -- Index for fast queries
    UNIQUE(txid, exchange, direction)
);

-- Price observations: Price at T+N after each flow
CREATE TABLE IF NOT EXISTS price_observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    flow_id INTEGER NOT NULL,             -- Reference to flow_events
    window_seconds INTEGER NOT NULL,       -- Which time window (60, 300, etc.)
    scheduled_time REAL NOT NULL,         -- When we should check
    observed_time REAL,                   -- When we actually checked
    price REAL,                           -- Price at this time
    price_delta REAL,                     -- price - price_t0
    price_delta_pct REAL,                 -- (price - price_t0) / price_t0 * 100

    FOREIGN KEY (flow_id) REFERENCES flow_events(id),
    UNIQUE(flow_id, window_seconds)
);

-- Correlation results: Calculated statistics per exchange
CREATE TABLE IF NOT EXISTS correlation_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    calculated_at REAL NOT NULL,
    exchange TEXT NOT NULL,
    direction TEXT NOT NULL,              -- 'inflow' or 'outflow'

    -- BTC bucket (group flows by size)
    btc_bucket_min REAL NOT NULL,
    btc_bucket_max REAL NOT NULL,

    -- Time window analyzed
    window_seconds INTEGER NOT NULL,

    -- Statistics
    sample_count INTEGER NOT NULL,
    avg_price_delta REAL,                 -- Average $ move
    avg_price_delta_pct REAL,             -- Average % move
    std_price_delta REAL,                 -- Standard deviation

    -- Correlation coefficient: how well does flow predict price?
    -- +1 = perfect positive correlation (flow up = price up)
    -- -1 = perfect negative correlation (flow up = price down)
    -- 0 = no correlation (random)
    pearson_correlation REAL,

    -- Win rate if we traded this
    -- For inflow: win if price goes DOWN
    -- For outflow: win if price goes UP
    win_rate REAL,

    -- Predictability score (0-100)
    -- Combines correlation strength and consistency
    predictability_score REAL,

    -- The formula coefficient
    -- price_delta = flow_btc * coefficient
    coefficient REAL,

    UNIQUE(exchange, direction, btc_bucket_min, btc_bucket_max, window_seconds)
);

-- Deterministic formulas: The final output we use for trading
CREATE TABLE IF NOT EXISTS deterministic_formulas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    exchange TEXT NOT NULL,
    direction TEXT NOT NULL,

    -- Formula: expected_delta = flow_btc * coefficient
    coefficient REAL NOT NULL,

    -- Best time window for this exchange
    optimal_window_seconds INTEGER NOT NULL,

    -- Confidence metrics
    sample_count INTEGER NOT NULL,
    r_squared REAL NOT NULL,              -- How well formula fits (0-1)
    win_rate REAL NOT NULL,               -- Historical accuracy
    avg_delta_per_btc REAL NOT NULL,      -- Average $ move per BTC

    -- Thresholds for trading
    min_flow_btc REAL NOT NULL,           -- Minimum BTC to trade
    min_expected_delta REAL NOT NULL,     -- Minimum $ move to trade

    -- Timestamps
    created_at REAL NOT NULL,
    last_validated REAL,

    UNIQUE(exchange, direction)
);

-- Create indexes for fast queries
CREATE INDEX IF NOT EXISTS idx_flow_exchange ON flow_events(exchange);
CREATE INDEX IF NOT EXISTS idx_flow_direction ON flow_events(direction);
CREATE INDEX IF NOT EXISTS idx_flow_timestamp ON flow_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_price_obs_flow ON price_observations(flow_id);
CREATE INDEX IF NOT EXISTS idx_price_obs_scheduled ON price_observations(scheduled_time);
"""


# =============================================================================
# CORE DATABASE CLASS
# =============================================================================

class DeterministicCorrelationDB:
    """Database for tracking flow-to-price correlations."""

    def __init__(self, config: CorrelationConfig = None):
        self.config = config or CorrelationConfig()
        self.conn = sqlite3.connect(self.config.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        self._init_schema()

    def _init_schema(self):
        """Initialize database schema."""
        with self.lock:
            self.conn.executescript(SCHEMA)
            self.conn.commit()

    def record_flow(
        self,
        exchange: str,
        direction: str,
        amount_btc: float,
        price_t0: float,
        txid: str = None,
        block_height: int = None
    ) -> Optional[int]:
        """
        Record a flow event and schedule price observations.

        Returns: flow_id if recorded, None if duplicate or below threshold
        """
        if amount_btc < self.config.min_flow_btc:
            return None

        if exchange.lower() not in [e.lower() for e in self.config.exchanges]:
            return None

        timestamp = time.time()

        with self.lock:
            try:
                cursor = self.conn.execute("""
                    INSERT INTO flow_events
                    (timestamp, exchange, direction, amount_btc, txid, block_height, price_t0)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (timestamp, exchange.lower(), direction.lower(), amount_btc,
                      txid, block_height, price_t0))

                flow_id = cursor.lastrowid

                # Schedule price observations
                for window in self.config.check_windows:
                    scheduled_time = timestamp + window
                    self.conn.execute("""
                        INSERT INTO price_observations
                        (flow_id, window_seconds, scheduled_time)
                        VALUES (?, ?, ?)
                    """, (flow_id, window, scheduled_time))

                self.conn.commit()
                return flow_id

            except sqlite3.IntegrityError:
                # Duplicate flow
                return None

    def get_pending_observations(self) -> List[Dict]:
        """Get price observations that need to be checked now."""
        now = time.time()

        with self.lock:
            cursor = self.conn.execute("""
                SELECT po.id, po.flow_id, po.window_seconds, po.scheduled_time,
                       fe.exchange, fe.direction, fe.amount_btc, fe.price_t0
                FROM price_observations po
                JOIN flow_events fe ON po.flow_id = fe.id
                WHERE po.observed_time IS NULL
                  AND po.scheduled_time <= ?
                ORDER BY po.scheduled_time
                LIMIT 100
            """, (now,))

            return [dict(row) for row in cursor.fetchall()]

    def record_observation(
        self,
        observation_id: int,
        price: float,
        price_t0: float
    ):
        """Record a price observation."""
        now = time.time()
        delta = price - price_t0
        delta_pct = (delta / price_t0) * 100 if price_t0 else 0

        with self.lock:
            self.conn.execute("""
                UPDATE price_observations
                SET observed_time = ?, price = ?, price_delta = ?, price_delta_pct = ?
                WHERE id = ?
            """, (now, price, delta, delta_pct, observation_id))
            self.conn.commit()

    def calculate_correlations(self) -> Dict:
        """
        Calculate correlation statistics for all exchanges.

        This is the CORE MATH that determines if trading is viable.
        """
        results = {}

        # BTC buckets for analysis
        btc_buckets = [
            (0.5, 5),
            (5, 20),
            (20, 50),
            (50, 100),
            (100, 500),
            (500, float('inf'))
        ]

        with self.lock:
            for exchange in self.config.exchanges:
                results[exchange] = {'inflow': {}, 'outflow': {}}

                for direction in ['inflow', 'outflow']:
                    for window in self.config.check_windows:
                        for btc_min, btc_max in btc_buckets:
                            stats = self._calculate_bucket_stats(
                                exchange, direction, window, btc_min, btc_max
                            )
                            if stats and stats['sample_count'] >= 10:
                                bucket_key = f"{btc_min}-{btc_max}btc_{window}s"
                                results[exchange][direction][bucket_key] = stats

        return results

    def _calculate_bucket_stats(
        self,
        exchange: str,
        direction: str,
        window_seconds: int,
        btc_min: float,
        btc_max: float
    ) -> Optional[Dict]:
        """Calculate statistics for a specific bucket."""

        cursor = self.conn.execute("""
            SELECT fe.amount_btc, po.price_delta, po.price_delta_pct
            FROM flow_events fe
            JOIN price_observations po ON fe.id = po.flow_id
            WHERE fe.exchange = ?
              AND fe.direction = ?
              AND po.window_seconds = ?
              AND fe.amount_btc >= ?
              AND fe.amount_btc < ?
              AND po.price IS NOT NULL
        """, (exchange, direction, window_seconds, btc_min, btc_max))

        rows = cursor.fetchall()
        if not rows:
            return None

        amounts = [r[0] for r in rows]
        deltas = [r[1] for r in rows]
        delta_pcts = [r[2] for r in rows]

        n = len(rows)
        if n < 2:
            return None

        # Calculate basic statistics
        avg_delta = statistics.mean(deltas)
        avg_delta_pct = statistics.mean(delta_pcts)
        std_delta = statistics.stdev(deltas) if n > 1 else 0

        # Calculate Pearson correlation between flow amount and price delta
        # For inflow: expect negative correlation (more inflow = more price drop)
        # For outflow: expect positive correlation (more outflow = more price rise)
        try:
            correlation = self._pearson(amounts, deltas)
        except:
            correlation = 0

        # Win rate calculation
        # For inflow: "win" if price went DOWN (short signal)
        # For outflow: "win" if price went UP (long signal)
        if direction == 'inflow':
            wins = sum(1 for d in deltas if d < 0)
        else:
            wins = sum(1 for d in deltas if d > 0)

        win_rate = wins / n if n > 0 else 0

        # Calculate coefficient: price_delta = amount_btc * coefficient
        # Using least squares regression
        try:
            coefficient = self._linear_coefficient(amounts, deltas)
        except:
            coefficient = 0

        # Predictability score (0-100)
        # Combines correlation strength and win rate
        predictability = (abs(correlation) * 50 + win_rate * 50)

        stats = {
            'sample_count': n,
            'avg_delta': avg_delta,
            'avg_delta_pct': avg_delta_pct,
            'std_delta': std_delta,
            'correlation': correlation,
            'win_rate': win_rate,
            'coefficient': coefficient,
            'predictability_score': predictability,
            'btc_range': f"{btc_min}-{btc_max}",
            'window_seconds': window_seconds
        }

        # Store in database
        self.conn.execute("""
            INSERT OR REPLACE INTO correlation_results
            (calculated_at, exchange, direction, btc_bucket_min, btc_bucket_max,
             window_seconds, sample_count, avg_price_delta, avg_price_delta_pct,
             std_price_delta, pearson_correlation, win_rate, predictability_score,
             coefficient)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (time.time(), exchange, direction, btc_min, btc_max, window_seconds,
              n, avg_delta, avg_delta_pct, std_delta, correlation, win_rate,
              predictability, coefficient))
        self.conn.commit()

        return stats

    def _pearson(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient."""
        n = len(x)
        if n < 2:
            return 0

        mean_x = sum(x) / n
        mean_y = sum(y) / n

        numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))

        sum_sq_x = sum((xi - mean_x) ** 2 for xi in x)
        sum_sq_y = sum((yi - mean_y) ** 2 for yi in y)

        denominator = (sum_sq_x * sum_sq_y) ** 0.5

        if denominator == 0:
            return 0

        return numerator / denominator

    def _linear_coefficient(self, x: List[float], y: List[float]) -> float:
        """Calculate linear regression coefficient (slope)."""
        n = len(x)
        if n < 2:
            return 0

        mean_x = sum(x) / n
        mean_y = sum(y) / n

        numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        denominator = sum((xi - mean_x) ** 2 for xi in x)

        if denominator == 0:
            return 0

        return numerator / denominator

    def build_deterministic_formula(self, exchange: str, direction: str) -> Optional[Dict]:
        """
        Build the final deterministic formula for trading.

        Returns formula if viable, None if not enough data or not predictable.
        """
        # Get best correlation result for this exchange/direction
        cursor = self.conn.execute("""
            SELECT * FROM correlation_results
            WHERE exchange = ? AND direction = ?
              AND sample_count >= 30
              AND predictability_score >= 60
            ORDER BY predictability_score DESC
            LIMIT 1
        """, (exchange, direction))

        row = cursor.fetchone()
        if not row:
            return None

        formula = {
            'exchange': exchange,
            'direction': direction,
            'coefficient': row['coefficient'],
            'optimal_window': row['window_seconds'],
            'sample_count': row['sample_count'],
            'r_squared': row['pearson_correlation'] ** 2,
            'win_rate': row['win_rate'],
            'avg_delta_per_btc': row['coefficient'],
            'min_flow_btc': row['btc_bucket_min'],
            'predictability': row['predictability_score']
        }

        # Store in deterministic_formulas table
        self.conn.execute("""
            INSERT OR REPLACE INTO deterministic_formulas
            (exchange, direction, coefficient, optimal_window_seconds,
             sample_count, r_squared, win_rate, avg_delta_per_btc,
             min_flow_btc, min_expected_delta, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (exchange, direction, formula['coefficient'], formula['optimal_window'],
              formula['sample_count'], formula['r_squared'], formula['win_rate'],
              formula['avg_delta_per_btc'], formula['min_flow_btc'],
              abs(formula['coefficient'] * formula['min_flow_btc']),
              time.time()))
        self.conn.commit()

        return formula

    def get_formula(self, exchange: str, direction: str) -> Optional[Dict]:
        """Get the deterministic formula for trading."""
        cursor = self.conn.execute("""
            SELECT * FROM deterministic_formulas
            WHERE exchange = ? AND direction = ?
        """, (exchange, direction))

        row = cursor.fetchone()
        if row:
            return dict(row)
        return None

    def predict_price_delta(
        self,
        exchange: str,
        direction: str,
        amount_btc: float
    ) -> Optional[Dict]:
        """
        Predict expected price delta using deterministic formula.

        Returns:
            {
                'expected_delta': float,  # Dollar amount price will move
                'expected_direction': str,  # 'up' or 'down'
                'confidence': float,  # 0-1 confidence score
                'time_window': int,  # Seconds until price impact
            }
        """
        formula = self.get_formula(exchange, direction)
        if not formula:
            return None

        if amount_btc < formula['min_flow_btc']:
            return None

        expected_delta = amount_btc * formula['coefficient']

        return {
            'expected_delta': expected_delta,
            'expected_direction': 'down' if expected_delta < 0 else 'up',
            'confidence': formula['r_squared'],
            'time_window': formula['optimal_window_seconds'],
            'win_rate': formula['win_rate']
        }

    def get_statistics_report(self) -> str:
        """Generate a human-readable statistics report."""
        lines = []
        lines.append("=" * 70)
        lines.append("DETERMINISTIC CORRELATION ANALYSIS")
        lines.append("=" * 70)

        # Count flows
        cursor = self.conn.execute("""
            SELECT exchange, direction, COUNT(*) as cnt, SUM(amount_btc) as total_btc
            FROM flow_events
            GROUP BY exchange, direction
            ORDER BY total_btc DESC
        """)

        lines.append("\nFLOW EVENTS RECORDED:")
        lines.append("-" * 50)
        for row in cursor.fetchall():
            lines.append(f"  {row['exchange']:12} {row['direction']:8} | "
                        f"{row['cnt']:5} events | {row['total_btc']:,.1f} BTC")

        # Count observations
        cursor = self.conn.execute("""
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN observed_time IS NOT NULL THEN 1 ELSE 0 END) as completed
            FROM price_observations
        """)
        row = cursor.fetchone()
        lines.append(f"\nPRICE OBSERVATIONS: {row['completed']}/{row['total']} completed")

        # Show formulas
        cursor = self.conn.execute("SELECT * FROM deterministic_formulas")
        formulas = cursor.fetchall()

        if formulas:
            lines.append("\n" + "=" * 70)
            lines.append("DETERMINISTIC FORMULAS (Ready for Trading)")
            lines.append("=" * 70)

            for f in formulas:
                lines.append(f"\n{f['exchange'].upper()} - {f['direction'].upper()}")
                lines.append("-" * 40)
                lines.append(f"  Formula: price_delta = flow_btc * {f['coefficient']:.4f}")
                lines.append(f"  Optimal window: {f['optimal_window_seconds']}s")
                lines.append(f"  R-squared: {f['r_squared']:.2%}")
                lines.append(f"  Win rate: {f['win_rate']:.1%}")
                lines.append(f"  Min flow: {f['min_flow_btc']:.1f} BTC")
                lines.append(f"  Samples: {f['sample_count']}")
        else:
            lines.append("\nNO FORMULAS YET - Need more data")
            lines.append("Collecting flow events and price observations...")

        return "\n".join(lines)

    def close(self):
        """Close database connection."""
        self.conn.close()


# =============================================================================
# CORRELATION COLLECTOR (Runs in background)
# =============================================================================

class CorrelationCollector:
    """
    Background service that:
    1. Receives flow events from pipeline
    2. Schedules and collects price observations
    3. Calculates correlations periodically
    """

    def __init__(
        self,
        db: DeterministicCorrelationDB,
        price_fetcher: Callable[[str], Optional[float]]
    ):
        self.db = db
        self.get_price = price_fetcher
        self.running = False
        self.observation_thread = None
        self.stats_thread = None

    def start(self):
        """Start background collection threads."""
        self.running = True

        # Thread to collect pending observations
        self.observation_thread = threading.Thread(
            target=self._observation_loop,
            daemon=True
        )
        self.observation_thread.start()

        # Thread to calculate statistics periodically
        self.stats_thread = threading.Thread(
            target=self._stats_loop,
            daemon=True
        )
        self.stats_thread.start()

    def stop(self):
        """Stop background threads."""
        self.running = False

    def on_flow(
        self,
        exchange: str,
        direction: str,
        amount_btc: float,
        txid: str = None,
        block_height: int = None
    ):
        """Called when a flow event is detected."""
        price = self.get_price(exchange)
        if price:
            flow_id = self.db.record_flow(
                exchange=exchange,
                direction=direction,
                amount_btc=amount_btc,
                price_t0=price,
                txid=txid,
                block_height=block_height
            )
            if flow_id:
                print(f"[CORR] Recorded: {exchange} {direction} {amount_btc:.2f} BTC @ ${price:,.0f}")

    def _observation_loop(self):
        """Background loop to collect price observations."""
        while self.running:
            try:
                pending = self.db.get_pending_observations()

                for obs in pending:
                    price = self.get_price(obs['exchange'])
                    if price:
                        self.db.record_observation(
                            observation_id=obs['id'],
                            price=price,
                            price_t0=obs['price_t0']
                        )

                time.sleep(5)  # Check every 5 seconds

            except Exception as e:
                print(f"[CORR] Observation error: {e}")
                time.sleep(10)

    def _stats_loop(self):
        """Background loop to calculate correlations."""
        while self.running:
            try:
                # Calculate stats every 5 minutes
                time.sleep(300)

                print("[CORR] Calculating correlations...")
                results = self.db.calculate_correlations()

                # Build formulas for any exchange with enough data
                for exchange in results:
                    for direction in ['inflow', 'outflow']:
                        formula = self.db.build_deterministic_formula(exchange, direction)
                        if formula:
                            print(f"[CORR] Formula ready: {exchange} {direction}")
                            print(f"       Coefficient: {formula['coefficient']:.4f}")
                            print(f"       Win rate: {formula['win_rate']:.1%}")

            except Exception as e:
                print(f"[CORR] Stats error: {e}")


# =============================================================================
# MAIN - Standalone testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DETERMINISTIC CORRELATION ENGINE")
    print("=" * 70)
    print()
    print("This module collects flow→price correlations to build")
    print("mathematically deterministic trading formulas.")
    print()
    print("NO TRADING happens until we have proven formulas.")
    print()

    # Initialize database
    db = DeterministicCorrelationDB()

    # Show current status
    print(db.get_statistics_report())

    db.close()
