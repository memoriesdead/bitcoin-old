"""
Signal Logger
=============

Captures every blockchain signal for later analysis.
Does NOT trade - only logs.

RenTech principle: Collect data before deploying.
"""

import sqlite3
import time
import json
import threading
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass, asdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SignalRecord:
    """A single signal record."""
    timestamp: float
    timestamp_ms: int
    direction: int              # 1=LONG, -1=SHORT, 0=NEUTRAL
    confidence: float
    should_trade: bool

    # Flow data
    inflow_btc: float
    outflow_btc: float
    net_flow: float
    flow_ratio: float           # outflow / (inflow + outflow)

    # Source info
    source_engine: str          # adaptive, pattern, rentech, ensemble
    exchange_breakdown: str     # JSON of per-exchange flows

    # Blockchain state
    block_height: int
    mempool_size: int
    avg_fee_rate: float

    # Context
    price_at_signal: float      # Price when signal generated
    volatility_1h: float        # 1-hour volatility
    volume_1h: float            # 1-hour volume


class SignalLogger:
    """
    Logs all signals to SQLite for analysis.

    Usage:
        logger = SignalLogger("/path/to/signals.db")
        logger.log_signal(signal_data)
    """

    def __init__(self, db_path: str = "data/signals.db"):
        """
        Initialize signal logger.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_db()
        self._lock = threading.Lock()

        # Stats
        self.signals_logged = 0
        self.start_time = time.time()

        logger.info(f"SignalLogger initialized: {self.db_path}")

    def _init_db(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                timestamp_ms INTEGER NOT NULL,
                direction INTEGER NOT NULL,
                confidence REAL NOT NULL,
                should_trade INTEGER NOT NULL,

                inflow_btc REAL,
                outflow_btc REAL,
                net_flow REAL,
                flow_ratio REAL,

                source_engine TEXT,
                exchange_breakdown TEXT,

                block_height INTEGER,
                mempool_size INTEGER,
                avg_fee_rate REAL,

                price_at_signal REAL,
                volatility_1h REAL,
                volume_1h REAL,

                created_at REAL DEFAULT (strftime('%s', 'now'))
            )
        """)

        # Indexes for fast queries
        c.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON signals(timestamp)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_direction ON signals(direction)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_confidence ON signals(confidence)")

        conn.commit()
        conn.close()

    def log_signal(self, signal: Dict[str, Any], price: float = 0.0) -> int:
        """
        Log a signal to the database.

        Args:
            signal: Signal dictionary from engine
            price: Current BTC price

        Returns:
            Signal ID
        """
        with self._lock:
            now = time.time()
            now_ms = int(now * 1000)

            # Extract fields with defaults
            record = SignalRecord(
                timestamp=now,
                timestamp_ms=now_ms,
                direction=signal.get('direction', 0),
                confidence=signal.get('confidence', 0.0),
                should_trade=signal.get('should_trade', False),

                inflow_btc=signal.get('inflow_btc', 0.0),
                outflow_btc=signal.get('outflow_btc', 0.0),
                net_flow=signal.get('net_flow', 0.0),
                flow_ratio=signal.get('flow_ratio', 0.5),

                source_engine=signal.get('source', 'unknown'),
                exchange_breakdown=json.dumps(signal.get('exchange_breakdown', {})),

                block_height=signal.get('block_height', 0),
                mempool_size=signal.get('mempool_size', 0),
                avg_fee_rate=signal.get('avg_fee_rate', 0.0),

                price_at_signal=price,
                volatility_1h=signal.get('volatility_1h', 0.0),
                volume_1h=signal.get('volume_1h', 0.0),
            )

            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()

            c.execute("""
                INSERT INTO signals (
                    timestamp, timestamp_ms, direction, confidence, should_trade,
                    inflow_btc, outflow_btc, net_flow, flow_ratio,
                    source_engine, exchange_breakdown,
                    block_height, mempool_size, avg_fee_rate,
                    price_at_signal, volatility_1h, volume_1h
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.timestamp, record.timestamp_ms, record.direction,
                record.confidence, int(record.should_trade),
                record.inflow_btc, record.outflow_btc, record.net_flow, record.flow_ratio,
                record.source_engine, record.exchange_breakdown,
                record.block_height, record.mempool_size, record.avg_fee_rate,
                record.price_at_signal, record.volatility_1h, record.volume_1h
            ))

            signal_id = c.lastrowid
            conn.commit()
            conn.close()

            self.signals_logged += 1

            if self.signals_logged % 100 == 0:
                logger.info(f"Signals logged: {self.signals_logged}")

            return signal_id

    def log_from_blockchain_feed(self, blockchain_signal: Dict, price: float = 0.0) -> int:
        """
        Log signal from per_exchange_feed.get_aggregated_signal().

        This is the format your blockchain feed outputs.
        """
        # Convert blockchain feed format to our format
        signal = {
            'direction': blockchain_signal.get('direction', 0),
            'confidence': blockchain_signal.get('confidence', 0.5),
            'should_trade': blockchain_signal.get('should_trade', False),
            'inflow_btc': blockchain_signal.get('total_inflow', 0.0),
            'outflow_btc': blockchain_signal.get('total_outflow', 0.0),
            'net_flow': blockchain_signal.get('net_flow', 0.0),
            'source': 'blockchain',
            'exchange_breakdown': blockchain_signal.get('per_exchange', {}),
            'block_height': blockchain_signal.get('block_height', 0),
        }

        # Calculate flow ratio
        total = signal['inflow_btc'] + signal['outflow_btc']
        if total > 0:
            signal['flow_ratio'] = signal['outflow_btc'] / total
        else:
            signal['flow_ratio'] = 0.5

        return self.log_signal(signal, price)

    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # Total signals
        total = c.execute("SELECT COUNT(*) FROM signals").fetchone()[0]

        # By direction
        longs = c.execute("SELECT COUNT(*) FROM signals WHERE direction = 1").fetchone()[0]
        shorts = c.execute("SELECT COUNT(*) FROM signals WHERE direction = -1").fetchone()[0]
        neutrals = c.execute("SELECT COUNT(*) FROM signals WHERE direction = 0").fetchone()[0]

        # Tradeable signals
        tradeable = c.execute("SELECT COUNT(*) FROM signals WHERE should_trade = 1").fetchone()[0]

        # Time range
        first = c.execute("SELECT MIN(timestamp) FROM signals").fetchone()[0]
        last = c.execute("SELECT MAX(timestamp) FROM signals").fetchone()[0]

        # Average confidence
        avg_conf = c.execute("SELECT AVG(confidence) FROM signals WHERE direction != 0").fetchone()[0]

        conn.close()

        runtime = time.time() - self.start_time
        rate = total / runtime if runtime > 0 else 0

        return {
            'total_signals': total,
            'long_signals': longs,
            'short_signals': shorts,
            'neutral_signals': neutrals,
            'tradeable_signals': tradeable,
            'tradeable_pct': (tradeable / total * 100) if total > 0 else 0,
            'avg_confidence': avg_conf or 0,
            'first_signal': first,
            'last_signal': last,
            'runtime_hours': runtime / 3600,
            'signals_per_hour': rate * 3600,
        }

    def export_csv(self, output_path: str = "signals_export.csv"):
        """Export signals to CSV for external analysis."""
        import csv

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        rows = c.execute("SELECT * FROM signals ORDER BY timestamp").fetchall()
        columns = [desc[0] for desc in c.description]

        conn.close()

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            writer.writerows(rows)

        logger.info(f"Exported {len(rows)} signals to {output_path}")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SIGNAL LOGGER TEST")
    print("=" * 60)

    # Create logger
    logger_instance = SignalLogger("test_signals.db")

    # Log some test signals
    test_signals = [
        {'direction': 1, 'confidence': 0.65, 'should_trade': True,
         'inflow_btc': 100, 'outflow_btc': 250, 'net_flow': 150},
        {'direction': -1, 'confidence': 0.55, 'should_trade': True,
         'inflow_btc': 300, 'outflow_btc': 100, 'net_flow': -200},
        {'direction': 0, 'confidence': 0.45, 'should_trade': False,
         'inflow_btc': 150, 'outflow_btc': 160, 'net_flow': 10},
    ]

    for signal in test_signals:
        signal_id = logger_instance.log_signal(signal, price=100000.0)
        print(f"Logged signal {signal_id}: dir={signal['direction']}, conf={signal['confidence']}")

    # Show stats
    print("\nStats:")
    stats = logger_instance.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Cleanup
    Path("test_signals.db").unlink(missing_ok=True)
