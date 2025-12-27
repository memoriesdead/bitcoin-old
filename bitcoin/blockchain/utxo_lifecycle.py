#!/usr/bin/env python3
"""
UTXO LIFECYCLE TRACKER - DETERMINISTIC SIGNALS
==============================================

The key insight: 100% deterministic signals come from tracking the FULL lifecycle:

    DEPOSIT TX → CONFIRMATION → EXCHANGE SPENDS UTXO → DESTINATION

When we see an exchange SPEND a deposited UTXO:
  - If destination is INTERNAL (exchange hot wallet) → They're consolidating to SELL → SHORT
  - If destination is EXTERNAL (user withdrawal) → Someone BOUGHT and is withdrawing → LONG

This is DETERMINISTIC because we see the actual action, not a prediction.

PIPELINE:
    1. Deposit to exchange address (UTXO created)
    2. Track confirmation count (1-conf = they CAN sell)
    3. Exchange spends UTXO (we see this in real-time)
    4. Classify destination:
       - Another exchange address = INTERNAL move = preparing to sell
       - Non-exchange address = EXTERNAL withdrawal = someone bought
    5. Generate deterministic signal
"""

import sqlite3
import threading
from pathlib import Path
from typing import Dict, Optional, Tuple, Set, List
from datetime import datetime
from dataclasses import dataclass
from enum import Enum


class SpendType(Enum):
    """Classification of how a UTXO was spent."""
    UNSPENT = "unspent"
    INTERNAL = "internal"      # Spent to another exchange address (hot wallet consolidation)
    EXTERNAL = "external"      # Spent to non-exchange address (withdrawal)
    MIXED = "mixed"            # Spent in TX with both internal and external outputs


@dataclass
class UTXOLifecycle:
    """Complete lifecycle of a single UTXO."""
    txid: str
    vout: int
    value_sat: int
    exchange: str
    address: str

    # Creation
    created_at: str
    created_block: Optional[int] = None

    # Confirmation
    confirmations: int = 0
    first_confirmed_at: Optional[str] = None

    # Spending
    spent: bool = False
    spent_at: Optional[str] = None
    spent_block: Optional[int] = None
    spending_txid: Optional[str] = None
    spend_type: SpendType = SpendType.UNSPENT

    # Destination analysis
    internal_output_btc: float = 0.0    # BTC going to exchange addresses
    external_output_btc: float = 0.0    # BTC going to non-exchange addresses

    def time_to_spend_seconds(self) -> Optional[float]:
        """How long from deposit to spend."""
        if not self.spent_at or not self.created_at:
            return None
        try:
            created = datetime.fromisoformat(self.created_at)
            spent = datetime.fromisoformat(self.spent_at)
            return (spent - created).total_seconds()
        except:
            return None


class UTXOLifecycleTracker:
    """
    Track complete UTXO lifecycle for deterministic signal generation.

    Schema extends basic UTXO tracking with:
    - Block height for confirmation counting
    - Spending transaction details
    - Destination classification (internal vs external)
    """

    def __init__(self, db_path: str = "/root/sovereign/utxo_lifecycle.db",
                 exchange_addresses: Optional[Set[str]] = None,
                 address_to_exchange: Optional[Dict[str, str]] = None):
        self.db_path = Path(db_path)
        self.lock = threading.Lock()

        # Exchange address lookup
        self.exchange_addresses = exchange_addresses or set()
        self.address_to_exchange = address_to_exchange or {}

        # In-memory cache for unspent UTXOs
        # Key: (txid, vout) -> UTXOLifecycle
        self.unspent: Dict[Tuple[str, int], UTXOLifecycle] = {}

        self._init_db()
        self._load_unspent()

    def _init_db(self):
        """Initialize database with extended schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS utxo_lifecycle (
                txid TEXT,
                vout INTEGER,
                value_sat INTEGER,
                exchange TEXT,
                address TEXT,

                -- Creation
                created_at TEXT,
                created_block INTEGER,

                -- Confirmation
                confirmations INTEGER DEFAULT 0,
                first_confirmed_at TEXT,

                -- Spending
                spent INTEGER DEFAULT 0,
                spent_at TEXT,
                spent_block INTEGER,
                spending_txid TEXT,
                spend_type TEXT DEFAULT 'unspent',

                -- Destination analysis
                internal_output_btc REAL DEFAULT 0,
                external_output_btc REAL DEFAULT 0,

                PRIMARY KEY (txid, vout)
            )
        """)

        # Indexes for common queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_exchange ON utxo_lifecycle(exchange)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_spent ON utxo_lifecycle(spent)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_spend_type ON utxo_lifecycle(spend_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_block ON utxo_lifecycle(created_block)")

        # Signals table - deterministic signals generated from lifecycle
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS lifecycle_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                exchange TEXT,
                signal_type TEXT,          -- 'SHORT_INTERNAL' or 'LONG_EXTERNAL'
                trigger_txid TEXT,          -- Spending TX that triggered signal
                utxo_txid TEXT,
                utxo_vout INTEGER,
                btc_amount REAL,
                deposit_to_spend_seconds REAL,

                -- Price tracking (filled in later)
                price_at_signal REAL,
                price_at_t1min REAL,
                price_at_t5min REAL,
                price_at_t10min REAL,
                price_moved_expected INTEGER   -- 1=yes, 0=no
            )
        """)

        conn.commit()
        conn.close()

    def _load_unspent(self):
        """Load unspent UTXOs into memory for fast lookup."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT txid, vout, value_sat, exchange, address,
                   created_at, created_block, confirmations, first_confirmed_at
            FROM utxo_lifecycle
            WHERE spent = 0
        """)

        count = 0
        for row in cursor.fetchall():
            utxo = UTXOLifecycle(
                txid=row[0],
                vout=row[1],
                value_sat=row[2],
                exchange=row[3],
                address=row[4],
                created_at=row[5],
                created_block=row[6],
                confirmations=row[7] or 0,
                first_confirmed_at=row[8]
            )
            self.unspent[(utxo.txid, utxo.vout)] = utxo
            count += 1

        conn.close()
        print(f"[LIFECYCLE] Loaded {count:,} unspent UTXOs")

    def add_deposit(self, txid: str, vout: int, value_sat: int,
                    exchange: str, address: str, block_height: Optional[int] = None):
        """Record a new deposit to exchange address."""
        key = (txid, vout)

        with self.lock:
            if key in self.unspent:
                return  # Already tracked

            utxo = UTXOLifecycle(
                txid=txid,
                vout=vout,
                value_sat=value_sat,
                exchange=exchange,
                address=address,
                created_at=datetime.now().isoformat(),
                created_block=block_height,
                confirmations=0 if block_height is None else 1
            )

            if block_height:
                utxo.first_confirmed_at = datetime.now().isoformat()

            self.unspent[key] = utxo

            # Persist
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR IGNORE INTO utxo_lifecycle
                (txid, vout, value_sat, exchange, address, created_at, created_block,
                 confirmations, first_confirmed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (txid, vout, value_sat, exchange, address, utxo.created_at,
                  block_height, utxo.confirmations, utxo.first_confirmed_at))
            conn.commit()
            conn.close()

    def update_confirmations(self, current_block: int):
        """Update confirmation counts for all unspent UTXOs."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            for key, utxo in self.unspent.items():
                if utxo.created_block:
                    new_conf = current_block - utxo.created_block + 1
                    if new_conf > utxo.confirmations:
                        utxo.confirmations = new_conf
                        if utxo.confirmations == 1 and not utxo.first_confirmed_at:
                            utxo.first_confirmed_at = datetime.now().isoformat()

                        cursor.execute("""
                            UPDATE utxo_lifecycle
                            SET confirmations = ?, first_confirmed_at = ?
                            WHERE txid = ? AND vout = ?
                        """, (new_conf, utxo.first_confirmed_at, utxo.txid, utxo.vout))

            conn.commit()
            conn.close()

    def process_spending_tx(self, tx: Dict, block_height: Optional[int] = None) -> List[Dict]:
        """
        Process a transaction that might spend exchange UTXOs.

        Returns list of deterministic signals generated.
        """
        signals = []
        spending_txid = tx.get('txid', '')

        # Analyze outputs first - what type of destinations?
        output_analysis = self._analyze_outputs(tx.get('outputs', []))

        # Check each input
        for inp in tx.get('inputs', []):
            prev_txid = inp.get('prev_txid')
            prev_vout = inp.get('prev_vout')

            if prev_txid is None or prev_vout is None:
                continue

            key = (prev_txid, prev_vout)

            with self.lock:
                if key not in self.unspent:
                    continue

                utxo = self.unspent.pop(key)

                # Classify the spend
                spend_type = self._classify_spend(output_analysis)

                utxo.spent = True
                utxo.spent_at = datetime.now().isoformat()
                utxo.spent_block = block_height
                utxo.spending_txid = spending_txid
                utxo.spend_type = spend_type
                utxo.internal_output_btc = output_analysis['internal_btc']
                utxo.external_output_btc = output_analysis['external_btc']

                # Update database
                self._save_spent_utxo(utxo)

                # Generate deterministic signal
                signal = self._generate_signal(utxo, spending_txid)
                if signal:
                    signals.append(signal)
                    self._save_signal(signal)

        return signals

    def _analyze_outputs(self, outputs: List[Dict]) -> Dict:
        """Analyze TX outputs - what portion goes to exchanges vs external?"""
        internal_btc = 0.0
        external_btc = 0.0
        internal_exchanges = set()

        for out in outputs:
            addr = out.get('address')
            btc = out.get('btc', 0)

            if addr and addr in self.exchange_addresses:
                internal_btc += btc
                exchange = self.address_to_exchange.get(addr, 'unknown')
                internal_exchanges.add(exchange)
            elif addr:
                external_btc += btc

        return {
            'internal_btc': internal_btc,
            'external_btc': external_btc,
            'internal_exchanges': internal_exchanges,
            'total_btc': internal_btc + external_btc
        }

    def _classify_spend(self, output_analysis: Dict) -> SpendType:
        """Classify spend type based on output analysis."""
        internal = output_analysis['internal_btc']
        external = output_analysis['external_btc']

        if internal > 0 and external == 0:
            return SpendType.INTERNAL
        elif external > 0 and internal == 0:
            return SpendType.EXTERNAL
        elif internal > 0 and external > 0:
            return SpendType.MIXED
        else:
            return SpendType.EXTERNAL  # Default to external if no clear outputs

    def _generate_signal(self, utxo: UTXOLifecycle, spending_txid: str) -> Optional[Dict]:
        """
        Generate deterministic signal from spent UTXO.

        INTERNAL spend (to exchange address) → SHORT
            - Exchange consolidating to hot wallet
            - Preparing to execute sell order
            - Supply about to hit order book

        EXTERNAL spend (to non-exchange address) → LONG
            - Customer withdrawal
            - Someone already bought and is taking custody
            - Buying pressure was absorbed
        """
        btc_amount = utxo.value_sat / 1e8
        time_to_spend = utxo.time_to_spend_seconds()

        if utxo.spend_type == SpendType.INTERNAL:
            signal_type = "SHORT_INTERNAL"
        elif utxo.spend_type == SpendType.EXTERNAL:
            signal_type = "LONG_EXTERNAL"
        elif utxo.spend_type == SpendType.MIXED:
            # Mixed - lean towards the larger portion
            if utxo.internal_output_btc > utxo.external_output_btc:
                signal_type = "SHORT_INTERNAL"
            else:
                signal_type = "LONG_EXTERNAL"
        else:
            return None

        return {
            'timestamp': datetime.now().isoformat(),
            'exchange': utxo.exchange,
            'signal_type': signal_type,
            'trigger_txid': spending_txid,
            'utxo_txid': utxo.txid,
            'utxo_vout': utxo.vout,
            'btc_amount': btc_amount,
            'deposit_to_spend_seconds': time_to_spend,
            'internal_btc': utxo.internal_output_btc,
            'external_btc': utxo.external_output_btc
        }

    def _save_spent_utxo(self, utxo: UTXOLifecycle):
        """Update database with spent UTXO details."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE utxo_lifecycle SET
                spent = 1,
                spent_at = ?,
                spent_block = ?,
                spending_txid = ?,
                spend_type = ?,
                internal_output_btc = ?,
                external_output_btc = ?
            WHERE txid = ? AND vout = ?
        """, (
            utxo.spent_at, utxo.spent_block, utxo.spending_txid,
            utxo.spend_type.value, utxo.internal_output_btc, utxo.external_output_btc,
            utxo.txid, utxo.vout
        ))
        conn.commit()
        conn.close()

    def _save_signal(self, signal: Dict):
        """Save signal to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO lifecycle_signals
            (timestamp, exchange, signal_type, trigger_txid, utxo_txid, utxo_vout,
             btc_amount, deposit_to_spend_seconds)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            signal['timestamp'], signal['exchange'], signal['signal_type'],
            signal['trigger_txid'], signal['utxo_txid'], signal['utxo_vout'],
            signal['btc_amount'], signal.get('deposit_to_spend_seconds')
        ))
        conn.commit()
        conn.close()

    def get_stats(self) -> Dict:
        """Get lifecycle tracker statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Count by spend type
        cursor.execute("""
            SELECT spend_type, COUNT(*), SUM(value_sat)/1e8
            FROM utxo_lifecycle
            WHERE spent = 1
            GROUP BY spend_type
        """)
        spend_stats = {row[0]: {'count': row[1], 'btc': row[2]} for row in cursor.fetchall()}

        # Signal stats
        cursor.execute("""
            SELECT signal_type, COUNT(*), SUM(btc_amount)
            FROM lifecycle_signals
            GROUP BY signal_type
        """)
        signal_stats = {row[0]: {'count': row[1], 'btc': row[2]} for row in cursor.fetchall()}

        # Average time to spend
        cursor.execute("""
            SELECT exchange, AVG(
                (julianday(spent_at) - julianday(created_at)) * 86400
            ) as avg_seconds
            FROM utxo_lifecycle
            WHERE spent = 1 AND spent_at IS NOT NULL AND created_at IS NOT NULL
            GROUP BY exchange
        """)
        avg_time = {row[0]: row[1] for row in cursor.fetchall()}

        conn.close()

        return {
            'unspent_count': len(self.unspent),
            'spend_types': spend_stats,
            'signals': signal_stats,
            'avg_time_to_spend_seconds': avg_time
        }


def test_lifecycle():
    """Test the lifecycle tracker."""
    tracker = UTXOLifecycleTracker("/tmp/test_lifecycle.db")

    # Simulate deposit
    tracker.add_deposit(
        txid="deposit123",
        vout=0,
        value_sat=100_000_000,  # 1 BTC
        exchange="coinbase",
        address="1CoinbaseAddr",
        block_height=900000
    )
    print(f"After deposit: {len(tracker.unspent)} unspent UTXOs")

    # Simulate spending to external address (withdrawal)
    external_tx = {
        'txid': 'spending456',
        'inputs': [{'prev_txid': 'deposit123', 'prev_vout': 0}],
        'outputs': [
            {'address': '1UserWallet', 'btc': 0.99}  # External
        ]
    }
    signals = tracker.process_spending_tx(external_tx, block_height=900010)
    print(f"Signals generated: {signals}")
    print(f"Stats: {tracker.get_stats()}")


if __name__ == "__main__":
    test_lifecycle()
