"""
Exchange UTXO Cache - Track UTXOs belonging to exchange addresses.

This enables OUTFLOW detection without txindex=1 by caching UTXOs
that are sent TO exchange addresses. When these UTXOs are spent,
we know BTC is leaving the exchange.

MATHEMATICAL PRINCIPLE:
- When we see output to exchange address E: cache (txid, vout, value, E)
- When we see input spending (txid, vout): lookup cache → OUTFLOW from E

This gives us DETERMINISTIC outflow detection for all transactions
we've observed since cache initialization.
"""

import json
import gzip
import sqlite3
import threading
from pathlib import Path
from typing import Dict, Optional, Tuple, Set
from datetime import datetime


class ExchangeUTXOCache:
    """
    Cache UTXOs belonging to exchange addresses.

    Storage: SQLite for persistence + in-memory dict for speed.

    Schema:
        utxos(txid TEXT, vout INT, value_sat INT, exchange TEXT, address TEXT, created_at TEXT)
        PRIMARY KEY (txid, vout)
    """

    def __init__(self, db_path: str = "/root/sovereign/exchange_utxos.db"):
        self.db_path = Path(db_path)
        self.lock = threading.Lock()

        # In-memory cache for O(1) lookup
        # Key: (txid, vout) -> (value_sat, exchange, address)
        self.cache: Dict[Tuple[str, int], Tuple[int, str, str]] = {}

        # Stats
        self.stats = {
            "total_utxos": 0,
            "spent_utxos": 0,
            "total_value_sat": 0,
        }

        self._init_db()

    def _init_db(self):
        """Initialize SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS utxos (
                txid TEXT,
                vout INTEGER,
                value_sat INTEGER,
                exchange TEXT,
                address TEXT,
                created_at TEXT,
                PRIMARY KEY (txid, vout)
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_exchange ON utxos(exchange)
        """)

        conn.commit()

        # Load existing UTXOs into memory
        cursor.execute("SELECT txid, vout, value_sat, exchange, address FROM utxos")
        for row in cursor.fetchall():
            txid, vout, value_sat, exchange, address = row
            self.cache[(txid, vout)] = (value_sat, exchange, address)
            self.stats["total_utxos"] += 1
            self.stats["total_value_sat"] += value_sat

        conn.close()

        print(f"[UTXO_CACHE] Loaded {self.stats['total_utxos']:,} UTXOs "
              f"({self.stats['total_value_sat'] / 1e8:.4f} BTC)")

    def add_utxo(self, txid: str, vout: int, value_sat: int, exchange: str, address: str):
        """Add a new UTXO to the cache (output to exchange address)."""
        key = (txid, vout)

        with self.lock:
            if key in self.cache:
                return  # Already tracked

            self.cache[key] = (value_sat, exchange, address)
            self.stats["total_utxos"] += 1
            self.stats["total_value_sat"] += value_sat

            # Persist to SQLite
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR IGNORE INTO utxos (txid, vout, value_sat, exchange, address, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (txid, vout, value_sat, exchange, address, datetime.now().isoformat()))
            conn.commit()
            conn.close()

    def spend_utxo(self, txid: str, vout: int) -> Optional[Tuple[int, str, str]]:
        """
        Mark a UTXO as spent and return its details if it was an exchange UTXO.

        Returns: (value_sat, exchange, address) if this was an exchange UTXO, else None
        """
        key = (txid, vout)

        with self.lock:
            if key not in self.cache:
                return None  # Not an exchange UTXO (or already spent)

            value_sat, exchange, address = self.cache.pop(key)
            self.stats["spent_utxos"] += 1
            self.stats["total_value_sat"] -= value_sat

            # Remove from SQLite
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM utxos WHERE txid = ? AND vout = ?", (txid, vout))
            conn.commit()
            conn.close()

            return (value_sat, exchange, address)

    def lookup(self, txid: str, vout: int) -> Optional[Tuple[int, str, str]]:
        """Look up a UTXO without spending it."""
        return self.cache.get((txid, vout))

    def get_exchange_balance(self, exchange: str) -> int:
        """Get total BTC (satoshis) in UTXOs for an exchange."""
        total = 0
        for (value_sat, ex, _) in self.cache.values():
            if ex == exchange:
                total += value_sat
        return total

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            "total_utxos": len(self.cache),
            "spent_utxos": self.stats["spent_utxos"],
            "total_btc": self.stats["total_value_sat"] / 1e8,
        }


class FlowDetectorWithCache:
    """
    Flow detector that uses UTXO cache for complete INFLOW/OUTFLOW detection.

    Process each transaction:
    1. Check inputs → any spending cached UTXO = OUTFLOW from exchange
    2. Check outputs → any to exchange address = INFLOW to exchange + cache UTXO
    """

    def __init__(self, exchange_addresses: Set[str], address_to_exchange: Dict[str, str],
                 cache_path: str = "/root/sovereign/exchange_utxos.db"):
        self.exchange_addresses = exchange_addresses
        self.address_to_exchange = address_to_exchange
        self.utxo_cache = ExchangeUTXOCache(cache_path)

        # Stats
        self.inflow_count = 0
        self.outflow_count = 0
        self.inflow_btc = 0.0
        self.outflow_btc = 0.0

    def process_transaction(self, tx: Dict) -> Dict:
        """
        Process a transaction and detect flows.

        Args:
            tx: Decoded transaction with 'txid', 'inputs', 'outputs'

        Returns:
            {
                'txid': str,
                'inflow': float (BTC entering exchanges),
                'outflow': float (BTC leaving exchanges),
                'net_flow': float (outflow - inflow),
                'direction': int (1=LONG, -1=SHORT, 0=NEUTRAL),
                'exchanges': list of involved exchanges
            }
        """
        txid = tx.get('txid', '')
        inflow = 0.0
        outflow = 0.0
        exchanges = set()

        # Check INPUTS for OUTFLOWS (spending exchange UTXOs)
        for inp in tx.get('inputs', []):
            prev_txid = inp.get('prev_txid')
            prev_vout = inp.get('prev_vout')

            if prev_txid and prev_vout is not None:
                result = self.utxo_cache.spend_utxo(prev_txid, prev_vout)
                if result:
                    value_sat, exchange, address = result
                    btc = value_sat / 1e8
                    outflow += btc
                    exchanges.add(exchange)
                    self.outflow_count += 1
                    self.outflow_btc += btc

        # Check OUTPUTS for INFLOWS (to exchange addresses)
        for i, out in enumerate(tx.get('outputs', [])):
            addr = out.get('address')
            btc = out.get('btc', 0)

            if addr and addr in self.exchange_addresses:
                exchange = self.address_to_exchange.get(addr, 'unknown')
                inflow += btc
                exchanges.add(exchange)
                self.inflow_count += 1
                self.inflow_btc += btc

                # Cache this UTXO for future outflow detection
                value_sat = int(btc * 1e8)
                self.utxo_cache.add_utxo(txid, i, value_sat, exchange, addr)

        # Calculate net flow and direction
        net_flow = outflow - inflow

        if net_flow > 0.1:
            direction = 1  # LONG (outflow > inflow)
        elif net_flow < -0.1:
            direction = -1  # SHORT (inflow > outflow)
        else:
            direction = 0  # NEUTRAL

        return {
            'txid': txid,
            'inflow': inflow,
            'outflow': outflow,
            'net_flow': net_flow,
            'direction': direction,
            'signal': 'LONG' if direction > 0 else 'SHORT' if direction < 0 else 'NEUTRAL',
            'exchanges': list(exchanges),
        }

    def get_stats(self) -> Dict:
        """Get detector statistics."""
        return {
            'inflow_count': self.inflow_count,
            'outflow_count': self.outflow_count,
            'inflow_btc': self.inflow_btc,
            'outflow_btc': self.outflow_btc,
            'net_btc': self.outflow_btc - self.inflow_btc,
            'cache': self.utxo_cache.get_stats(),
        }


if __name__ == '__main__':
    # Test the cache
    cache = ExchangeUTXOCache("/tmp/test_utxo_cache.db")

    # Add a test UTXO
    cache.add_utxo("abc123", 0, 100000000, "binance", "34xp4vRoCGJym3xR7yCVPFHoCNxv4Twseo")
    print(f"After add: {cache.get_stats()}")

    # Spend it
    result = cache.spend_utxo("abc123", 0)
    print(f"Spent UTXO: {result}")
    print(f"After spend: {cache.get_stats()}")
