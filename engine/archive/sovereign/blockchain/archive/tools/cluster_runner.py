#!/usr/bin/env python3
"""
CONTINUOUS ADDRESS CLUSTERING RUNNER
=====================================
Phase 1: Run address clustering continuously to discover addresses.

Uses common-input-ownership heuristic:
- If A and B are inputs in same TX, same entity controls both
- Starting from 72 seed addresses, grow to millions

Persistence: SQLite database (not JSON) for robustness.

Run on VPS:
    cd /root/sovereign && python3 blockchain/cluster_runner.py
"""

import sys
import time
import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Set, List, Optional

sys.path.insert(0, '/root/sovereign')
sys.path.insert(0, '/root/sovereign/blockchain')

try:
    from zmq_subscriber import BlockchainZMQ
    from tx_decoder import TransactionDecoder
    from address_collector import KNOWN_COLD_WALLETS
    from exchange_utxo_cache import ExchangeUTXOCache
except ImportError:
    from blockchain.zmq_subscriber import BlockchainZMQ
    from blockchain.tx_decoder import TransactionDecoder
    from blockchain.address_collector import KNOWN_COLD_WALLETS
    from blockchain.exchange_utxo_cache import ExchangeUTXOCache


class SQLiteAddressCluster:
    """
    Address clustering with SQLite persistence.

    Schema:
        addresses(address TEXT PRIMARY KEY, exchange TEXT, discovered_at TEXT, source TEXT)
        stats(key TEXT PRIMARY KEY, value TEXT)
    """

    def __init__(self, db_path: str = "/root/sovereign/address_clusters.db"):
        self.db_path = Path(db_path)
        self.lock = threading.Lock()

        # In-memory cache for O(1) lookup
        self.address_to_exchange: Dict[str, str] = {}
        self.all_addresses: Set[str] = set()

        # Stats
        self.txs_processed = 0
        self.addresses_discovered = 0
        self.session_discovered = 0

        self._init_db()
        self._load_seeds()
        self._load_from_db()

    def _init_db(self):
        """Initialize SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS addresses (
                address TEXT PRIMARY KEY,
                exchange TEXT NOT NULL,
                discovered_at TEXT,
                source TEXT DEFAULT 'clustering'
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_exchange ON addresses(exchange)
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stats (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)

        conn.commit()
        conn.close()

    def _load_seeds(self):
        """Load seed addresses from KNOWN_COLD_WALLETS."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        seed_count = 0
        for exchange, addresses in KNOWN_COLD_WALLETS.items():
            for addr in addresses:
                if addr not in self.all_addresses:
                    self.address_to_exchange[addr] = exchange
                    self.all_addresses.add(addr)

                    # Insert into DB if not exists
                    cursor.execute("""
                        INSERT OR IGNORE INTO addresses (address, exchange, discovered_at, source)
                        VALUES (?, ?, ?, 'seed')
                    """, (addr, exchange, datetime.now().isoformat()))
                    seed_count += 1

        conn.commit()
        conn.close()
        print(f"[CLUSTER] Loaded {seed_count} seed addresses from {len(KNOWN_COLD_WALLETS)} exchanges")

    def _load_from_db(self):
        """Load previously discovered addresses from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT address, exchange FROM addresses")
        loaded = 0
        for row in cursor.fetchall():
            addr, exchange = row
            if addr not in self.all_addresses:
                self.address_to_exchange[addr] = exchange
                self.all_addresses.add(addr)
                loaded += 1

        conn.close()
        print(f"[CLUSTER] Loaded {loaded} addresses from database (total: {len(self.all_addresses):,})")

    def process_transaction(self, tx: Dict) -> List[str]:
        """
        Process transaction for address clustering.

        Common-input-ownership heuristic:
        If any input address is known exchange, ALL input addresses belong to that exchange.
        """
        self.txs_processed += 1
        discovered = []

        # Extract input addresses
        input_addresses = []
        for inp in tx.get('inputs', []):
            addr = inp.get('address')
            if addr:
                input_addresses.append(addr)

        if not input_addresses:
            return []

        # Check if any input is known exchange address
        known_exchange = None
        for addr in input_addresses:
            if addr in self.address_to_exchange:
                known_exchange = self.address_to_exchange[addr]
                break

        if not known_exchange:
            return []

        # Add all other input addresses to same exchange cluster
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            for addr in input_addresses:
                if addr and addr not in self.all_addresses:
                    self.address_to_exchange[addr] = known_exchange
                    self.all_addresses.add(addr)
                    self.addresses_discovered += 1
                    self.session_discovered += 1
                    discovered.append(addr)

                    # Persist to database
                    cursor.execute("""
                        INSERT OR IGNORE INTO addresses (address, exchange, discovered_at, source)
                        VALUES (?, ?, ?, 'clustering')
                    """, (addr, known_exchange, datetime.now().isoformat()))

            conn.commit()
            conn.close()

        # Log discoveries
        if discovered:
            if self.session_discovered <= 20 or self.session_discovered % 100 == 0:
                print(f"[CLUSTER] {known_exchange} +{len(discovered)} addresses (total: {len(self.all_addresses):,})")

        return discovered

    def get_exchange(self, address: str) -> Optional[str]:
        """Get exchange for an address."""
        return self.address_to_exchange.get(address)

    def is_exchange_address(self, address: str) -> bool:
        """Check if address belongs to an exchange."""
        return address in self.all_addresses

    def get_stats(self) -> Dict:
        """Get clustering statistics."""
        # Count per exchange
        exchange_counts = {}
        for addr, ex in self.address_to_exchange.items():
            exchange_counts[ex] = exchange_counts.get(ex, 0) + 1

        return {
            'total_addresses': len(self.all_addresses),
            'session_discovered': self.session_discovered,
            'txs_processed': self.txs_processed,
            'exchanges': len(exchange_counts),
            'per_exchange': exchange_counts,
        }

    def save_stats(self):
        """Save stats to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        stats = self.get_stats()
        cursor.execute("""
            INSERT OR REPLACE INTO stats (key, value) VALUES (?, ?)
        """, ('last_stats', json.dumps(stats)))
        cursor.execute("""
            INSERT OR REPLACE INTO stats (key, value) VALUES (?, ?)
        """, ('last_update', datetime.now().isoformat()))

        conn.commit()
        conn.close()


class ContinuousClusterRunner:
    """
    Run address clustering continuously on live blockchain transactions.
    """

    def __init__(self, db_path: str = "/root/sovereign/address_clusters.db",
                 utxo_db_path: str = "/root/sovereign/exchange_utxos.db"):
        print("=" * 70)
        print("CONTINUOUS ADDRESS CLUSTERING RUNNER")
        print("=" * 70)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # Initialize clustering
        self.cluster = SQLiteAddressCluster(db_path)

        # Initialize UTXO cache
        self.utxo_cache = ExchangeUTXOCache(utxo_db_path)

        # TX decoder
        self.decoder = TransactionDecoder()

        # Stats
        self.start_time = 0
        self.tx_count = 0
        self.last_stats_time = 0

        # ZMQ subscriber
        self.zmq = BlockchainZMQ(
            rawtx_endpoint="tcp://127.0.0.1:28332",
            on_transaction=self._on_transaction
        )

    def _on_transaction(self, raw_tx: bytes):
        """Process incoming transaction."""
        self.tx_count += 1

        try:
            tx = self.decoder.decode(raw_tx)
            if not tx:
                return
        except Exception:
            return

        # Process through clustering
        discovered = self.cluster.process_transaction(tx)

        # For newly discovered addresses, scan for their UTXOs
        # (This would require scantxoutset which is slow, so we skip for now)
        # Instead, we just start tracking new outputs to these addresses

        # Check outputs for inflows to known addresses
        txid = tx.get('txid', '')
        for i, out in enumerate(tx.get('outputs', [])):
            addr = out.get('address')
            btc = out.get('btc', 0)

            if addr and self.cluster.is_exchange_address(addr):
                exchange = self.cluster.get_exchange(addr)
                value_sat = int(btc * 1e8)
                self.utxo_cache.add_utxo(txid, i, value_sat, exchange, addr)

    def _print_stats(self):
        """Print current statistics."""
        elapsed = time.time() - self.start_time
        hours = elapsed / 3600

        stats = self.cluster.get_stats()

        print()
        print("=" * 70)
        print(f"[{hours:.1f}h] CLUSTERING STATS")
        print("=" * 70)
        print(f"Transactions processed: {self.tx_count:,}")
        print(f"Total addresses: {stats['total_addresses']:,}")
        print(f"Session discovered: {stats['session_discovered']:,}")
        print(f"Discovery rate: {stats['session_discovered'] / max(hours, 0.01):.1f}/hour")
        print()
        print("Per Exchange:")
        for ex, count in sorted(stats['per_exchange'].items(), key=lambda x: -x[1]):
            print(f"  {ex:<15} {count:>10,}")
        print("=" * 70)
        print()

        # Save stats
        self.cluster.save_stats()

    def run(self):
        """Run continuously."""
        print(f"Initial stats: {self.cluster.get_stats()}")
        print()
        print("Connecting to ZMQ...")

        self.start_time = time.time()
        self.last_stats_time = self.start_time
        self.zmq.start()

        try:
            while True:
                time.sleep(60)

                # Print stats every 10 minutes
                if time.time() - self.last_stats_time >= 600:
                    self._print_stats()
                    self.last_stats_time = time.time()

        except KeyboardInterrupt:
            print("\n[INTERRUPTED]")
        finally:
            self.zmq.stop()
            self._print_stats()
            print("Clustering stopped.")


if __name__ == '__main__':
    runner = ContinuousClusterRunner()
    runner.run()
