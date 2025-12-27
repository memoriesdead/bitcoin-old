#!/usr/bin/env python3
"""
FULL BLOCKCHAIN SCAN - ALL DATA
================================
Process EVERY block from genesis to now.
No shortcuts. No sampling. ALL transactions.

Run:
    python3 blockchain/full_blockchain_scan.py
"""

import sys
import time
import json
import sqlite3
import subprocess
from datetime import datetime
from typing import Dict, Set, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, '/root/sovereign')
sys.path.insert(0, '/root/sovereign/blockchain')


class FullBlockchainScan:
    """
    Scan ENTIRE blockchain for address clustering.
    No limits. Complete coverage.
    """

    def __init__(self, db_path: str = "/root/sovereign/address_clusters.db"):
        print("=" * 70)
        print("FULL BLOCKCHAIN SCAN - 100% DATA COVERAGE")
        print("=" * 70)
        print(f"Started: {datetime.now()}")
        print()

        self.db_path = db_path
        self._init_db()
        self._load_seeds()
        self._load_existing()

        # Get blockchain info
        self.height = self._rpc_int('getblockcount')
        print(f"Blockchain height: {self.height:,} blocks")
        print(f"Starting addresses: {len(self.addresses):,}")
        print()

        # Progress tracking
        self.blocks_done = 0
        self.txs_done = 0
        self.discovered = 0
        self.start_time = 0

        # Resume from checkpoint
        self.checkpoint = self._get_checkpoint()
        if self.checkpoint > 0:
            print(f"Resuming from block {self.checkpoint:,}")

    def _init_db(self):
        """Initialize database."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS addresses (
                address TEXT PRIMARY KEY,
                exchange TEXT NOT NULL,
                discovered_at TEXT,
                source TEXT DEFAULT 'scan'
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_ex ON addresses(exchange)")
        c.execute("""
            CREATE TABLE IF NOT EXISTS scan_progress (
                id INTEGER PRIMARY KEY,
                last_block INTEGER,
                updated_at TEXT
            )
        """)
        conn.commit()
        conn.close()

    def _load_seeds(self):
        """Load seed addresses."""
        self.addresses: Set[str] = set()
        self.addr_to_exchange: Dict[str, str] = {}

        # Import seeds
        try:
            from address_collector import KNOWN_COLD_WALLETS
        except:
            from blockchain.address_collector import KNOWN_COLD_WALLETS

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        for exchange, addrs in KNOWN_COLD_WALLETS.items():
            for addr in addrs:
                self.addresses.add(addr)
                self.addr_to_exchange[addr] = exchange
                c.execute("""
                    INSERT OR IGNORE INTO addresses (address, exchange, discovered_at, source)
                    VALUES (?, ?, ?, 'seed')
                """, (addr, exchange, datetime.now().isoformat()))

        conn.commit()
        conn.close()
        print(f"Loaded {len(self.addresses)} seed addresses")

    def _load_existing(self):
        """Load previously discovered addresses."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT address, exchange FROM addresses")
        for row in c.fetchall():
            self.addresses.add(row[0])
            self.addr_to_exchange[row[0]] = row[1]
        conn.close()
        print(f"Total addresses in DB: {len(self.addresses):,}")

    def _get_checkpoint(self) -> int:
        """Get last processed block."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT last_block FROM scan_progress ORDER BY id DESC LIMIT 1")
        row = c.fetchone()
        conn.close()
        return row[0] if row else 0

    def _save_checkpoint(self, block: int):
        """Save progress checkpoint."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            INSERT INTO scan_progress (last_block, updated_at)
            VALUES (?, ?)
        """, (block, datetime.now().isoformat()))
        conn.commit()
        conn.close()

    def _rpc(self, method: str, *params) -> str:
        """Execute RPC call."""
        cmd = ['bitcoin-cli', method] + [str(p) for p in params]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return result.stdout.strip()

    def _rpc_int(self, method: str) -> int:
        """Execute RPC call returning int."""
        return int(self._rpc(method))

    def _rpc_json(self, method: str, *params) -> Dict:
        """Execute RPC call returning JSON."""
        return json.loads(self._rpc(method, *params))

    def process_block(self, height: int) -> int:
        """Process single block. Returns addresses discovered."""
        try:
            block_hash = self._rpc('getblockhash', height)
            block = self._rpc_json('getblock', block_hash, 2)
        except Exception as e:
            return 0

        found = 0
        batch_inserts = []

        for tx in block.get('tx', []):
            # Get input addresses from prevout
            input_addrs = []
            for vin in tx.get('vin', []):
                if 'coinbase' in vin:
                    continue
                prevout = vin.get('prevout', {})
                if prevout:
                    spk = prevout.get('scriptPubKey', {})
                    addr = spk.get('address')
                    if addr:
                        input_addrs.append(addr)

            if not input_addrs:
                continue

            # Check if any input is known exchange
            known_ex = None
            for addr in input_addrs:
                if addr in self.addr_to_exchange:
                    known_ex = self.addr_to_exchange[addr]
                    break

            if not known_ex:
                continue

            # Cluster all inputs together
            for addr in input_addrs:
                if addr not in self.addresses:
                    self.addresses.add(addr)
                    self.addr_to_exchange[addr] = known_ex
                    batch_inserts.append((addr, known_ex, datetime.now().isoformat(), 'scan'))
                    found += 1

            self.txs_done += 1

        # Batch insert
        if batch_inserts:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.executemany("""
                INSERT OR IGNORE INTO addresses (address, exchange, discovered_at, source)
                VALUES (?, ?, ?, ?)
            """, batch_inserts)
            conn.commit()
            conn.close()

        self.blocks_done += 1
        self.discovered += found
        return found

    def scan_range(self, start: int, end: int, workers: int = 8):
        """Scan block range with parallel workers."""
        self.start_time = time.time()
        total = end - start + 1

        print(f"Scanning blocks {start:,} to {end:,} ({total:,} blocks)")
        print(f"Using {workers} parallel workers")
        print()

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            batch_start = start

            while batch_start <= end:
                # Submit batch
                batch_end = min(batch_start + 100, end)
                for h in range(batch_start, batch_end + 1):
                    futures[executor.submit(self.process_block, h)] = h

                # Wait for batch
                for future in as_completed(futures):
                    try:
                        future.result()
                    except:
                        pass

                futures.clear()
                batch_start = batch_end + 1

                # Progress
                self._print_progress(total)

                # Checkpoint every 1000 blocks
                if self.blocks_done % 1000 == 0:
                    self._save_checkpoint(start + self.blocks_done)

        self._save_checkpoint(end)
        self._print_final()

    def scan_all(self, workers: int = 8):
        """Scan entire blockchain from checkpoint."""
        start = max(self.checkpoint, 0)
        self.scan_range(start, self.height, workers)

    def _print_progress(self, total: int):
        """Print progress line."""
        elapsed = time.time() - self.start_time
        rate = self.blocks_done / max(elapsed, 1)
        remaining = total - self.blocks_done
        eta = remaining / max(rate, 0.01)

        print(f"\r[{self.blocks_done:,}/{total:,}] "
              f"{rate:.1f} blk/s | "
              f"TXs: {self.txs_done:,} | "
              f"Addrs: {len(self.addresses):,} (+{self.discovered:,}) | "
              f"ETA: {eta/3600:.1f}h", end='', flush=True)

    def _print_final(self):
        """Print final stats."""
        elapsed = time.time() - self.start_time

        # Count per exchange
        exchange_counts = {}
        for addr, ex in self.addr_to_exchange.items():
            exchange_counts[ex] = exchange_counts.get(ex, 0) + 1

        print()
        print()
        print("=" * 70)
        print("SCAN COMPLETE")
        print("=" * 70)
        print(f"Time: {elapsed/3600:.2f} hours")
        print(f"Blocks: {self.blocks_done:,}")
        print(f"Transactions: {self.txs_done:,}")
        print(f"New addresses: {self.discovered:,}")
        print(f"Total addresses: {len(self.addresses):,}")
        print()
        print("PER EXCHANGE:")
        for ex, count in sorted(exchange_counts.items(), key=lambda x: -x[1]):
            print(f"  {ex:<20} {count:>12,}")
        print("=" * 70)


def main():
    scanner = FullBlockchainScan()

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, help='Start block')
    parser.add_argument('--end', type=int, help='End block')
    parser.add_argument('--workers', type=int, default=8, help='Parallel workers')
    parser.add_argument('--recent', type=int, help='Only last N blocks')
    args = parser.parse_args()

    if args.recent:
        start = scanner.height - args.recent
        scanner.scan_range(start, scanner.height, args.workers)
    elif args.start is not None:
        end = args.end if args.end else scanner.height
        scanner.scan_range(args.start, end, args.workers)
    else:
        scanner.scan_all(args.workers)


if __name__ == '__main__':
    main()
