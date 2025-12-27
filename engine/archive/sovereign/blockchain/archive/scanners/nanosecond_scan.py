#!/usr/bin/env python3
"""
NANOSECOND SCAN - COMPLETE BLOCKCHAIN EXTRACTION
=================================================
Extract ALL address data from Bitcoin Core node.
No external APIs. Pure blockchain data.

This runs continuously until ALL blocks are processed.
"""

import sys
import time
import json
import sqlite3
import subprocess
import socket
from datetime import datetime
from typing import Dict, Set, Optional, List

sys.path.insert(0, '/root/sovereign')


class NanosecondScan:
    """
    Ultra-efficient blockchain scanner using direct RPC.
    Extracts ALL transactions to build complete address clusters.
    """

    def __init__(self, db_path: str = "/root/sovereign/address_clusters.db"):
        print("=" * 70)
        print("NANOSECOND SCAN - COMPLETE BLOCKCHAIN EXTRACTION")
        print("=" * 70)
        print(f"Started: {datetime.now()}")
        print()

        self.db_path = db_path
        self.rpc_socket = None

        # Initialize database
        self._init_db()

        # Load existing addresses
        self.addresses: Set[str] = set()
        self.addr_to_exchange: Dict[str, str] = {}
        self._load_existing()

        # Stats
        self.start_time = 0
        self.blocks_done = 0
        self.txs_done = 0
        self.discovered = 0
        self.checkpoint = self._get_checkpoint()

        # Get blockchain height
        self.height = int(self._rpc('getblockcount'))
        print(f"Blockchain height: {self.height:,}")
        print(f"Starting addresses: {len(self.addresses):,}")
        print(f"Resume from block: {self.checkpoint:,}")
        print()

    def _init_db(self):
        """Initialize database tables."""
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
            CREATE TABLE IF NOT EXISTS scan_checkpoint (
                id INTEGER PRIMARY KEY,
                last_block INTEGER,
                blocks_scanned INTEGER,
                addresses_found INTEGER,
                updated_at TEXT
            )
        """)

        # Transaction history for debugging
        c.execute("""
            CREATE TABLE IF NOT EXISTS discovered_txs (
                txid TEXT PRIMARY KEY,
                block_height INTEGER,
                exchange TEXT,
                addresses_found INTEGER,
                discovered_at TEXT
            )
        """)

        conn.commit()
        conn.close()

    def _load_existing(self):
        """Load all existing addresses from database."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT address, exchange FROM addresses")
        for row in c.fetchall():
            self.addresses.add(row[0])
            self.addr_to_exchange[row[0]] = row[1]
        conn.close()

    def _get_checkpoint(self) -> int:
        """Get last scanned block."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT last_block FROM scan_checkpoint ORDER BY id DESC LIMIT 1")
        row = c.fetchone()
        conn.close()
        return row[0] if row else 0

    def _save_checkpoint(self, block: int):
        """Save scan progress."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            INSERT INTO scan_checkpoint (last_block, blocks_scanned, addresses_found, updated_at)
            VALUES (?, ?, ?, ?)
        """, (block, self.blocks_done, self.discovered, datetime.now().isoformat()))
        conn.commit()
        conn.close()

    def _rpc(self, method: str, *params) -> str:
        """Execute Bitcoin RPC call via bitcoin-cli."""
        cmd = ['bitcoin-cli', method] + [str(p) for p in params]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                return None
            return result.stdout.strip()
        except Exception:
            return None

    def _rpc_json(self, method: str, *params) -> Optional[Dict]:
        """Execute RPC returning JSON."""
        result = self._rpc(method, *params)
        if result:
            try:
                return json.loads(result)
            except:
                pass
        return None

    def _process_block(self, height: int) -> int:
        """Process single block. Returns new addresses discovered."""
        # Get block hash
        block_hash = self._rpc('getblockhash', height)
        if not block_hash:
            return 0

        # Get full block with transactions (verbosity=2)
        block = self._rpc_json('getblock', block_hash, 2)
        if not block:
            return 0

        found = 0
        batch_inserts = []

        for tx in block.get('tx', []):
            self.txs_done += 1
            txid = tx.get('txid', '')

            # Extract INPUT addresses from prevout
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

            # Check if ANY input belongs to known exchange
            known_ex = None
            for addr in input_addrs:
                if addr in self.addr_to_exchange:
                    known_ex = self.addr_to_exchange[addr]
                    break

            if not known_ex:
                continue

            # CLUSTER: All inputs belong to same entity
            tx_found = 0
            for addr in input_addrs:
                if addr not in self.addresses:
                    self.addresses.add(addr)
                    self.addr_to_exchange[addr] = known_ex
                    batch_inserts.append((addr, known_ex, datetime.now().isoformat(), 'nanoscan'))
                    found += 1
                    tx_found += 1

            # Log discovery
            if tx_found > 0:
                conn = sqlite3.connect(self.db_path)
                c = conn.cursor()
                c.execute("""
                    INSERT OR IGNORE INTO discovered_txs VALUES (?, ?, ?, ?, ?)
                """, (txid, height, known_ex, tx_found, datetime.now().isoformat()))
                conn.commit()
                conn.close()

        # Batch insert new addresses
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

    def scan_range(self, start: int, end: int):
        """Scan block range."""
        total = end - start + 1
        self.start_time = time.time()

        print(f"Scanning blocks {start:,} to {end:,} ({total:,} blocks)")
        print()

        for height in range(start, end + 1):
            try:
                self._process_block(height)
            except Exception as e:
                print(f"\n[ERR] Block {height}: {e}")
                continue

            # Progress every block
            elapsed = time.time() - self.start_time
            rate = self.blocks_done / max(elapsed, 1)
            remaining = total - self.blocks_done
            eta = remaining / max(rate, 0.01)

            print(f"\r[{self.blocks_done:,}/{total:,}] {rate:.2f} blk/s | "
                  f"TXs: {self.txs_done:,} | "
                  f"Addrs: {len(self.addresses):,} (+{self.discovered:,}) | "
                  f"ETA: {eta/3600:.1f}h", end='', flush=True)

            # Checkpoint every 100 blocks
            if self.blocks_done % 100 == 0:
                self._save_checkpoint(height)

        self._save_checkpoint(end)
        self._print_final()

    def scan_all(self):
        """Scan entire blockchain from checkpoint."""
        start = max(self.checkpoint, 0)
        self.scan_range(start, self.height)

    def scan_backwards(self, num_blocks: int = 100000):
        """Scan backwards from current height. More likely to find active addresses."""
        end = self.height
        start = max(0, end - num_blocks)
        self.scan_range(start, end)

    def _print_final(self):
        """Print final statistics."""
        elapsed = time.time() - self.start_time
        hours = elapsed / 3600

        # Count per exchange
        exchange_counts = {}
        for addr, ex in self.addr_to_exchange.items():
            exchange_counts[ex] = exchange_counts.get(ex, 0) + 1

        print()
        print()
        print("=" * 70)
        print("SCAN COMPLETE")
        print("=" * 70)
        print(f"Time: {hours:.2f} hours")
        print(f"Blocks: {self.blocks_done:,}")
        print(f"Transactions: {self.txs_done:,}")
        print(f"New addresses: {self.discovered:,}")
        print(f"Total addresses: {len(self.addresses):,}")
        print()
        print("PER EXCHANGE:")
        for ex, count in sorted(exchange_counts.items(), key=lambda x: -x[1])[:30]:
            print(f"  {ex:<30} {count:>12,}")
        print("=" * 70)


class FastRPCScanner:
    """
    Alternative scanner using HTTP RPC directly (faster than bitcoin-cli).
    """

    def __init__(self,
                 rpc_user: str = "bitcoin",
                 rpc_pass: str = "bitcoin",
                 rpc_host: str = "127.0.0.1",
                 rpc_port: int = 8332,
                 db_path: str = "/root/sovereign/address_clusters.db"):

        import urllib.request
        import base64

        self.rpc_url = f"http://{rpc_host}:{rpc_port}"
        self.auth = base64.b64encode(f"{rpc_user}:{rpc_pass}".encode()).decode()
        self.db_path = db_path

        print("=" * 70)
        print("FAST RPC SCANNER - HTTP DIRECT")
        print("=" * 70)

        # Load addresses
        self.addresses: Set[str] = set()
        self.addr_to_exchange: Dict[str, str] = {}
        self._load_existing()

        print(f"Starting addresses: {len(self.addresses):,}")

    def _load_existing(self):
        """Load existing addresses."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT address, exchange FROM addresses")
        for row in c.fetchall():
            self.addresses.add(row[0])
            self.addr_to_exchange[row[0]] = row[1]
        conn.close()

    def _rpc(self, method: str, params: List = None) -> Optional[Dict]:
        """Execute RPC via HTTP."""
        import urllib.request

        payload = json.dumps({
            "jsonrpc": "1.0",
            "id": "scanner",
            "method": method,
            "params": params or []
        }).encode()

        req = urllib.request.Request(self.rpc_url)
        req.add_header("Authorization", f"Basic {self.auth}")
        req.add_header("Content-Type", "application/json")

        try:
            with urllib.request.urlopen(req, payload, timeout=300) as resp:
                result = json.loads(resp.read().decode())
                return result.get('result')
        except Exception as e:
            return None

    def scan_block(self, height: int) -> int:
        """Scan single block via HTTP RPC."""
        block_hash = self._rpc('getblockhash', [height])
        if not block_hash:
            return 0

        block = self._rpc('getblock', [block_hash, 2])
        if not block:
            return 0

        found = 0
        batch = []

        for tx in block.get('tx', []):
            input_addrs = []
            for vin in tx.get('vin', []):
                if 'coinbase' in vin:
                    continue
                prevout = vin.get('prevout', {})
                if prevout:
                    addr = prevout.get('scriptPubKey', {}).get('address')
                    if addr:
                        input_addrs.append(addr)

            if not input_addrs:
                continue

            # Find known exchange
            known_ex = None
            for addr in input_addrs:
                if addr in self.addr_to_exchange:
                    known_ex = self.addr_to_exchange[addr]
                    break

            if not known_ex:
                continue

            # Cluster inputs
            for addr in input_addrs:
                if addr not in self.addresses:
                    self.addresses.add(addr)
                    self.addr_to_exchange[addr] = known_ex
                    batch.append((addr, known_ex, datetime.now().isoformat(), 'fastscan'))
                    found += 1

        # Batch insert
        if batch:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.executemany("INSERT OR IGNORE INTO addresses VALUES (?, ?, ?, ?)", batch)
            conn.commit()
            conn.close()

        return found


def main():
    """Run the scanner."""
    import argparse

    parser = argparse.ArgumentParser(description='Nanosecond blockchain scanner')
    parser.add_argument('--start', type=int, help='Start block')
    parser.add_argument('--end', type=int, help='End block')
    parser.add_argument('--recent', type=int, help='Scan last N blocks')
    parser.add_argument('--all', action='store_true', help='Scan entire blockchain')
    parser.add_argument('--fast', action='store_true', help='Use HTTP RPC (faster)')
    args = parser.parse_args()

    scanner = NanosecondScan()

    if args.all:
        print("FULL BLOCKCHAIN SCAN - This will take days!")
        print()
        scanner.scan_all()
    elif args.recent:
        end = scanner.height
        start = max(0, end - args.recent)
        scanner.scan_range(start, end)
    elif args.start is not None:
        end = args.end if args.end else scanner.height
        scanner.scan_range(args.start, end)
    else:
        # Default: scan last 100k blocks
        print("Default: Scanning last 100,000 blocks")
        scanner.scan_backwards(100000)


if __name__ == '__main__':
    main()
