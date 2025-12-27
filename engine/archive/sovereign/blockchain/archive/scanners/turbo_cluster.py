#!/usr/bin/env python3
"""
TURBO CLUSTER - PARALLEL BLOCKCHAIN EXTRACTION
===============================================
Multi-threaded blockchain scanner for rapid address clustering.

Uses HTTP RPC directly (faster than bitcoin-cli) with batch requests.
Runs multiple worker threads to maximize throughput.
"""

import sys
import time
import json
import sqlite3
import urllib.request
import base64
import threading
from datetime import datetime
from typing import Dict, Set, Optional, List, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue

sys.path.insert(0, '/root/sovereign')


class TurboCluster:
    """
    High-performance parallel blockchain scanner.
    """

    def __init__(self,
                 rpc_user: str = "bitcoin",
                 rpc_pass: str = "bitcoin",
                 rpc_host: str = "127.0.0.1",
                 rpc_port: int = 8332,
                 db_path: str = "/root/sovereign/address_clusters.db",
                 workers: int = 4):

        print("=" * 70)
        print("TURBO CLUSTER - PARALLEL EXTRACTION")
        print("=" * 70)
        print(f"Started: {datetime.now()}")
        print(f"Workers: {workers}")
        print()

        self.rpc_url = f"http://{rpc_host}:{rpc_port}"
        self.auth = base64.b64encode(f"{rpc_user}:{rpc_pass}".encode()).decode()
        self.db_path = db_path
        self.workers = workers

        # Thread-safe data structures
        self.addresses: Set[str] = set()
        self.addr_to_exchange: Dict[str, str] = {}
        self.lock = threading.Lock()
        self.db_lock = threading.Lock()

        # Load existing
        self._load_existing()

        # Stats
        self.blocks_done = 0
        self.txs_done = 0
        self.discovered = 0
        self.start_time = 0

        # Get height
        self.height = self._rpc('getblockcount')
        print(f"Blockchain height: {self.height:,}")
        print(f"Starting addresses: {len(self.addresses):,}")
        print()

    def _load_existing(self):
        """Load existing addresses."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT address, exchange FROM addresses")
        for row in c.fetchall():
            self.addresses.add(row[0])
            self.addr_to_exchange[row[0]] = row[1]
        conn.close()

    def _rpc(self, method: str, params: List = None) -> Optional[any]:
        """Execute single RPC call via HTTP."""
        payload = json.dumps({
            "jsonrpc": "1.0",
            "id": "turbo",
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

    def _rpc_batch(self, calls: List[Tuple[str, List]]) -> List[any]:
        """Execute batch RPC call."""
        payload = json.dumps([
            {"jsonrpc": "1.0", "id": f"batch_{i}", "method": m, "params": p or []}
            for i, (m, p) in enumerate(calls)
        ]).encode()

        req = urllib.request.Request(self.rpc_url)
        req.add_header("Authorization", f"Basic {self.auth}")
        req.add_header("Content-Type", "application/json")

        try:
            with urllib.request.urlopen(req, payload, timeout=600) as resp:
                results = json.loads(resp.read().decode())
                return [r.get('result') for r in results]
        except Exception as e:
            return [None] * len(calls)

    def _process_block(self, height: int) -> Tuple[int, int]:
        """
        Process single block. Returns (txs_processed, addresses_found).
        """
        # Get block hash
        block_hash = self._rpc('getblockhash', [height])
        if not block_hash:
            return 0, 0

        # Get full block with transactions
        block = self._rpc('getblock', [block_hash, 2])
        if not block:
            return 0, 0

        txs = 0
        found = 0
        batch = []

        for tx in block.get('tx', []):
            txs += 1

            # Extract inputs
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

            # Find known exchange in inputs
            known_ex = None
            with self.lock:
                for addr in input_addrs:
                    if addr in self.addr_to_exchange:
                        known_ex = self.addr_to_exchange[addr]
                        break

            if not known_ex:
                continue

            # Cluster inputs
            with self.lock:
                for addr in input_addrs:
                    if addr not in self.addresses:
                        self.addresses.add(addr)
                        self.addr_to_exchange[addr] = known_ex
                        batch.append((addr, known_ex, datetime.now().isoformat(), 'turbo'))
                        found += 1

            # Also check outputs for internal transfers
            output_addrs = []
            output_values = []
            for vout in tx.get('vout', []):
                addr = vout.get('scriptPubKey', {}).get('address')
                value = vout.get('value', 0)
                if addr:
                    output_addrs.append(addr)
                    output_values.append(value)

            # Change detection (2 outputs, smaller is change)
            if len(output_addrs) == 2 and len(output_values) == 2:
                change_idx = 0 if output_values[0] < output_values[1] else 1
                change_addr = output_addrs[change_idx]

                with self.lock:
                    if change_addr not in self.addresses:
                        self.addresses.add(change_addr)
                        self.addr_to_exchange[change_addr] = known_ex
                        batch.append((change_addr, known_ex, datetime.now().isoformat(), 'turbo_change'))
                        found += 1

        # Save batch
        if batch:
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                c = conn.cursor()
                c.executemany("""
                    INSERT OR IGNORE INTO addresses (address, exchange, discovered_at, source)
                    VALUES (?, ?, ?, ?)
                """, batch)
                conn.commit()
                conn.close()

        return txs, found

    def _worker(self, block_queue: Queue, result_queue: Queue):
        """Worker thread to process blocks."""
        while True:
            height = block_queue.get()
            if height is None:
                break

            try:
                txs, found = self._process_block(height)
                result_queue.put((height, txs, found))
            except Exception as e:
                result_queue.put((height, 0, 0))

            block_queue.task_done()

    def scan_range(self, start: int, end: int):
        """Scan block range with parallel workers."""
        total = end - start + 1
        self.start_time = time.time()

        print(f"Scanning blocks {start:,} to {end:,} ({total:,} blocks)")
        print(f"Using {self.workers} parallel workers")
        print()

        # Create queues
        block_queue = Queue()
        result_queue = Queue()

        # Start workers
        threads = []
        for _ in range(self.workers):
            t = threading.Thread(target=self._worker, args=(block_queue, result_queue))
            t.daemon = True
            t.start()
            threads.append(t)

        # Add blocks to queue
        for height in range(start, end + 1):
            block_queue.put(height)

        # Process results
        blocks_received = 0
        while blocks_received < total:
            height, txs, found = result_queue.get()
            blocks_received += 1
            self.blocks_done += 1
            self.txs_done += txs
            self.discovered += found

            elapsed = time.time() - self.start_time
            rate = self.blocks_done / max(elapsed, 1)
            eta = (total - self.blocks_done) / max(rate, 0.01)

            if found > 0:
                print(f"\n[FOUND] Block {height}: +{found} addresses")

            print(f"\r[{self.blocks_done:,}/{total:,}] {rate:.1f} blk/s | "
                  f"TXs: {self.txs_done:,} | "
                  f"Addrs: {len(self.addresses):,} (+{self.discovered:,}) | "
                  f"ETA: {eta/60:.1f}m", end='', flush=True)

        # Stop workers
        for _ in range(self.workers):
            block_queue.put(None)

        for t in threads:
            t.join()

        self._print_final()

    def scan_recent(self, num_blocks: int = 10000):
        """Scan recent blocks."""
        end = self.height
        start = max(0, end - num_blocks)
        self.scan_range(start, end)

    def _print_final(self):
        """Print final statistics."""
        elapsed = time.time() - self.start_time

        exchange_counts = defaultdict(int)
        for addr, ex in self.addr_to_exchange.items():
            exchange_counts[ex] += 1

        print()
        print()
        print("=" * 70)
        print("TURBO CLUSTER COMPLETE")
        print("=" * 70)
        print(f"Time: {elapsed/60:.1f} minutes")
        print(f"Blocks: {self.blocks_done:,}")
        print(f"Transactions: {self.txs_done:,}")
        print(f"New addresses: {self.discovered:,}")
        print(f"Total addresses: {len(self.addresses):,}")
        print()
        print("Major exchanges:")
        for ex in ['binance', 'coinbase', 'okx', 'bybit', 'bitfinex', 'kraken', 'huobi']:
            print(f"  {ex:<15} {exchange_counts.get(ex, 0):>10,}")
        print("=" * 70)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Turbo parallel blockchain scanner')
    parser.add_argument('--start', type=int, help='Start block')
    parser.add_argument('--end', type=int, help='End block')
    parser.add_argument('--recent', type=int, default=50000, help='Scan last N blocks')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--user', default='bitcoin', help='RPC user')
    parser.add_argument('--pass', dest='passwd', default='bitcoin123secure', help='RPC password')
    args = parser.parse_args()

    scanner = TurboCluster(
        rpc_user=args.user,
        rpc_pass=args.passwd,
        workers=args.workers
    )

    if args.start is not None:
        end = args.end if args.end else scanner.height
        scanner.scan_range(args.start, end)
    else:
        scanner.scan_recent(args.recent)


if __name__ == '__main__':
    main()
