#!/usr/bin/env python3
"""
HISTORICAL BACKFILL - RAPID ADDRESS DISCOVERY
==============================================
Process historical blocks to expand 71 seeds → millions of addresses.

Strategy:
1. Start from recent blocks (highest exchange activity)
2. Process each transaction through clustering
3. Discover new addresses via common-input-ownership
4. Save progress continuously

Run on VPS:
    python3 blockchain/historical_backfill.py
"""

import sys
import time
import json
import sqlite3
import subprocess
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Set, Optional

sys.path.insert(0, '/root/sovereign')
sys.path.insert(0, '/root/sovereign/blockchain')

try:
    from cluster_runner import SQLiteAddressCluster
    from tx_decoder import TransactionDecoder
except ImportError:
    from blockchain.cluster_runner import SQLiteAddressCluster
    from blockchain.tx_decoder import TransactionDecoder


class HistoricalBackfill:
    """
    Rapidly process historical blocks to discover exchange addresses.
    """

    def __init__(self,
                 cluster_db: str = "/root/sovereign/address_clusters.db",
                 start_block: Optional[int] = None,
                 batch_size: int = 100):

        print("=" * 70)
        print("HISTORICAL BACKFILL - RAPID ADDRESS DISCOVERY")
        print("=" * 70)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        self.cluster = SQLiteAddressCluster(cluster_db)
        self.decoder = TransactionDecoder()
        self.batch_size = batch_size

        # Get current block height
        self.current_height = self._get_block_height()
        print(f"Current block height: {self.current_height:,}")

        # Start from specified block or 50,000 blocks back
        if start_block is None:
            self.start_block = max(0, self.current_height - 50000)
        else:
            self.start_block = start_block

        print(f"Starting from block: {self.start_block:,}")
        print(f"Blocks to process: {self.current_height - self.start_block:,}")
        print()

        # Stats
        self.blocks_processed = 0
        self.txs_processed = 0
        self.addresses_discovered = 0
        self.start_time = 0
        self.lock = threading.Lock()

    def _get_block_height(self) -> int:
        """Get current blockchain height."""
        try:
            result = subprocess.run(
                ['bitcoin-cli', 'getblockcount'],
                capture_output=True, text=True, timeout=10
            )
            return int(result.stdout.strip())
        except Exception as e:
            print(f"Error getting block height: {e}")
            return 875000  # Approximate current height

    def _get_block_hash(self, height: int) -> Optional[str]:
        """Get block hash for height."""
        try:
            result = subprocess.run(
                ['bitcoin-cli', 'getblockhash', str(height)],
                capture_output=True, text=True, timeout=10
            )
            return result.stdout.strip()
        except Exception:
            return None

    def _get_block(self, block_hash: str) -> Optional[Dict]:
        """Get full block with transactions."""
        try:
            result = subprocess.run(
                ['bitcoin-cli', 'getblock', block_hash, '2'],
                capture_output=True, text=True, timeout=60
            )
            return json.loads(result.stdout)
        except Exception:
            return None

    def _process_block(self, height: int) -> int:
        """Process a single block and return addresses discovered."""
        block_hash = self._get_block_hash(height)
        if not block_hash:
            return 0

        block = self._get_block(block_hash)
        if not block:
            return 0

        discovered = 0
        txs = block.get('tx', [])

        for tx in txs:
            # Convert to our format
            parsed_tx = self._parse_tx(tx)
            if parsed_tx:
                new_addrs = self.cluster.process_transaction(parsed_tx)
                discovered += len(new_addrs)

        with self.lock:
            self.blocks_processed += 1
            self.txs_processed += len(txs)
            self.addresses_discovered += discovered

        return discovered

    def _parse_tx(self, tx: Dict) -> Optional[Dict]:
        """Parse raw transaction to our format."""
        try:
            inputs = []
            for vin in tx.get('vin', []):
                if 'coinbase' in vin:
                    continue
                # For historical, we need prevout info
                prevout = vin.get('prevout', {})
                addr = None
                if prevout:
                    scriptpubkey = prevout.get('scriptPubKey', {})
                    addr = scriptpubkey.get('address')
                inputs.append({
                    'prev_txid': vin.get('txid'),
                    'prev_vout': vin.get('vout'),
                    'address': addr
                })

            outputs = []
            for i, vout in enumerate(tx.get('vout', [])):
                scriptpubkey = vout.get('scriptPubKey', {})
                addr = scriptpubkey.get('address')
                outputs.append({
                    'address': addr,
                    'btc': vout.get('value', 0),
                    'n': i
                })

            return {
                'txid': tx.get('txid'),
                'inputs': inputs,
                'outputs': outputs
            }
        except Exception:
            return None

    def _print_progress(self):
        """Print progress stats."""
        elapsed = time.time() - self.start_time
        blocks_per_sec = self.blocks_processed / max(elapsed, 1)
        remaining = self.current_height - self.start_block - self.blocks_processed
        eta_sec = remaining / max(blocks_per_sec, 0.01)

        stats = self.cluster.get_stats()

        print(f"\r[{self.blocks_processed:,}/{self.current_height - self.start_block:,}] "
              f"{blocks_per_sec:.1f} blk/s | "
              f"TXs: {self.txs_processed:,} | "
              f"Addrs: {stats['total_addresses']:,} (+{self.addresses_discovered:,}) | "
              f"ETA: {eta_sec/60:.0f}m", end='', flush=True)

    def run_sequential(self):
        """Process blocks sequentially (slower but reliable)."""
        self.start_time = time.time()

        print(f"Processing {self.current_height - self.start_block:,} blocks sequentially...")
        print()

        for height in range(self.start_block, self.current_height + 1):
            self._process_block(height)

            if self.blocks_processed % 10 == 0:
                self._print_progress()

            if self.blocks_processed % 100 == 0:
                self.cluster.save_stats()

        print()
        self._print_final_stats()

    def run_batch(self, workers: int = 4):
        """Process blocks in parallel batches."""
        self.start_time = time.time()

        print(f"Processing {self.current_height - self.start_block:,} blocks with {workers} workers...")
        print()

        heights = list(range(self.start_block, self.current_height + 1))

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(self._process_block, h): h for h in heights}

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception:
                    pass

                if self.blocks_processed % 10 == 0:
                    self._print_progress()

                if self.blocks_processed % 100 == 0:
                    self.cluster.save_stats()

        print()
        self._print_final_stats()

    def _print_final_stats(self):
        """Print final statistics."""
        elapsed = time.time() - self.start_time
        stats = self.cluster.get_stats()

        print()
        print("=" * 70)
        print("BACKFILL COMPLETE")
        print("=" * 70)
        print(f"Time: {elapsed/60:.1f} minutes")
        print(f"Blocks: {self.blocks_processed:,}")
        print(f"Transactions: {self.txs_processed:,}")
        print(f"Addresses discovered: {self.addresses_discovered:,}")
        print(f"Total addresses: {stats['total_addresses']:,}")
        print()
        print("Per Exchange:")
        for ex, count in sorted(stats['per_exchange'].items(), key=lambda x: -x[1]):
            print(f"  {ex:<15} {count:>10,}")
        print("=" * 70)


class FastBackfill:
    """
    Ultra-fast backfill using getblock with verbosity=1 + batch getrawtransaction.
    Processes blocks in memory without repeated RPC calls.
    """

    def __init__(self, cluster_db: str = "/root/sovereign/address_clusters.db"):
        print("=" * 70)
        print("FAST HISTORICAL BACKFILL")
        print("=" * 70)

        self.cluster = SQLiteAddressCluster(cluster_db)
        self.current_height = self._rpc('getblockcount')
        print(f"Block height: {self.current_height:,}")
        print(f"Starting addresses: {len(self.cluster.all_addresses):,}")
        print()

        self.blocks_processed = 0
        self.txs_processed = 0
        self.start_time = 0

    def _rpc(self, method: str, *params) -> any:
        """Execute bitcoin-cli RPC call."""
        cmd = ['bitcoin-cli', method] + [str(p) for p in params]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            return None
        try:
            return json.loads(result.stdout)
        except:
            return result.stdout.strip()

    def process_blocks(self, start: int, end: int):
        """Process a range of blocks."""
        self.start_time = time.time()
        total = end - start + 1

        print(f"Processing blocks {start:,} to {end:,} ({total:,} blocks)...")
        print()

        for height in range(start, end + 1):
            # Get block with full transaction data (verbosity=2)
            block_hash = self._rpc('getblockhash', height)
            if not block_hash:
                continue

            block = self._rpc('getblock', block_hash, 2)
            if not block:
                continue

            # Process each transaction
            for tx in block.get('tx', []):
                self._process_tx(tx)
                self.txs_processed += 1

            self.blocks_processed += 1

            if self.blocks_processed % 5 == 0:
                self._print_progress(total)

            if self.blocks_processed % 50 == 0:
                self.cluster.save_stats()

        self._print_final()

    def _process_tx(self, tx: Dict):
        """Process transaction for address clustering."""
        # Extract input addresses from prevout
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
            return

        # Check if any input is known
        known_exchange = None
        for addr in input_addrs:
            if addr in self.cluster.address_to_exchange:
                known_exchange = self.cluster.address_to_exchange[addr]
                break

        if not known_exchange:
            return

        # Add all input addresses to cluster
        for addr in input_addrs:
            if addr not in self.cluster.all_addresses:
                self.cluster.address_to_exchange[addr] = known_exchange
                self.cluster.all_addresses.add(addr)
                self.cluster.session_discovered += 1

                # Persist
                conn = sqlite3.connect(self.cluster.db_path)
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR IGNORE INTO addresses (address, exchange, discovered_at, source)
                    VALUES (?, ?, ?, 'backfill')
                """, (addr, known_exchange, datetime.now().isoformat()))
                conn.commit()
                conn.close()

    def _print_progress(self, total: int):
        """Print progress."""
        elapsed = time.time() - self.start_time
        rate = self.blocks_processed / max(elapsed, 1)
        remaining = total - self.blocks_processed
        eta = remaining / max(rate, 0.01)

        print(f"\r[{self.blocks_processed:,}/{total:,}] "
              f"{rate:.1f} blk/s | "
              f"TXs: {self.txs_processed:,} | "
              f"Addrs: {len(self.cluster.all_addresses):,} | "
              f"ETA: {eta/60:.0f}m", end='', flush=True)

    def _print_final(self):
        """Print final stats."""
        elapsed = time.time() - self.start_time
        stats = self.cluster.get_stats()

        print()
        print()
        print("=" * 70)
        print("BACKFILL COMPLETE")
        print("=" * 70)
        print(f"Time: {elapsed/60:.1f} minutes")
        print(f"Blocks: {self.blocks_processed:,}")
        print(f"Transactions: {self.txs_processed:,}")
        print(f"Total addresses: {stats['total_addresses']:,}")
        print()
        for ex, count in sorted(stats['per_exchange'].items(), key=lambda x: -x[1]):
            print(f"  {ex:<15} {count:>10,}")


def main():
    """Run historical backfill."""
    import argparse

    parser = argparse.ArgumentParser(description='Historical address backfill')
    parser.add_argument('--start', type=int, help='Start block height')
    parser.add_argument('--blocks', type=int, default=10000, help='Number of blocks to process')
    parser.add_argument('--fast', action='store_true', help='Use fast mode')
    args = parser.parse_args()

    if args.fast:
        backfill = FastBackfill()
        end = backfill.current_height
        start = args.start if args.start else end - args.blocks
        backfill.process_blocks(start, end)
    else:
        backfill = HistoricalBackfill(start_block=args.start)
        backfill.run_sequential()


if __name__ == '__main__':
    main()
