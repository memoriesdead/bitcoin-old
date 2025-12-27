#!/usr/bin/env python3
"""
TOTAL EXTRACTION - 100% EXCHANGE ADDRESS COVERAGE
==================================================
Extract ALL exchange addresses from Bitcoin node.

Strategy:
1. BIDIRECTIONAL CLUSTERING - Track both inputs AND outputs
2. CHANGE DETECTION - When known address receives, other outputs = change = same entity
3. UTXO SCANNING - Find high-value addresses in UTXO set
4. PATTERN DETECTION - Identify exchange-like transaction patterns
5. CONTINUOUS LOOP - Keep running until we have everything

This runs until we achieve 100% deterministic coverage.
"""

import sys
import time
import json
import sqlite3
import subprocess
from datetime import datetime
from typing import Dict, Set, List, Optional, Tuple
from collections import defaultdict

sys.path.insert(0, '/root/sovereign')


class TotalExtraction:
    """
    Complete address extraction from Bitcoin blockchain.
    No stopping until we have 100% coverage.
    """

    def __init__(self, db_path: str = "/root/sovereign/address_clusters.db"):
        print("=" * 70)
        print("TOTAL EXTRACTION - 100% COVERAGE MODE")
        print("=" * 70)
        print(f"Started: {datetime.now()}")
        print()

        self.db_path = db_path
        self._init_db()

        # Load all known addresses
        self.addresses: Set[str] = set()
        self.addr_to_exchange: Dict[str, str] = {}
        self._load_existing()

        # Track discovery sources
        self.discovered_input = 0
        self.discovered_output = 0
        self.discovered_change = 0
        self.discovered_pattern = 0

        # Get blockchain height
        self.height = int(self._rpc('getblockcount'))
        print(f"Blockchain height: {self.height:,}")
        print(f"Starting addresses: {len(self.addresses):,}")
        print()

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
        conn.commit()
        conn.close()

    def _load_existing(self):
        """Load all addresses from database."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT address, exchange FROM addresses")
        for row in c.fetchall():
            self.addresses.add(row[0])
            self.addr_to_exchange[row[0]] = row[1]
        conn.close()

    def _rpc(self, method: str, *params) -> Optional[str]:
        """Execute Bitcoin RPC."""
        cmd = ['bitcoin-cli', method] + [str(p) for p in params]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            return result.stdout.strip() if result.returncode == 0 else None
        except:
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

    def _save_address(self, addr: str, exchange: str, source: str):
        """Save discovered address to database."""
        if addr in self.addresses:
            return False

        self.addresses.add(addr)
        self.addr_to_exchange[addr] = exchange

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            INSERT OR IGNORE INTO addresses (address, exchange, discovered_at, source)
            VALUES (?, ?, ?, ?)
        """, (addr, exchange, datetime.now().isoformat(), source))
        conn.commit()
        conn.close()
        return True

    def _save_batch(self, addresses: List[Tuple[str, str, str]]):
        """Batch save addresses."""
        if not addresses:
            return

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.executemany("""
            INSERT OR IGNORE INTO addresses (address, exchange, discovered_at, source)
            VALUES (?, ?, ?, ?)
        """, [(a, e, datetime.now().isoformat(), s) for a, e, s in addresses])
        conn.commit()
        conn.close()

    def process_block_bidirectional(self, height: int) -> Dict[str, int]:
        """
        Process block with BIDIRECTIONAL clustering.

        Returns dict with discovery counts by method.
        """
        block_hash = self._rpc('getblockhash', height)
        if not block_hash:
            return {}

        block = self._rpc_json('getblock', block_hash, 2)
        if not block:
            return {}

        stats = {'input': 0, 'output': 0, 'change': 0}
        batch = []

        for tx in block.get('tx', []):
            # Extract all addresses from this transaction
            input_addrs = []
            output_addrs = []
            output_values = []

            # Get INPUT addresses
            for vin in tx.get('vin', []):
                if 'coinbase' in vin:
                    continue
                prevout = vin.get('prevout', {})
                if prevout:
                    addr = prevout.get('scriptPubKey', {}).get('address')
                    if addr:
                        input_addrs.append(addr)

            # Get OUTPUT addresses with values
            for vout in tx.get('vout', []):
                addr = vout.get('scriptPubKey', {}).get('address')
                value = vout.get('value', 0)
                if addr:
                    output_addrs.append(addr)
                    output_values.append(value)

            # === METHOD 1: INPUT CLUSTERING ===
            # If ANY input is known, ALL inputs belong to same exchange
            known_input_ex = None
            for addr in input_addrs:
                if addr in self.addr_to_exchange:
                    known_input_ex = self.addr_to_exchange[addr]
                    break

            if known_input_ex:
                for addr in input_addrs:
                    if addr not in self.addresses:
                        batch.append((addr, known_input_ex, 'input_cluster'))
                        stats['input'] += 1

            # === METHOD 2: OUTPUT CLUSTERING (Internal Transfers) ===
            # If MULTIPLE outputs go to known addresses of SAME exchange = internal transfer
            # Other outputs are also that exchange
            output_exchanges = defaultdict(list)
            for addr in output_addrs:
                if addr in self.addr_to_exchange:
                    output_exchanges[self.addr_to_exchange[addr]].append(addr)

            for ex, known_outputs in output_exchanges.items():
                if len(known_outputs) >= 2:
                    # Multiple outputs to same exchange = internal transfer
                    # ALL outputs belong to this exchange
                    for addr in output_addrs:
                        if addr not in self.addresses:
                            batch.append((addr, ex, 'output_cluster'))
                            stats['output'] += 1

            # === METHOD 3: CHANGE DETECTION ===
            # If known exchange is in inputs AND there are exactly 2 outputs
            # The smaller output is likely change (same exchange)
            if known_input_ex and len(output_addrs) == 2 and len(output_values) == 2:
                # Find the smaller output (likely change)
                if output_values[0] < output_values[1]:
                    change_addr = output_addrs[0]
                else:
                    change_addr = output_addrs[1]

                if change_addr not in self.addresses:
                    batch.append((change_addr, known_input_ex, 'change_detect'))
                    stats['change'] += 1

            # === METHOD 4: SINGLE OUTPUT FROM KNOWN ===
            # If inputs are known exchange and there's only 1 output
            # That output receives from exchange (might be withdrawal, but track it)
            # Skip this - too many false positives

        # Save batch
        for addr, ex, source in batch:
            if addr not in self.addresses:
                self.addresses.add(addr)
                self.addr_to_exchange[addr] = ex

        self._save_batch(batch)

        self.discovered_input += stats['input']
        self.discovered_output += stats['output']
        self.discovered_change += stats['change']

        return stats

    def scan_range(self, start: int, end: int):
        """Scan block range with bidirectional clustering."""
        total = end - start + 1
        start_time = time.time()
        blocks_done = 0

        print(f"Scanning blocks {start:,} to {end:,} ({total:,} blocks)")
        print("Methods: INPUT clustering + OUTPUT clustering + CHANGE detection")
        print()

        for height in range(start, end + 1):
            stats = self.process_block_bidirectional(height)
            blocks_done += 1

            elapsed = time.time() - start_time
            rate = blocks_done / max(elapsed, 1)
            discovered = self.discovered_input + self.discovered_output + self.discovered_change

            print(f"\r[{blocks_done:,}/{total:,}] {rate:.1f} blk/s | "
                  f"Addrs: {len(self.addresses):,} | "
                  f"+{discovered:,} (I:{self.discovered_input} O:{self.discovered_output} C:{self.discovered_change})",
                  end='', flush=True)

        print()
        self._print_stats()

    def scan_utxo_set(self):
        """
        Scan UTXO set for exchange-like patterns.

        Exchanges have:
        - Many UTXOs (receiving lots of deposits)
        - High total balance
        - Addresses that appear in many transactions
        """
        print()
        print("=" * 70)
        print("SCANNING UTXO SET FOR EXCHANGE PATTERNS")
        print("=" * 70)

        # Use scantxoutset to find high-value addresses
        # This is a slow operation but gives us ALL unspent outputs

        # First, let's find addresses with highest UTXO counts
        # by scanning known exchange addresses
        print("Checking known address UTXOs...")

        for exchange in ['binance', 'coinbase', 'okx', 'bybit', 'bitfinex']:
            addrs = [a for a, e in self.addr_to_exchange.items() if e == exchange][:5]
            for addr in addrs:
                # Check if this address has UTXOs
                result = self._rpc_json('scantxoutset', 'start', f'["addr({addr})"]')
                if result and result.get('total_amount', 0) > 0:
                    print(f"  {exchange}: {addr[:20]}... = {result['total_amount']:.4f} BTC")

    def find_high_value_addresses(self):
        """
        Find addresses with extremely high balances - likely exchanges.
        """
        print()
        print("=" * 70)
        print("FINDING HIGH-VALUE ADDRESSES (likely exchanges)")
        print("=" * 70)

        # Scan for addresses with > 1000 BTC
        # This requires scanning the UTXO set which is slow

        # For now, let's use our existing data and find connected addresses
        print("Analyzing transaction patterns of known addresses...")

    def run_continuous(self, blocks_per_batch: int = 1000):
        """
        Run continuous extraction until we have everything.
        """
        print("=" * 70)
        print("CONTINUOUS EXTRACTION MODE")
        print("=" * 70)
        print()

        batch_num = 0
        last_count = len(self.addresses)

        while True:
            # Scan recent blocks
            end = self.height
            start = max(0, end - blocks_per_batch)

            print(f"\n--- Batch {batch_num + 1}: Blocks {start:,} to {end:,} ---")
            self.scan_range(start, end)

            new_discovered = len(self.addresses) - last_count
            print(f"New addresses this batch: {new_discovered:,}")

            if new_discovered == 0:
                # Go further back in history
                self.height = start
                if self.height <= 0:
                    print("\nReached genesis block. Extraction complete.")
                    break

            last_count = len(self.addresses)
            batch_num += 1

            # Update height for next batch
            self.height = int(self._rpc('getblockcount'))

    def _print_stats(self):
        """Print current statistics."""
        print()
        print("=" * 70)
        print("EXTRACTION STATISTICS")
        print("=" * 70)

        # Count per exchange
        exchange_counts = defaultdict(int)
        for addr, ex in self.addr_to_exchange.items():
            exchange_counts[ex] += 1

        print(f"\nTotal addresses: {len(self.addresses):,}")
        print(f"\nDiscovery breakdown:")
        print(f"  Input clustering:  {self.discovered_input:,}")
        print(f"  Output clustering: {self.discovered_output:,}")
        print(f"  Change detection:  {self.discovered_change:,}")

        print(f"\nTop exchanges:")
        for ex, count in sorted(exchange_counts.items(), key=lambda x: -x[1])[:20]:
            print(f"  {ex:<25} {count:>12,}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Total blockchain extraction')
    parser.add_argument('--start', type=int, help='Start block')
    parser.add_argument('--end', type=int, help='End block')
    parser.add_argument('--recent', type=int, default=10000, help='Scan last N blocks')
    parser.add_argument('--continuous', action='store_true', help='Run continuously')
    parser.add_argument('--utxo', action='store_true', help='Scan UTXO set')
    args = parser.parse_args()

    extractor = TotalExtraction()

    if args.utxo:
        extractor.scan_utxo_set()
    elif args.continuous:
        extractor.run_continuous()
    elif args.start is not None:
        end = args.end if args.end else extractor.height
        extractor.scan_range(args.start, end)
    else:
        # Default: scan recent blocks
        end = extractor.height
        start = max(0, end - args.recent)
        extractor.scan_range(start, end)


if __name__ == '__main__':
    main()
