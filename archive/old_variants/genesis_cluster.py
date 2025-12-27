#!/usr/bin/env python3
"""
GENESIS CLUSTER - 100% Address Discovery
=========================================
Scans entire blockchain from genesis to find ALL exchange addresses.

Uses common-input-ownership heuristic:
- If 2 addresses are inputs to same TX = same owner
- Seeds from 7.5M known addresses
- Discovers ALL linked addresses

Run: python3 genesis_cluster.py
"""

import sqlite3
import subprocess
import json
import time
import os
from datetime import datetime
from typing import Set, Dict, List

# Config
DB_PATH = "/root/sovereign/walletexplorer_addresses.db"
PROGRESS_FILE = "/root/sovereign/genesis_progress.json"
BATCH_SIZE = 100  # Blocks per batch
SAVE_INTERVAL = 1000  # Save progress every N blocks

class GenesisCluster:
    def __init__(self):
        print("=" * 70)
        print("GENESIS CLUSTER - 100% ADDRESS DISCOVERY")
        print("=" * 70)
        print(f"Started: {datetime.now()}")
        print()

        self.addresses: Set[str] = set()
        self.addr_to_exchange: Dict[str, str] = {}
        self.new_addresses = 0
        self.blocks_scanned = 0
        self.start_block = 0

        self._load_addresses()
        self._load_progress()

    def _load_addresses(self):
        """Load all known addresses from database."""
        print("Loading known addresses...")
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT address, exchange FROM addresses")
        for row in c.fetchall():
            self.addresses.add(row[0])
            self.addr_to_exchange[row[0]] = row[1]
        conn.close()
        print(f"Loaded {len(self.addresses):,} seed addresses")
        print()

    def _load_progress(self):
        """Load scan progress."""
        if os.path.exists(PROGRESS_FILE):
            with open(PROGRESS_FILE, 'r') as f:
                data = json.load(f)
                self.start_block = data.get('last_block', 0) + 1
                self.new_addresses = data.get('new_addresses', 0)
                print(f"Resuming from block {self.start_block:,}")
                print(f"Previously found: {self.new_addresses:,} new addresses")
        else:
            print("Starting fresh from genesis (block 0)")

    def _save_progress(self, block_height: int):
        """Save scan progress."""
        with open(PROGRESS_FILE, 'w') as f:
            json.dump({
                'last_block': block_height,
                'new_addresses': self.new_addresses,
                'timestamp': datetime.now().isoformat(),
                'total_addresses': len(self.addresses)
            }, f)

    def _save_addresses(self, new_addrs: List[tuple]):
        """Batch save new addresses to database."""
        if not new_addrs:
            return

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        now = datetime.now().isoformat()

        for addr, exchange in new_addrs:
            try:
                c.execute(
                    "INSERT OR IGNORE INTO addresses (address, exchange, downloaded_at) VALUES (?, ?, ?)",
                    (addr, exchange, now)
                )
                if c.rowcount > 0:
                    self.new_addresses += 1
            except:
                pass

        conn.commit()
        conn.close()

    def _get_block(self, height: int) -> dict:
        """Get block with full transaction data."""
        try:
            # Get block hash
            result = subprocess.run(
                ['/usr/local/bin/bitcoin-cli', 'getblockhash', str(height)],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode != 0:
                return None
            block_hash = result.stdout.strip()

            # Get block with verbosity 2 (full TX data)
            result = subprocess.run(
                ['/usr/local/bin/bitcoin-cli', 'getblock', block_hash, '2'],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode != 0:
                return None

            return json.loads(result.stdout)
        except Exception as e:
            return None

    def _process_block(self, block: dict) -> List[tuple]:
        """Process block and find new exchange addresses."""
        new_addrs = []

        for tx in block.get('tx', []):
            # Skip coinbase
            if 'coinbase' in tx.get('vin', [{}])[0]:
                continue

            # Get all input addresses
            input_addrs = []
            for vin in tx.get('vin', []):
                # For full block data, we need prevout info
                prevout = vin.get('prevout', {})
                if prevout:
                    spk = prevout.get('scriptPubKey', {})
                    addr = spk.get('address')
                    if addr:
                        input_addrs.append(addr)

            # Get all output addresses
            output_addrs = []
            output_values = []
            for vout in tx.get('vout', []):
                spk = vout.get('scriptPubKey', {})
                addr = spk.get('address')
                value = vout.get('value', 0)
                if addr:
                    output_addrs.append(addr)
                    output_values.append(value)

            # Find known exchange in this TX
            known_exchange = None
            known_in_inputs = False

            for addr in input_addrs:
                if addr in self.addr_to_exchange:
                    known_exchange = self.addr_to_exchange[addr]
                    known_in_inputs = True
                    break

            if not known_exchange:
                for addr in output_addrs:
                    if addr in self.addr_to_exchange:
                        known_exchange = self.addr_to_exchange[addr]
                        break

            if not known_exchange:
                continue

            # CLUSTER: All inputs belong to same entity
            if known_in_inputs:
                for addr in input_addrs:
                    if addr not in self.addresses:
                        self.addresses.add(addr)
                        self.addr_to_exchange[addr] = known_exchange
                        new_addrs.append((addr, known_exchange))

                # Change output detection (smaller output in 2-output TX)
                if len(output_addrs) == 2 and len(output_values) == 2:
                    if output_values[0] < output_values[1]:
                        change_addr = output_addrs[0]
                    else:
                        change_addr = output_addrs[1]
                    if change_addr not in self.addresses:
                        self.addresses.add(change_addr)
                        self.addr_to_exchange[change_addr] = known_exchange
                        new_addrs.append((change_addr, known_exchange))

        return new_addrs

    def scan(self):
        """Main scan loop - genesis to current."""
        # Get current block height
        result = subprocess.run(
            ['/usr/local/bin/bitcoin-cli', 'getblockcount'],
            capture_output=True, text=True, timeout=10
        )
        current_height = int(result.stdout.strip())

        print(f"Scanning blocks {self.start_block:,} to {current_height:,}")
        print(f"Total blocks to scan: {current_height - self.start_block:,}")
        print()

        start_time = time.time()
        last_print = time.time()

        for height in range(self.start_block, current_height + 1):
            # Get and process block
            block = self._get_block(height)
            if block:
                new_addrs = self._process_block(block)
                if new_addrs:
                    self._save_addresses(new_addrs)

            self.blocks_scanned += 1

            # Progress update every 10 seconds
            if time.time() - last_print > 10:
                elapsed = time.time() - start_time
                rate = self.blocks_scanned / elapsed if elapsed > 0 else 0
                remaining = (current_height - height) / rate if rate > 0 else 0

                print(f"Block {height:,}/{current_height:,} | "
                      f"+{self.new_addresses:,} addrs | "
                      f"{rate:.1f} blk/s | "
                      f"ETA: {remaining/3600:.1f}h")
                last_print = time.time()

            # Save progress periodically
            if height % SAVE_INTERVAL == 0:
                self._save_progress(height)

        # Final save
        self._save_progress(current_height)

        print()
        print("=" * 70)
        print("GENESIS SCAN COMPLETE")
        print("=" * 70)
        print(f"Blocks scanned: {self.blocks_scanned:,}")
        print(f"New addresses found: {self.new_addresses:,}")
        print(f"Total addresses: {len(self.addresses):,}")
        print(f"Time: {(time.time() - start_time)/3600:.2f} hours")


def main():
    cluster = GenesisCluster()
    cluster.scan()


if __name__ == '__main__':
    main()
