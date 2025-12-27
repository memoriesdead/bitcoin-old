#!/usr/bin/env python3
"""
GENESIS CLUSTER V3 - FAST VERSION
==================================
Uses getblock verbosity 3 for inline prevout data (Bitcoin Core 25+)
No individual transaction lookups needed = 10x faster
"""

import sqlite3
import subprocess
import json
import time
import os
from datetime import datetime
from typing import Set, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

DB_PATH = "/root/sovereign/walletexplorer_addresses.db"
PROGRESS_FILE = "/root/sovereign/genesis_v3_progress.json"
SAVE_INTERVAL = 1000
WORKERS = 4  # Parallel block fetching

class GenesisClusterV3:
    def __init__(self):
        print("=" * 70)
        print("GENESIS CLUSTER V3 - FAST MODE")
        print("=" * 70)
        self.addresses: Set[str] = set()
        self.addr_to_exchange: Dict[str, str] = {}
        self.new_addresses = 0
        self.blocks_scanned = 0
        self.start_block = 0
        self._load_addresses()
        self._load_progress()

    def _load_addresses(self):
        print("Loading addresses...")
        conn = sqlite3.connect(DB_PATH)
        for row in conn.cursor().execute("SELECT address, exchange FROM addresses"):
            self.addresses.add(row[0])
            self.addr_to_exchange[row[0]] = row[1]
        conn.close()
        print(f"Loaded {len(self.addresses):,} seeds")

    def _load_progress(self):
        if os.path.exists(PROGRESS_FILE):
            with open(PROGRESS_FILE) as f:
                data = json.load(f)
                self.start_block = data.get('last_block', 0) + 1
                self.new_addresses = data.get('new_addresses', 0)
                print(f"Resuming from block {self.start_block:,}")
        else:
            print("Starting fresh")

    def _save_progress(self, h):
        with open(PROGRESS_FILE, 'w') as f:
            json.dump({'last_block': h, 'new_addresses': self.new_addresses}, f)

    def _save_addresses(self, addrs):
        if not addrs:
            return
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        now = datetime.now().isoformat()
        for addr, ex in addrs:
            try:
                c.execute("INSERT OR IGNORE INTO addresses (address,exchange,downloaded_at) VALUES (?,?,?)",
                         (addr, ex, now))
                if c.rowcount > 0:
                    self.new_addresses += 1
            except:
                pass
        conn.commit()
        conn.close()

    def _get_block_fast(self, height: int) -> dict:
        """Get block with verbosity 3 (includes prevout data)."""
        try:
            r = subprocess.run(
                ['/usr/local/bin/bitcoin-cli', 'getblockhash', str(height)],
                capture_output=True, text=True, timeout=10
            )
            if r.returncode != 0:
                return None

            # Verbosity 3 includes prevout info inline
            r = subprocess.run(
                ['/usr/local/bin/bitcoin-cli', 'getblock', r.stdout.strip(), '3'],
                capture_output=True, text=True, timeout=60
            )
            if r.returncode != 0:
                return None
            return json.loads(r.stdout)
        except:
            return None

    def _process_block(self, block: dict) -> List[tuple]:
        """Process block for address clustering."""
        new_addrs = []

        for tx in block.get('tx', []):
            vin = tx.get('vin', [])

            # Skip coinbase
            if vin and 'coinbase' in vin[0]:
                continue

            # Get input addresses from prevout (inline in verbosity 3)
            input_addrs = []
            for v in vin:
                prevout = v.get('prevout', {})
                spk = prevout.get('scriptPubKey', {})
                addr = spk.get('address')
                if addr:
                    input_addrs.append(addr)

            # Get output addresses
            output_addrs = []
            output_values = []
            for vout in tx.get('vout', []):
                spk = vout.get('scriptPubKey', {})
                addr = spk.get('address')
                val = vout.get('value', 0)
                if addr:
                    output_addrs.append(addr)
                    output_values.append(val)

            # Find known exchange
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

            # Cluster: all inputs = same owner
            if known_in_inputs and len(input_addrs) > 1:
                for addr in input_addrs:
                    if addr not in self.addresses:
                        self.addresses.add(addr)
                        self.addr_to_exchange[addr] = known_exchange
                        new_addrs.append((addr, known_exchange))

                # Change output (smallest value)
                if len(output_addrs) >= 2 and output_values:
                    min_idx = output_values.index(min(output_values))
                    change = output_addrs[min_idx]
                    if change not in self.addresses:
                        self.addresses.add(change)
                        self.addr_to_exchange[change] = known_exchange
                        new_addrs.append((change, known_exchange))

        return new_addrs

    def scan(self):
        r = subprocess.run(
            ['/usr/local/bin/bitcoin-cli', 'getblockcount'],
            capture_output=True, text=True, timeout=10
        )
        top = int(r.stdout.strip())

        print(f"Scanning {self.start_block:,} to {top:,}")
        print(f"Blocks: {top - self.start_block:,}")
        print()

        t0 = time.time()
        last_print = t0

        for h in range(self.start_block, top + 1):
            block = self._get_block_fast(h)
            if block:
                new = self._process_block(block)
                if new:
                    self._save_addresses(new)

            self.blocks_scanned += 1

            if time.time() - last_print > 30:
                elapsed = time.time() - t0
                rate = self.blocks_scanned / elapsed
                eta = (top - h) / rate / 3600 if rate > 0 else 0
                print(f"Block {h:,}/{top:,} | +{self.new_addresses:,} | {rate:.0f} blk/s | ETA: {eta:.1f}h")
                last_print = time.time()

            if h % SAVE_INTERVAL == 0:
                self._save_progress(h)

        self._save_progress(top)
        print(f"\nDONE: +{self.new_addresses:,} addresses | Total: {len(self.addresses):,}")

if __name__ == '__main__':
    GenesisClusterV3().scan()
