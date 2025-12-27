#!/usr/bin/env python3
"""
GENESIS CLUSTER V2 - 100% Address Discovery (FIXED)
====================================================
Properly looks up input addresses from previous transactions.

The issue with V1: getblock doesn't include prevout addresses.
Fix: Use getrawtransaction to get actual spending addresses.
"""

import sqlite3
import subprocess
import json
import time
import os
from datetime import datetime
from typing import Set, Dict, List, Optional

# Config
DB_PATH = "/root/sovereign/walletexplorer_addresses.db"
PROGRESS_FILE = "/root/sovereign/genesis_v2_progress.json"
SAVE_INTERVAL = 500  # Save progress every N blocks

class GenesisClusterV2:
    def __init__(self):
        print("=" * 70)
        print("GENESIS CLUSTER V2 - FIXED INPUT ADDRESS LOOKUP")
        print("=" * 70)
        print(f"Started: {datetime.now()}")
        print()

        self.addresses: Set[str] = set()
        self.addr_to_exchange: Dict[str, str] = {}
        self.new_addresses = 0
        self.blocks_scanned = 0
        self.start_block = 0
        self.tx_cache: Dict[str, dict] = {}  # Cache for transaction lookups

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

        saved = 0
        for addr, exchange in new_addrs:
            try:
                c.execute(
                    "INSERT OR IGNORE INTO addresses (address, exchange, downloaded_at) VALUES (?, ?, ?)",
                    (addr, exchange, now)
                )
                if c.rowcount > 0:
                    saved += 1
            except:
                pass

        conn.commit()
        conn.close()
        self.new_addresses += saved

    def _get_tx(self, txid: str) -> Optional[dict]:
        """Get transaction with full prevout info."""
        if txid in self.tx_cache:
            return self.tx_cache[txid]

        try:
            result = subprocess.run(
                ['/usr/local/bin/bitcoin-cli', 'getrawtransaction', txid, '2'],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode != 0:
                return None

            tx = json.loads(result.stdout)

            # Keep cache small (last 10000 txs)
            if len(self.tx_cache) > 10000:
                # Remove oldest entries
                keys = list(self.tx_cache.keys())[:5000]
                for k in keys:
                    del self.tx_cache[k]

            self.tx_cache[txid] = tx
            return tx
        except:
            return None

    def _get_input_address(self, vin: dict) -> Optional[str]:
        """Get the address that is spending this input."""
        # Check if prevout is included (Bitcoin Core 25+)
        prevout = vin.get('prevout', {})
        if prevout:
            spk = prevout.get('scriptPubKey', {})
            return spk.get('address')

        # Otherwise look up the previous transaction
        prev_txid = vin.get('txid')
        prev_vout = vin.get('vout')

        if not prev_txid or prev_vout is None:
            return None

        prev_tx = self._get_tx(prev_txid)
        if not prev_tx:
            return None

        vouts = prev_tx.get('vout', [])
        if prev_vout >= len(vouts):
            return None

        spk = vouts[prev_vout].get('scriptPubKey', {})
        return spk.get('address')

    def _get_block_txids(self, height: int) -> List[str]:
        """Get list of transaction IDs in a block."""
        try:
            # Get block hash
            result = subprocess.run(
                ['/usr/local/bin/bitcoin-cli', 'getblockhash', str(height)],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode != 0:
                return []
            block_hash = result.stdout.strip()

            # Get block with verbosity 1 (just txids)
            result = subprocess.run(
                ['/usr/local/bin/bitcoin-cli', 'getblock', block_hash, '1'],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode != 0:
                return []

            block = json.loads(result.stdout)
            return block.get('tx', [])
        except:
            return []

    def _process_tx(self, txid: str) -> List[tuple]:
        """Process a single transaction for address clustering."""
        new_addrs = []

        tx = self._get_tx(txid)
        if not tx:
            return new_addrs

        # Skip coinbase
        vin = tx.get('vin', [])
        if vin and 'coinbase' in vin[0]:
            return new_addrs

        # Get all input addresses
        input_addrs = []
        for v in vin:
            addr = self._get_input_address(v)
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

        # Check inputs first (spending from exchange = exchange hot wallet)
        for addr in input_addrs:
            if addr in self.addr_to_exchange:
                known_exchange = self.addr_to_exchange[addr]
                known_in_inputs = True
                break

        # Check outputs (deposit to exchange)
        if not known_exchange:
            for addr in output_addrs:
                if addr in self.addr_to_exchange:
                    known_exchange = self.addr_to_exchange[addr]
                    break

        if not known_exchange:
            return new_addrs

        # CLUSTER: All inputs belong to same entity (common-input-ownership)
        if known_in_inputs and len(input_addrs) > 1:
            for addr in input_addrs:
                if addr not in self.addresses:
                    self.addresses.add(addr)
                    self.addr_to_exchange[addr] = known_exchange
                    new_addrs.append((addr, known_exchange))

            # Change output detection (smallest output in multi-output TX)
            if len(output_addrs) >= 2:
                # Find smallest output (likely change)
                min_idx = output_values.index(min(output_values))
                change_addr = output_addrs[min_idx]
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
        batch_new = 0

        for height in range(self.start_block, current_height + 1):
            # Get transaction IDs in this block
            txids = self._get_block_txids(height)

            # Process each transaction
            for txid in txids:
                new_addrs = self._process_tx(txid)
                if new_addrs:
                    self._save_addresses(new_addrs)
                    batch_new += len(new_addrs)

            self.blocks_scanned += 1

            # Progress update every 30 seconds
            if time.time() - last_print > 30:
                elapsed = time.time() - start_time
                rate = self.blocks_scanned / elapsed if elapsed > 0 else 0
                remaining = (current_height - height) / rate if rate > 0 else 0

                print(f"Block {height:,}/{current_height:,} | "
                      f"+{self.new_addresses:,} addrs | "
                      f"{rate:.1f} blk/s | "
                      f"ETA: {remaining/3600:.1f}h")
                last_print = time.time()
                batch_new = 0

            # Save progress periodically
            if height % SAVE_INTERVAL == 0:
                self._save_progress(height)
                # Clear tx cache periodically
                self.tx_cache.clear()

        # Final save
        self._save_progress(current_height)

        elapsed = time.time() - start_time
        print()
        print("=" * 70)
        print("GENESIS SCAN V2 COMPLETE")
        print("=" * 70)
        print(f"Blocks scanned: {self.blocks_scanned:,}")
        print(f"New addresses found: {self.new_addresses:,}")
        print(f"Total addresses: {len(self.addresses):,}")
        print(f"Time: {elapsed/3600:.2f} hours")


def main():
    cluster = GenesisClusterV2()
    cluster.scan()


if __name__ == '__main__':
    main()
