#!/usr/bin/env python3
"""
GENESIS CLUSTER V4 - PARALLEL + REST API
==========================================
Uses REST API + parallel workers for maximum speed
"""

import sqlite3
import subprocess
import json
import time
import os
import requests
from datetime import datetime
from typing import Set, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count

DB_PATH = "/root/sovereign/walletexplorer_addresses.db"
PROGRESS_FILE = "/root/sovereign/genesis_v4_progress.json"
SAVE_INTERVAL = 2000
REST_URL = "http://127.0.0.1:8332/rest"
WORKERS = 8
BATCH_SIZE = 50  # Blocks per batch

# Global address sets (loaded once)
ADDRESSES = set()
ADDR_TO_EX = {}

def load_addresses():
    global ADDRESSES, ADDR_TO_EX
    conn = sqlite3.connect(DB_PATH)
    for row in conn.cursor().execute("SELECT address, exchange FROM addresses"):
        ADDRESSES.add(row[0])
        ADDR_TO_EX[row[0]] = row[1]
    conn.close()
    return len(ADDRESSES)

def get_block_hash(height):
    try:
        r = subprocess.run(
            ['/usr/local/bin/bitcoin-cli', 'getblockhash', str(height)],
            capture_output=True, text=True, timeout=10
        )
        return r.stdout.strip() if r.returncode == 0 else None
    except:
        return None

def get_block_rest(block_hash):
    """Get block via REST API (faster than RPC)."""
    try:
        r = requests.get(f"{REST_URL}/block/{block_hash}.json", timeout=30)
        return r.json() if r.status_code == 200 else None
    except:
        return None

def get_block_rpc(block_hash):
    """Fallback to RPC if REST fails."""
    try:
        r = subprocess.run(
            ['/usr/local/bin/bitcoin-cli', 'getblock', block_hash, '3'],
            capture_output=True, text=True, timeout=60
        )
        return json.loads(r.stdout) if r.returncode == 0 else None
    except:
        return None

def process_block(height):
    """Process single block, return new addresses found."""
    block_hash = get_block_hash(height)
    if not block_hash:
        return []

    block = get_block_rest(block_hash)
    if not block:
        block = get_block_rpc(block_hash)
    if not block:
        return []

    new_addrs = []

    for tx in block.get('tx', []):
        vin = tx.get('vin', [])
        if vin and 'coinbase' in vin[0]:
            continue

        # Get input addresses
        input_addrs = []
        for v in vin:
            # Try prevout first (verbosity 3)
            prevout = v.get('prevout', {})
            if prevout:
                addr = prevout.get('scriptPubKey', {}).get('address')
                if addr:
                    input_addrs.append(addr)

        # Get output addresses
        output_addrs = []
        output_vals = []
        for vout in tx.get('vout', []):
            addr = vout.get('scriptPubKey', {}).get('address')
            val = vout.get('value', 0)
            if addr:
                output_addrs.append(addr)
                output_vals.append(val)

        # Find known exchange
        known_ex = None
        in_inputs = False

        for addr in input_addrs:
            if addr in ADDRESSES:
                known_ex = ADDR_TO_EX.get(addr)
                in_inputs = True
                break

        if not known_ex:
            for addr in output_addrs:
                if addr in ADDRESSES:
                    known_ex = ADDR_TO_EX.get(addr)
                    break

        if not known_ex:
            continue

        # Cluster inputs
        if in_inputs and len(input_addrs) > 1:
            for addr in input_addrs:
                if addr not in ADDRESSES:
                    new_addrs.append((addr, known_ex))

            # Change output
            if len(output_addrs) >= 2 and output_vals:
                min_idx = output_vals.index(min(output_vals))
                change = output_addrs[min_idx]
                if change not in ADDRESSES:
                    new_addrs.append((change, known_ex))

    return new_addrs

def process_batch(heights):
    """Process batch of blocks in parallel."""
    results = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(process_block, h): h for h in heights}
        for f in as_completed(futures):
            try:
                results.extend(f.result())
            except:
                pass
    return results

class GenesisClusterV4:
    def __init__(self):
        print("=" * 70)
        print("GENESIS CLUSTER V4 - PARALLEL MODE")
        print("=" * 70)

        self.new_addresses = 0
        self.start_block = 0

        print("Loading addresses...")
        count = load_addresses()
        print(f"Loaded {count:,} seeds")

        self._load_progress()

    def _load_progress(self):
        if os.path.exists(PROGRESS_FILE):
            with open(PROGRESS_FILE) as f:
                data = json.load(f)
                self.start_block = data.get('last_block', 0) + 1
                self.new_addresses = data.get('new_addresses', 0)
                print(f"Resuming from block {self.start_block:,}")

    def _save_progress(self, h):
        with open(PROGRESS_FILE, 'w') as f:
            json.dump({'last_block': h, 'new_addresses': self.new_addresses}, f)

    def _save_addresses(self, addrs):
        if not addrs:
            return 0
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        now = datetime.now().isoformat()
        saved = 0
        for addr, ex in addrs:
            if addr not in ADDRESSES:
                try:
                    c.execute("INSERT OR IGNORE INTO addresses VALUES (?,?,0,?)", (addr, ex, now))
                    if c.rowcount > 0:
                        saved += 1
                        ADDRESSES.add(addr)
                        ADDR_TO_EX[addr] = ex
                except:
                    pass
        conn.commit()
        conn.close()
        return saved

    def scan(self):
        r = subprocess.run(['/usr/local/bin/bitcoin-cli', 'getblockcount'],
                          capture_output=True, text=True)
        top = int(r.stdout.strip())

        print(f"Scanning {self.start_block:,} to {top:,} ({WORKERS} workers)")
        print()

        t0 = time.time()
        last_print = t0
        blocks_done = 0

        h = self.start_block
        while h <= top:
            # Create batch
            batch = list(range(h, min(h + BATCH_SIZE, top + 1)))

            # Process in parallel
            new_addrs = process_batch(batch)
            saved = self._save_addresses(new_addrs)
            self.new_addresses += saved

            blocks_done += len(batch)
            h += len(batch)

            if time.time() - last_print > 20:
                elapsed = time.time() - t0
                rate = blocks_done / elapsed
                eta = (top - h) / rate / 3600 if rate > 0 else 0
                print(f"Block {h:,}/{top:,} | +{self.new_addresses:,} | {rate:.0f} blk/s | ETA: {eta:.1f}h")
                last_print = time.time()

            if h % SAVE_INTERVAL < BATCH_SIZE:
                self._save_progress(h - 1)

        self._save_progress(top)
        print(f"\nDONE: +{self.new_addresses:,} new addresses")

if __name__ == '__main__':
    GenesisClusterV4().scan()
