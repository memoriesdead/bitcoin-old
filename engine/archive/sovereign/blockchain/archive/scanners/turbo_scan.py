#!/usr/bin/env python3
"""
TURBO SCAN - SIMPLE SEQUENTIAL THAT WORKS
==========================================
No threading complexity. Just loops through blocks.
"""

import sys
import time
import json
import sqlite3
import subprocess
from datetime import datetime

sys.path.insert(0, '/root/sovereign')
sys.path.insert(0, '/root/sovereign/blockchain')


def rpc(method, *params):
    """Bitcoin RPC call."""
    cmd = ['bitcoin-cli', method] + [str(p) for p in params]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        return None
    try:
        return json.loads(r.stdout)
    except:
        return r.stdout.strip()


def main():
    print("=" * 70)
    print("TURBO SCAN - SEQUENTIAL ADDRESS DISCOVERY")
    print("=" * 70)

    # Load seeds
    try:
        from address_collector import KNOWN_COLD_WALLETS
    except:
        from blockchain.address_collector import KNOWN_COLD_WALLETS

    addresses = set()
    addr_to_ex = {}

    db_path = "/root/sovereign/address_clusters.db"
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Create table
    c.execute("""
        CREATE TABLE IF NOT EXISTS addresses (
            address TEXT PRIMARY KEY,
            exchange TEXT NOT NULL,
            discovered_at TEXT,
            source TEXT
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_ex ON addresses(exchange)")

    # Load seeds
    for ex, addrs in KNOWN_COLD_WALLETS.items():
        for addr in addrs:
            addresses.add(addr)
            addr_to_ex[addr] = ex
            c.execute("INSERT OR IGNORE INTO addresses VALUES (?, ?, ?, 'seed')",
                      (addr, ex, datetime.now().isoformat()))

    # Load existing
    c.execute("SELECT address, exchange FROM addresses")
    for row in c.fetchall():
        addresses.add(row[0])
        addr_to_ex[row[0]] = row[1]

    conn.commit()

    height = int(rpc('getblockcount'))
    print(f"Height: {height:,}")
    print(f"Starting addresses: {len(addresses):,}")

    # Parse args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--blocks', type=int, default=1000)
    args = parser.parse_args()

    start = height - args.blocks
    total = args.blocks
    discovered = 0
    txs = 0
    blocks_done = 0
    start_time = time.time()

    print(f"Scanning {total:,} blocks from {start:,} to {height:,}...")
    print()

    for h in range(start, height + 1):
        try:
            block_hash = rpc('getblockhash', h)
            if not block_hash:
                blocks_done += 1
                continue

            block = rpc('getblock', block_hash, 2)
            if not block:
                blocks_done += 1
                continue

            batch = []

            for tx in block.get('tx', []):
                txs += 1

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
                    if addr in addr_to_ex:
                        known_ex = addr_to_ex[addr]
                        break

                if not known_ex:
                    continue

                # Cluster all inputs
                for addr in input_addrs:
                    if addr not in addresses:
                        addresses.add(addr)
                        addr_to_ex[addr] = known_ex
                        batch.append((addr, known_ex, datetime.now().isoformat(), 'scan'))
                        discovered += 1

            # Batch insert
            if batch:
                c.executemany("INSERT OR IGNORE INTO addresses VALUES (?, ?, ?, ?)", batch)
                conn.commit()

            blocks_done += 1

            # Progress every block
            elapsed = time.time() - start_time
            rate = blocks_done / max(elapsed, 1)
            remaining = total - blocks_done
            eta = remaining / max(rate, 0.01)

            print(f"\r[{blocks_done:,}/{total:,}] {rate:.2f} blk/s | "
                  f"TXs: {txs:,} | Addrs: {len(addresses):,} (+{discovered:,}) | "
                  f"ETA: {eta/60:.1f}m", end='', flush=True)

        except Exception as e:
            blocks_done += 1
            print(f"\n[ERR] Block {h}: {e}")
            continue

    conn.close()

    print()
    print()
    print("=" * 70)
    print(f"COMPLETE: {discovered:,} new addresses discovered")
    print(f"Total addresses: {len(addresses):,}")
    print("=" * 70)

    # Per exchange counts
    counts = {}
    for a, e in addr_to_ex.items():
        counts[e] = counts.get(e, 0) + 1
    for e, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {e:<20} {n:>10,}")


if __name__ == '__main__':
    main()
