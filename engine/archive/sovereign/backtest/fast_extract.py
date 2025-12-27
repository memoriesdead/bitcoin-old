#!/usr/bin/env python3
"""
FAST EXTRACT - Optimized bitcoin-cli extraction

Uses subprocess with bitcoin-cli but optimized for speed.
"""
import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import multiprocessing

BITCOIN_CLI = r"C:\Program Files\Bitcoin\daemon\bitcoin-cli.exe"
START_BLOCK = 716000
WORKERS = multiprocessing.cpu_count()


def cli_raw(args: list, timeout: int = 60) -> str:
    """Raw CLI call returning string."""
    try:
        result = subprocess.run(
            [BITCOIN_CLI] + args,
            capture_output=True, text=True, timeout=timeout
        )
        return result.stdout.strip()
    except:
        return ""


def process_blocks(args):
    """Process a range of blocks. Called by worker process."""
    start, end, exchange_addrs_list = args
    exchange_addrs = set(exchange_addrs_list)
    flows = []

    for height in range(start, end):
        try:
            # Get block hash
            bhash = cli_raw(["getblockhash", str(height)])
            if not bhash:
                continue

            # Get block with full tx (verbosity 2)
            block_str = cli_raw(["getblock", bhash, "2"], timeout=120)
            if not block_str:
                continue

            block = json.loads(block_str)
            btime = block.get("time", 0)

            for tx in block.get("tx", []):
                txid = tx.get("txid", "")

                # Outputs
                for vout in tx.get("vout", []):
                    val = vout.get("value", 0)
                    if val == 0:
                        continue
                    script = vout.get("scriptPubKey", {})
                    addr = script.get("address", "")
                    if addr in exchange_addrs:
                        flows.append((height, btime, txid, addr, "inflow", val))

                # Inputs
                for vin in tx.get("vin", []):
                    prev = vin.get("prevout", {})
                    if not prev:
                        continue
                    val = prev.get("value", 0)
                    script = prev.get("scriptPubKey", {})
                    addr = script.get("address", "")
                    if addr in exchange_addrs:
                        flows.append((height, btime, txid, addr, "outflow", val))

        except Exception as e:
            continue

    return flows, end - start


def main():
    print("=" * 60)
    print(f"FAST EXTRACT - {WORKERS} CPU cores")
    print("=" * 60)

    # Get current block
    current = int(cli_raw(["getblockcount"]))
    total_blocks = current - START_BLOCK
    print(f"[1] Blocks: {START_BLOCK:,} to {current:,} ({total_blocks:,} total)")

    # Load exchanges
    print("[2] Loading exchanges...")
    with open("data/exchanges.json") as f:
        ex_data = json.load(f)

    exchange_addrs = set()
    addr_to_ex = {}
    for ex, addrs in ex_data.items():
        for a in addrs:
            exchange_addrs.add(a)
            addr_to_ex[a] = ex

    print(f"    {len(exchange_addrs):,} addresses")

    # Prepare batches - smaller for more parallelism
    BATCH = 20
    batches = []
    addr_list = list(exchange_addrs)  # Convert to list for pickling

    for s in range(START_BLOCK, current, BATCH):
        e = min(s + BATCH, current)
        batches.append((s, e, addr_list))

    print(f"[3] Processing {len(batches)} batches with {WORKERS} workers...")
    print(f"    Estimated time: 15-30 minutes")
    print()

    # Setup DB
    db_path = Path("data/exchange_flows_2022_2025.db")
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("DROP TABLE IF EXISTS flows")
    c.execute("""CREATE TABLE flows (
        block_height INTEGER, block_time INTEGER, txid TEXT,
        address TEXT, exchange TEXT, flow_type TEXT, value_btc REAL
    )""")
    conn.commit()

    # Process in parallel
    start_time = time.time()
    done_blocks = 0
    total_flows = 0

    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(process_blocks, b): b for b in batches}

        for future in as_completed(futures):
            try:
                flows, count = future.result()
                done_blocks += count
                total_flows += len(flows)

                # Insert flows with exchange name
                for f in flows:
                    height, btime, txid, addr, ftype, val = f
                    ex = addr_to_ex.get(addr, "unknown")
                    c.execute("INSERT INTO flows VALUES (?,?,?,?,?,?,?)",
                             (height, btime, txid, addr, ex, ftype, val))

                # Progress
                elapsed = time.time() - start_time
                pct = 100 * done_blocks / total_blocks
                rate = done_blocks / elapsed if elapsed > 0 else 0
                eta = (total_blocks - done_blocks) / rate / 60 if rate > 0 else 0

                if done_blocks % 500 == 0:
                    print(f"[{pct:5.1f}%] {done_blocks:,}/{total_blocks:,} | {total_flows:,} flows | {rate:.1f} blk/s | ETA: {eta:.0f}m")
                    conn.commit()

            except Exception as e:
                print(f"Error: {e}")

    conn.commit()

    # Indices
    print("\n[4] Creating indices...")
    c.execute("CREATE INDEX idx_time ON flows(block_time)")
    c.execute("CREATE INDEX idx_ex ON flows(exchange)")
    conn.commit()

    # Final stats
    total = c.execute("SELECT COUNT(*) FROM flows").fetchone()[0]
    conn.close()

    elapsed = time.time() - start_time
    print()
    print("=" * 60)
    print(f"DONE in {elapsed/60:.1f} minutes!")
    print(f"Total flows: {total:,}")
    print(f"Database: {db_path}")
    print("=" * 60)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
