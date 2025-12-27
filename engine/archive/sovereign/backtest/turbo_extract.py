#!/usr/bin/env python3
"""
TURBO EXTRACT - Maximum speed extraction from Bitcoin Core

Extracts exchange flows from 2022-2025 in parallel.
Target: 30-60 minutes instead of 8-12 hours.

Usage:
    python -m engine.sovereign.backtest.turbo_extract
"""
import json
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import threading
import time

# Config
BITCOIN_CLI = r"C:\Program Files\Bitcoin\daemon\bitcoin-cli.exe"
START_BLOCK = 716000  # ~Jan 2022
WORKERS = 8  # Parallel threads
BATCH_SIZE = 100  # Blocks per batch

# Global stats
stats = {"blocks": 0, "flows": 0, "errors": 0}
stats_lock = threading.Lock()


def cli(cmd: list, timeout: int = 120) -> str:
    """Fast CLI call."""
    try:
        result = subprocess.run(
            [BITCOIN_CLI] + cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except:
        return ""


def get_block_count() -> int:
    return int(cli(["getblockcount"]) or "0")


def process_block_range(start: int, end: int, exchange_addrs: set) -> list:
    """Process a range of blocks, return exchange flows."""
    flows = []

    for height in range(start, end):
        try:
            # Get block hash
            block_hash = cli(["getblockhash", str(height)])
            if not block_hash:
                continue

            # Get block with verbosity=2 (includes tx details)
            block_json = cli(["getblock", block_hash, "2"], timeout=180)
            if not block_json:
                continue

            block = json.loads(block_json)
            block_time = block.get("time", 0)

            for tx in block.get("tx", []):
                txid = tx.get("txid", "")

                # Check outputs (inflows)
                for vout in tx.get("vout", []):
                    value = vout.get("value", 0)
                    if value == 0:
                        continue

                    script = vout.get("scriptPubKey", {})
                    addr = script.get("address", "")

                    if addr in exchange_addrs:
                        flows.append((height, block_time, txid, addr, "inflow", value))

                # Check inputs (outflows) - need prevout info
                for vin in tx.get("vin", []):
                    prevout = vin.get("prevout", {})
                    if not prevout:
                        continue

                    value = prevout.get("value", 0)
                    script = prevout.get("scriptPubKey", {})
                    addr = script.get("address", "")

                    if addr in exchange_addrs:
                        flows.append((height, block_time, txid, addr, "outflow", value))

            with stats_lock:
                stats["blocks"] += 1
                stats["flows"] += len([f for f in flows if f[0] == height])

        except Exception as e:
            with stats_lock:
                stats["errors"] += 1

    return flows


def progress_monitor(total_blocks: int):
    """Print progress every 10 seconds."""
    start_time = time.time()
    while True:
        time.sleep(10)
        with stats_lock:
            blocks = stats["blocks"]
            flows = stats["flows"]
            errors = stats["errors"]

        if blocks == 0:
            continue

        elapsed = time.time() - start_time
        rate = blocks / elapsed
        remaining = (total_blocks - blocks) / rate if rate > 0 else 0
        pct = 100 * blocks / total_blocks

        print(f"[{pct:5.1f}%] {blocks:,}/{total_blocks:,} blocks | {flows:,} flows | {rate:.1f} blk/s | ETA: {remaining/60:.1f} min")

        if blocks >= total_blocks:
            break


def main():
    print("=" * 60)
    print("TURBO EXTRACT - Exchange Flows 2022-2025")
    print("=" * 60)

    # Load exchange addresses
    exchange_file = Path("data/exchanges.json")
    if not exchange_file.exists():
        print("[!] data/exchanges.json not found!")
        return

    print("[1] Loading exchange addresses...")
    with open(exchange_file) as f:
        exchange_data = json.load(f)

    # Build fast lookup set
    exchange_addrs = set()
    addr_to_exchange = {}
    for exchange, addrs in exchange_data.items():
        for addr in addrs:
            exchange_addrs.add(addr)
            addr_to_exchange[addr] = exchange

    print(f"    {len(exchange_addrs):,} addresses loaded")

    # Get block range
    current_block = get_block_count()
    total_blocks = current_block - START_BLOCK
    print(f"[2] Blocks to process: {START_BLOCK:,} to {current_block:,} ({total_blocks:,} blocks)")

    # Setup database
    db_path = Path("data/exchange_flows_2022_2025.db")
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute("DROP TABLE IF EXISTS flows")
    c.execute("""CREATE TABLE flows (
        block_height INTEGER,
        block_time INTEGER,
        txid TEXT,
        address TEXT,
        exchange TEXT,
        flow_type TEXT,
        value_btc REAL
    )""")
    c.execute("CREATE INDEX idx_time ON flows(block_time)")
    c.execute("CREATE INDEX idx_exchange ON flows(exchange)")
    conn.commit()

    # Create block batches
    batches = []
    for start in range(START_BLOCK, current_block, BATCH_SIZE):
        end = min(start + BATCH_SIZE, current_block)
        batches.append((start, end))

    print(f"[3] Processing {len(batches)} batches with {WORKERS} workers...")
    print("    (This will take 30-60 minutes)")
    print()

    # Start progress monitor
    monitor_thread = threading.Thread(target=progress_monitor, args=(total_blocks,), daemon=True)
    monitor_thread.start()

    # Process in parallel
    all_flows = []
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {
            executor.submit(process_block_range, start, end, exchange_addrs): (start, end)
            for start, end in batches
        }

        for future in as_completed(futures):
            try:
                flows = future.result()
                all_flows.extend(flows)

                # Batch insert every 10k flows
                if len(all_flows) >= 10000:
                    for flow in all_flows:
                        height, btime, txid, addr, ftype, value = flow
                        exchange = addr_to_exchange.get(addr, "unknown")
                        c.execute("INSERT INTO flows VALUES (?,?,?,?,?,?,?)",
                                 (height, btime, txid, addr, exchange, ftype, value))
                    conn.commit()
                    all_flows = []

            except Exception as e:
                print(f"[!] Batch error: {e}")

    # Insert remaining flows
    for flow in all_flows:
        height, btime, txid, addr, ftype, value = flow
        exchange = addr_to_exchange.get(addr, "unknown")
        c.execute("INSERT INTO flows VALUES (?,?,?,?,?,?,?)",
                 (height, btime, txid, addr, exchange, ftype, value))

    conn.commit()

    # Final stats
    total_flows = c.execute("SELECT COUNT(*) FROM flows").fetchone()[0]
    inflows = c.execute("SELECT COUNT(*) FROM flows WHERE flow_type='inflow'").fetchone()[0]
    outflows = c.execute("SELECT COUNT(*) FROM flows WHERE flow_type='outflow'").fetchone()[0]

    conn.close()

    print()
    print("=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Total flows:  {total_flows:,}")
    print(f"  Inflows:    {inflows:,}")
    print(f"  Outflows:   {outflows:,}")
    print(f"Database:     {db_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
