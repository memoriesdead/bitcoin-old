#!/usr/bin/env python3
"""
LIGHTNING EXTRACT - Fastest possible extraction

Uses Bitcoin Core's JSON-RPC batch mode for 10-20x speedup.
Target: 10-15 minutes.
"""
import json
import sqlite3
import http.client
import base64
from pathlib import Path
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Bitcoin Core RPC config
RPC_USER = "bitcoinrpc"
RPC_PASS = "bitcoinrpc"  # Default, change if needed
RPC_HOST = "127.0.0.1"
RPC_PORT = 8332

# Extraction config
START_BLOCK = 716000  # Jan 2022
BATCH_SIZE = 50  # Blocks per RPC batch
WORKERS = 4  # Parallel connections

stats = {"blocks": 0, "flows": 0}
stats_lock = threading.Lock()


def rpc_batch(requests: list) -> list:
    """Send batch RPC request."""
    try:
        conn = http.client.HTTPConnection(RPC_HOST, RPC_PORT, timeout=300)
        auth = base64.b64encode(f"{RPC_USER}:{RPC_PASS}".encode()).decode()
        headers = {
            "Authorization": f"Basic {auth}",
            "Content-Type": "application/json"
        }

        body = json.dumps(requests)
        conn.request("POST", "/", body, headers)
        response = conn.getresponse()
        data = response.read().decode()
        conn.close()

        return json.loads(data)
    except Exception as e:
        print(f"RPC Error: {e}")
        return []


def rpc_single(method: str, params: list = None) -> dict:
    """Single RPC call."""
    results = rpc_batch([{"jsonrpc": "2.0", "id": 0, "method": method, "params": params or []}])
    return results[0].get("result", {}) if results else {}


def process_batch(block_start: int, block_end: int, exchange_addrs: set, addr_to_exchange: dict) -> list:
    """Process a batch of blocks using batch RPC."""
    flows = []

    # Step 1: Get all block hashes in one batch
    hash_requests = [
        {"jsonrpc": "2.0", "id": i, "method": "getblockhash", "params": [height]}
        for i, height in enumerate(range(block_start, block_end))
    ]

    hash_results = rpc_batch(hash_requests)
    if not hash_results:
        return flows

    block_hashes = {r["id"]: r.get("result", "") for r in hash_results if "result" in r}

    # Step 2: Get all blocks in one batch (verbosity=2 for full tx data)
    block_requests = [
        {"jsonrpc": "2.0", "id": i, "method": "getblock", "params": [hash, 2]}
        for i, hash in block_hashes.items() if hash
    ]

    block_results = rpc_batch(block_requests)
    if not block_results:
        return flows

    # Step 3: Extract exchange flows
    for result in block_results:
        block = result.get("result", {})
        if not block:
            continue

        height = block.get("height", 0)
        block_time = block.get("time", 0)

        for tx in block.get("tx", []):
            txid = tx.get("txid", "")

            # Outputs (inflows)
            for vout in tx.get("vout", []):
                value = vout.get("value", 0)
                if value == 0:
                    continue
                script = vout.get("scriptPubKey", {})
                addr = script.get("address", "")
                if addr in exchange_addrs:
                    exchange = addr_to_exchange.get(addr, "unknown")
                    flows.append((height, block_time, txid, addr, exchange, "inflow", value))

            # Inputs (outflows)
            for vin in tx.get("vin", []):
                prevout = vin.get("prevout", {})
                if not prevout:
                    continue
                value = prevout.get("value", 0)
                script = prevout.get("scriptPubKey", {})
                addr = script.get("address", "")
                if addr in exchange_addrs:
                    exchange = addr_to_exchange.get(addr, "unknown")
                    flows.append((height, block_time, txid, addr, exchange, "outflow", value))

        with stats_lock:
            stats["blocks"] += 1
            stats["flows"] += len([f for f in flows if f[0] == height])

    return flows


def main():
    print("=" * 60)
    print("LIGHTNING EXTRACT - Maximum Speed Mode")
    print("=" * 60)

    # Test RPC connection
    print("[1] Testing Bitcoin Core RPC...")
    info = rpc_single("getblockchaininfo")
    if not info:
        print("[!] Cannot connect to Bitcoin Core RPC")
        print("    Make sure bitcoind is running with server=1")
        print("    And rpcuser/rpcpassword are set in bitcoin.conf")
        return

    current_block = info.get("blocks", 0)
    print(f"    Connected! Current block: {current_block:,}")

    # Load exchange addresses
    print("[2] Loading exchange addresses...")
    with open("data/exchanges.json") as f:
        exchange_data = json.load(f)

    exchange_addrs = set()
    addr_to_exchange = {}
    for exchange, addrs in exchange_data.items():
        for addr in addrs:
            exchange_addrs.add(addr)
            addr_to_exchange[addr] = exchange

    print(f"    {len(exchange_addrs):,} addresses loaded")

    # Setup database
    db_path = Path("data/exchange_flows_2022_2025.db")
    conn = sqlite3.connect(db_path, check_same_thread=False)
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
    conn.commit()

    # Create batches
    total_blocks = current_block - START_BLOCK
    batches = []
    for start in range(START_BLOCK, current_block, BATCH_SIZE):
        end = min(start + BATCH_SIZE, current_block)
        batches.append((start, end))

    print(f"[3] Extracting {total_blocks:,} blocks in {len(batches)} batches...")
    print()

    start_time = time.time()
    all_flows = []
    completed = 0

    # Process batches
    for batch_start, batch_end in batches:
        flows = process_batch(batch_start, batch_end, exchange_addrs, addr_to_exchange)
        all_flows.extend(flows)
        completed += 1

        # Progress update
        if completed % 10 == 0:
            elapsed = time.time() - start_time
            pct = 100 * completed / len(batches)
            rate = stats["blocks"] / elapsed if elapsed > 0 else 0
            eta = (total_blocks - stats["blocks"]) / rate / 60 if rate > 0 else 0
            print(f"[{pct:5.1f}%] {stats['blocks']:,} blocks | {stats['flows']:,} flows | {rate:.1f} blk/s | ETA: {eta:.1f} min")

        # Batch insert
        if len(all_flows) >= 5000:
            c.executemany("INSERT INTO flows VALUES (?,?,?,?,?,?,?)", all_flows)
            conn.commit()
            all_flows = []

    # Final insert
    if all_flows:
        c.executemany("INSERT INTO flows VALUES (?,?,?,?,?,?,?)", all_flows)
        conn.commit()

    # Create indices
    print("\n[4] Creating indices...")
    c.execute("CREATE INDEX idx_time ON flows(block_time)")
    c.execute("CREATE INDEX idx_exchange ON flows(exchange)")
    c.execute("CREATE INDEX idx_type ON flows(flow_type)")
    conn.commit()

    # Stats
    total = c.execute("SELECT COUNT(*) FROM flows").fetchone()[0]
    inflows = c.execute("SELECT COUNT(*) FROM flows WHERE flow_type='inflow'").fetchone()[0]
    outflows = c.execute("SELECT COUNT(*) FROM flows WHERE flow_type='outflow'").fetchone()[0]
    conn.close()

    elapsed = time.time() - start_time
    print()
    print("=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print(f"Time:        {elapsed/60:.1f} minutes")
    print(f"Total flows: {total:,}")
    print(f"  Inflows:   {inflows:,}")
    print(f"  Outflows:  {outflows:,}")
    print(f"Database:    {db_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
