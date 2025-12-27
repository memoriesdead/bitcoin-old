#!/usr/bin/env python3
"""
Extract 2022-2025 Bitcoin data from local node.

Fills the gap between ORBITAAL (ends 2021) and present.
Extracts transaction flows to/from known exchange addresses.

Usage:
    python -m engine.sovereign.backtest.extract_node_data
"""
import json
import sqlite3
import subprocess
from datetime import datetime
from pathlib import Path

# Bitcoin Core RPC
BITCOIN_CLI = r"C:\Program Files\Bitcoin\daemon\bitcoin-cli.exe"

# Block ranges for 2022-2025
# Block 716,000 ~ Jan 1, 2022
# Block 927,785 ~ Dec 13, 2025 (current)
START_BLOCK = 716000  # ~Jan 2022
END_BLOCK = None  # Current tip


def bitcoin_cli(cmd: list) -> dict:
    """Execute bitcoin-cli command and return JSON result."""
    try:
        result = subprocess.run(
            [BITCOIN_CLI] + cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode == 0:
            return json.loads(result.stdout) if result.stdout.strip() else {}
        else:
            print(f"Error: {result.stderr}")
            return {}
    except Exception as e:
        print(f"CLI Error: {e}")
        return {}


def get_block_count() -> int:
    """Get current block height."""
    result = subprocess.run(
        [BITCOIN_CLI, "getblockcount"],
        capture_output=True, text=True
    )
    return int(result.stdout.strip())


def get_block_hash(height: int) -> str:
    """Get block hash for height."""
    result = subprocess.run(
        [BITCOIN_CLI, "getblockhash", str(height)],
        capture_output=True, text=True
    )
    return result.stdout.strip()


def get_block(block_hash: str, verbosity: int = 2) -> dict:
    """Get block with transactions."""
    return bitcoin_cli(["getblock", block_hash, str(verbosity)])


def extract_exchange_flows(start_block: int, end_block: int, db_path: Path):
    """Extract flows to/from exchanges for date range."""

    # Load exchange addresses
    exchange_file = Path("data/exchanges.json")
    if not exchange_file.exists():
        print("[!] exchanges.json not found")
        return

    with open(exchange_file) as f:
        exchange_data = json.load(f)

    # Build address lookup
    exchange_addrs = {}
    for exchange, addrs in exchange_data.items():
        for addr in addrs:
            exchange_addrs[addr] = exchange

    print(f"[+] Loaded {len(exchange_addrs):,} exchange addresses")

    # Setup database
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS exchange_flows (
        block_height INTEGER,
        block_time INTEGER,
        txid TEXT,
        exchange TEXT,
        flow_type TEXT,  -- 'inflow' or 'outflow'
        value_btc REAL,
        address TEXT
    )''')

    c.execute('''CREATE INDEX IF NOT EXISTS idx_flows_time
                 ON exchange_flows(block_time)''')
    c.execute('''CREATE INDEX IF NOT EXISTS idx_flows_exchange
                 ON exchange_flows(exchange)''')

    conn.commit()

    print(f"[+] Scanning blocks {start_block:,} to {end_block:,}")
    print(f"[+] This will take a while...")

    total_flows = 0

    for height in range(start_block, end_block + 1):
        if height % 1000 == 0:
            print(f"    Block {height:,} / {end_block:,} ({100*height/end_block:.1f}%) - {total_flows:,} flows found")
            conn.commit()

        try:
            block_hash = get_block_hash(height)
            block = get_block(block_hash, verbosity=2)

            if not block:
                continue

            block_time = block.get('time', 0)

            for tx in block.get('tx', []):
                txid = tx.get('txid', '')

                # Check outputs (inflows to exchanges)
                for vout in tx.get('vout', []):
                    value = vout.get('value', 0)
                    script = vout.get('scriptPubKey', {})
                    addresses = script.get('addresses', [])

                    # Also check 'address' field (newer format)
                    if not addresses and 'address' in script:
                        addresses = [script['address']]

                    for addr in addresses:
                        if addr in exchange_addrs:
                            exchange = exchange_addrs[addr]
                            c.execute('''INSERT INTO exchange_flows
                                        VALUES (?,?,?,?,?,?,?)''',
                                     (height, block_time, txid, exchange,
                                      'inflow', value, addr))
                            total_flows += 1

                # Check inputs (outflows from exchanges)
                for vin in tx.get('vin', []):
                    if 'prevout' in vin:
                        prevout = vin['prevout']
                        value = prevout.get('value', 0)
                        script = prevout.get('scriptPubKey', {})
                        addresses = script.get('addresses', [])

                        if not addresses and 'address' in script:
                            addresses = [script['address']]

                        for addr in addresses:
                            if addr in exchange_addrs:
                                exchange = exchange_addrs[addr]
                                c.execute('''INSERT INTO exchange_flows
                                            VALUES (?,?,?,?,?,?,?)''',
                                         (height, block_time, txid, exchange,
                                          'outflow', value, addr))
                                total_flows += 1

        except Exception as e:
            if height % 1000 == 0:
                print(f"    [!] Block {height} error: {e}")
            continue

    conn.commit()
    conn.close()

    print(f"\n[+] COMPLETE: {total_flows:,} exchange flows extracted")
    print(f"[+] Saved to: {db_path}")


def main():
    print("=" * 60)
    print("BITCOIN CORE DATA EXTRACTOR")
    print("Filling 2022-2025 gap from local node")
    print("=" * 60)

    # Get current block height
    current = get_block_count()
    print(f"[+] Current block: {current:,}")

    end_block = END_BLOCK or current

    db_path = Path("data/node_flows_2022_2025.db")

    extract_exchange_flows(START_BLOCK, end_block, db_path)


if __name__ == '__main__':
    main()
