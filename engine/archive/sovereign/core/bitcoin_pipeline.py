#!/usr/bin/env python3
"""
BITCOIN CORE DATA PIPELINE

Real-time and historical feature extraction from Bitcoin Core.
Designed for RenTech-style pattern recognition.

Features extracted per block:
- Transaction metrics (count, value, fees)
- UTXO dynamics (created, destroyed, age)
- Whale activity (large transactions)
- Network metrics (size, weight, fullness)
"""
import json
import subprocess
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Iterator, Dict, List
import time

BITCOIN_CLI = r"C:\Program Files\Bitcoin\daemon\bitcoin-cli.exe"
DB_PATH = Path("data/bitcoin_features.db")


@dataclass
class BlockFeatures:
    """Features extracted from a single block."""
    height: int
    timestamp: int
    hash: str

    # Transaction metrics
    tx_count: int
    total_value_btc: float
    total_fees_btc: float
    avg_fee_rate: float  # sat/vB

    # UTXO metrics
    utxo_created: int
    utxo_destroyed: int
    net_utxo_change: int

    # Whale metrics (>100 BTC)
    whale_tx_count: int
    whale_value_btc: float

    # Large tx metrics (>10 BTC)
    large_tx_count: int
    large_value_btc: float

    # Network metrics
    block_size: int
    block_weight: int
    block_fullness: float  # weight / 4_000_000

    # Miner metrics
    coinbase_value_btc: float
    coinbase_outputs: int


def cli(args: List[str], timeout: int = 120) -> str:
    """Execute bitcoin-cli command."""
    try:
        result = subprocess.run(
            [BITCOIN_CLI] + args,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception as e:
        return ""


def get_block_count() -> int:
    """Get current blockchain height."""
    return int(cli(["getblockcount"]) or "0")


def get_block_hash(height: int) -> str:
    """Get block hash for height."""
    return cli(["getblockhash", str(height)])


def get_block(block_hash: str, verbosity: int = 2) -> dict:
    """Get block with specified verbosity."""
    data = cli(["getblock", block_hash, str(verbosity)], timeout=180)
    return json.loads(data) if data else {}


def extract_block_features(height: int) -> Optional[BlockFeatures]:
    """Extract all features from a block."""
    block_hash = get_block_hash(height)
    if not block_hash:
        return None

    block = get_block(block_hash, verbosity=2)
    if not block:
        return None

    # Initialize counters
    total_value = 0.0
    total_fees = 0.0
    total_vsize = 0
    utxo_created = 0
    utxo_destroyed = 0
    whale_count = 0
    whale_value = 0.0
    large_count = 0
    large_value = 0.0
    coinbase_value = 0.0
    coinbase_outputs = 0

    for tx in block.get("tx", []):
        is_coinbase = "coinbase" in tx.get("vin", [{}])[0]
        tx_value = 0.0
        tx_fee = 0.0
        vsize = tx.get("vsize", 0)
        total_vsize += vsize

        # Count UTXOs
        utxo_created += len(tx.get("vout", []))
        if not is_coinbase:
            utxo_destroyed += len(tx.get("vin", []))

        # Sum outputs
        for vout in tx.get("vout", []):
            value = vout.get("value", 0)
            tx_value += value

            if is_coinbase:
                coinbase_value += value
                coinbase_outputs += 1

        # Calculate fee (inputs - outputs) for non-coinbase
        if not is_coinbase:
            input_value = sum(
                vin.get("prevout", {}).get("value", 0)
                for vin in tx.get("vin", [])
            )
            tx_fee = input_value - tx_value
            total_fees += max(0, tx_fee)

        total_value += tx_value

        # Whale detection (>100 BTC)
        if tx_value > 100:
            whale_count += 1
            whale_value += tx_value

        # Large tx detection (>10 BTC)
        if tx_value > 10:
            large_count += 1
            large_value += tx_value

    # Calculate averages
    avg_fee_rate = (total_fees * 1e8 / total_vsize) if total_vsize > 0 else 0

    return BlockFeatures(
        height=height,
        timestamp=block.get("time", 0),
        hash=block_hash,
        tx_count=len(block.get("tx", [])),
        total_value_btc=total_value,
        total_fees_btc=total_fees,
        avg_fee_rate=avg_fee_rate,
        utxo_created=utxo_created,
        utxo_destroyed=utxo_destroyed,
        net_utxo_change=utxo_created - utxo_destroyed,
        whale_tx_count=whale_count,
        whale_value_btc=whale_value,
        large_tx_count=large_count,
        large_value_btc=large_value,
        block_size=block.get("size", 0),
        block_weight=block.get("weight", 0),
        block_fullness=block.get("weight", 0) / 4_000_000,
        coinbase_value_btc=coinbase_value,
        coinbase_outputs=coinbase_outputs,
    )


def init_database():
    """Initialize the features database."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""CREATE TABLE IF NOT EXISTS block_features (
        height INTEGER PRIMARY KEY,
        timestamp INTEGER,
        hash TEXT,
        tx_count INTEGER,
        total_value_btc REAL,
        total_fees_btc REAL,
        avg_fee_rate REAL,
        utxo_created INTEGER,
        utxo_destroyed INTEGER,
        net_utxo_change INTEGER,
        whale_tx_count INTEGER,
        whale_value_btc REAL,
        large_tx_count INTEGER,
        large_value_btc REAL,
        block_size INTEGER,
        block_weight INTEGER,
        block_fullness REAL,
        coinbase_value_btc REAL,
        coinbase_outputs INTEGER
    )""")

    c.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON block_features(timestamp)")
    conn.commit()
    return conn


def store_features(conn: sqlite3.Connection, features: BlockFeatures):
    """Store block features in database."""
    c = conn.cursor()
    c.execute("""INSERT OR REPLACE INTO block_features VALUES
        (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (features.height, features.timestamp, features.hash,
         features.tx_count, features.total_value_btc, features.total_fees_btc,
         features.avg_fee_rate, features.utxo_created, features.utxo_destroyed,
         features.net_utxo_change, features.whale_tx_count, features.whale_value_btc,
         features.large_tx_count, features.large_value_btc, features.block_size,
         features.block_weight, features.block_fullness, features.coinbase_value_btc,
         features.coinbase_outputs))


def get_last_processed_height(conn: sqlite3.Connection) -> int:
    """Get the last processed block height."""
    c = conn.cursor()
    result = c.execute("SELECT MAX(height) FROM block_features").fetchone()
    return result[0] if result[0] else 0


def extract_range(start_height: int, end_height: int, progress_interval: int = 100):
    """Extract features for a range of blocks."""
    conn = init_database()
    total = end_height - start_height

    print(f"Extracting blocks {start_height:,} to {end_height:,} ({total:,} blocks)")

    start_time = time.time()
    processed = 0

    for height in range(start_height, end_height + 1):
        features = extract_block_features(height)

        if features:
            store_features(conn, features)
            processed += 1

        if processed % progress_interval == 0:
            conn.commit()
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (total - processed) / rate / 60 if rate > 0 else 0
            pct = 100 * processed / total
            print(f"[{pct:5.1f}%] {processed:,}/{total:,} | {rate:.1f} blk/s | ETA: {eta:.0f}m")

    conn.commit()
    conn.close()
    print(f"Complete: {processed:,} blocks processed")


def extract_recent(num_blocks: int = 1000):
    """Extract features for most recent blocks."""
    current = get_block_count()
    start = max(0, current - num_blocks)
    extract_range(start, current)


def continuous_extraction():
    """Continuously extract new blocks as they arrive."""
    conn = init_database()
    last_height = get_last_processed_height(conn)

    print(f"Starting continuous extraction from height {last_height:,}")

    while True:
        current = get_block_count()

        if current > last_height:
            for height in range(last_height + 1, current + 1):
                features = extract_block_features(height)
                if features:
                    store_features(conn, features)
                    print(f"Block {height}: {features.tx_count} txs, "
                          f"{features.whale_tx_count} whales, "
                          f"{features.total_fees_btc:.4f} BTC fees")
                last_height = height

            conn.commit()

        time.sleep(10)  # Check every 10 seconds


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "recent":
            num = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
            extract_recent(num)
        elif sys.argv[1] == "range":
            start = int(sys.argv[2])
            end = int(sys.argv[3])
            extract_range(start, end)
        elif sys.argv[1] == "continuous":
            continuous_extraction()
    else:
        print("Usage:")
        print("  python bitcoin_pipeline.py recent [num_blocks]")
        print("  python bitcoin_pipeline.py range <start> <end>")
        print("  python bitcoin_pipeline.py continuous")
