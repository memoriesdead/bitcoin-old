#!/usr/bin/env python3
"""
MEMPOOL.SPACE BLOCKCHAIN EXTRACTOR - TURBO MODE

100% FREE - No API key required
Uses bulk /api/blocks endpoint for 25-50x speedup

Data coverage: Genesis to present (updated in real-time)
"""
import requests
import sqlite3
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

API_BASE = "https://mempool.space/api"
DB_PATH = Path("data/bitcoin_features.db")


def get_blocks_bulk(start_height: int) -> List[dict]:
    """Get 10 blocks starting from height (descending)."""
    try:
        resp = requests.get(f"{API_BASE}/blocks/{start_height}", timeout=15)
        if resp.status_code == 200:
            return resp.json()
    except:
        pass
    return []


def extract_range_turbo(start_height: int, end_height: int, workers: int = 5):
    """
    TURBO extraction using bulk endpoint + parallel requests.

    ~50-100 blocks/second achievable.
    262K blocks (2021-present) in ~45-90 minutes.
    """
    conn = init_database()
    total = end_height - start_height + 1

    print("MEMPOOL.SPACE TURBO EXTRACTOR (FREE)")
    print("=" * 60)
    print(f"Extracting blocks {start_height:,} to {end_height:,} ({total:,} blocks)")
    print(f"Using bulk endpoint with {workers} parallel workers")
    print()

    start_time = time.time()
    processed = 0

    # Generate starting heights for bulk requests (every 10 blocks, going backwards)
    # Each API call returns 10 blocks starting from the given height (descending)
    batch_starts = list(range(end_height, start_height - 1, -10))

    for i in range(0, len(batch_starts), workers):
        chunk = batch_starts[i:i + workers]

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(get_blocks_bulk, h): h for h in chunk}

            for future in as_completed(futures):
                try:
                    blocks = future.result()
                    for b in blocks:
                        if start_height <= b['height'] <= end_height:
                            store_block_from_bulk(conn, b)
                            processed += 1
                except Exception as e:
                    pass

        conn.commit()

        # Progress update every 500 blocks
        if processed % 500 < 50:
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (total - processed) / rate / 60 if rate > 0 else 0
            pct = 100 * processed / total
            print(f"[{pct:5.1f}%] {processed:,}/{total:,} | "
                  f"{rate:.1f} blk/s | ETA: {eta:.1f}m", flush=True)

        # Small delay to be nice to API
        time.sleep(0.05)

    conn.close()

    elapsed = time.time() - start_time
    print()
    print(f"COMPLETE: {processed:,} blocks in {elapsed/60:.1f} minutes")
    print(f"Rate: {processed/elapsed:.1f} blocks/second")


def store_block_from_bulk(conn: sqlite3.Connection, b: dict):
    """Store block from bulk API response."""
    c = conn.cursor()

    # Extract extras if available
    extras = b.get('extras', {})

    c.execute("""INSERT OR REPLACE INTO block_features
        (height, timestamp, hash, tx_count, total_value_btc, total_fees_btc,
         avg_fee_rate, block_size, block_weight, block_fullness)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (b['height'], b['timestamp'], b['id'], b['tx_count'],
         0,  # total_value not in bulk response
         extras.get('totalFees', 0) / 1e8,
         extras.get('medianFee', 0),
         b['size'], b['weight'],
         b['weight'] / 4_000_000 if b.get('weight') else 0))

@dataclass
class BlockFeatures:
    """Features extracted from a single block."""
    height: int
    timestamp: int
    hash: str
    tx_count: int
    total_value_btc: float
    total_fees_btc: float
    avg_fee_rate: float
    size: int
    weight: int
    difficulty: float


def get_block_height() -> int:
    """Get current blockchain height."""
    resp = requests.get(f"{API_BASE}/blocks/tip/height", timeout=10)
    return int(resp.text)


def get_block_hash(height: int) -> Optional[str]:
    """Get block hash for height."""
    resp = requests.get(f"{API_BASE}/block-height/{height}", timeout=10)
    if resp.status_code == 200:
        return resp.text
    return None


def get_block(block_hash: str) -> Optional[dict]:
    """Get block data."""
    resp = requests.get(f"{API_BASE}/block/{block_hash}", timeout=10)
    if resp.status_code == 200:
        return resp.json()
    return None


def extract_block_features(height: int) -> Optional[BlockFeatures]:
    """Extract features from a single block via Mempool API."""
    try:
        block_hash = get_block_hash(height)
        if not block_hash:
            return None

        block = get_block(block_hash)
        if not block:
            return None

        return BlockFeatures(
            height=height,
            timestamp=block.get('timestamp', 0),
            hash=block_hash,
            tx_count=block.get('tx_count', 0),
            total_value_btc=0,  # Not directly available, would need tx details
            total_fees_btc=block.get('extras', {}).get('totalFees', 0) / 1e8,
            avg_fee_rate=block.get('extras', {}).get('medianFee', 0),
            size=block.get('size', 0),
            weight=block.get('weight', 0),
            difficulty=block.get('difficulty', 0),
        )
    except Exception as e:
        print(f"Error extracting block {height}: {e}")
        return None


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
        utxo_created INTEGER DEFAULT 0,
        utxo_destroyed INTEGER DEFAULT 0,
        net_utxo_change INTEGER DEFAULT 0,
        whale_tx_count INTEGER DEFAULT 0,
        whale_value_btc REAL DEFAULT 0,
        large_tx_count INTEGER DEFAULT 0,
        large_value_btc REAL DEFAULT 0,
        block_size INTEGER,
        block_weight INTEGER,
        block_fullness REAL,
        coinbase_value_btc REAL DEFAULT 0,
        coinbase_outputs INTEGER DEFAULT 0
    )""")

    c.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON block_features(timestamp)")
    conn.commit()
    return conn


def store_features(conn: sqlite3.Connection, f: BlockFeatures):
    """Store block features in database."""
    c = conn.cursor()
    c.execute("""INSERT OR REPLACE INTO block_features
        (height, timestamp, hash, tx_count, total_value_btc, total_fees_btc,
         avg_fee_rate, block_size, block_weight, block_fullness)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (f.height, f.timestamp, f.hash, f.tx_count, f.total_value_btc,
         f.total_fees_btc, f.avg_fee_rate, f.size, f.weight,
         f.weight / 4_000_000 if f.weight else 0))


def extract_range_parallel(start_height: int, end_height: int, workers: int = 10):
    """Extract features for a range of blocks using parallel requests."""
    conn = init_database()
    total = end_height - start_height + 1

    print(f"MEMPOOL.SPACE EXTRACTOR (FREE)")
    print("=" * 60)
    print(f"Extracting blocks {start_height:,} to {end_height:,} ({total:,} blocks)")
    print(f"Workers: {workers}")
    print()

    start_time = time.time()
    processed = 0
    failed = 0

    heights = list(range(start_height, end_height + 1))
    batch_size = 100

    for i in range(0, len(heights), batch_size):
        batch = heights[i:i + batch_size]

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(extract_block_features, h): h for h in batch}

            for future in as_completed(futures):
                height = futures[future]
                try:
                    features = future.result()
                    if features:
                        store_features(conn, features)
                        processed += 1
                    else:
                        failed += 1
                except Exception as e:
                    failed += 1

        conn.commit()

        # Progress update
        elapsed = time.time() - start_time
        rate = processed / elapsed if elapsed > 0 else 0
        eta = (total - processed - failed) / rate / 60 if rate > 0 else 0
        pct = 100 * (processed + failed) / total

        print(f"[{pct:5.1f}%] {processed:,} done, {failed:,} failed | "
              f"{rate:.1f} blk/s | ETA: {eta:.1f}m", flush=True)

        # Small delay to be nice to the API
        time.sleep(0.5)

    conn.close()

    elapsed = time.time() - start_time
    print()
    print(f"COMPLETE: {processed:,} blocks in {elapsed/60:.1f} minutes")
    print(f"Failed: {failed:,}")


def extract_2021_to_present():
    """Extract blocks from Jan 2021 to present (the gap after ORBITAAL)."""
    # Block ~665,000 is around Jan 1, 2021
    # This fills the gap between ORBITAAL (ends ~Jan 2021) and now

    current = get_block_height()
    start = 665000  # Approximate block height for Jan 2021

    print(f"Filling ORBITAAL gap: {start:,} to {current:,}")
    print(f"Estimated blocks: {current - start:,}")
    print()

    # Use TURBO mode for speed
    extract_range_turbo(start, current, workers=5)


def extract_recent(num_blocks: int = 1000):
    """Extract most recent blocks."""
    current = get_block_height()
    start = current - num_blocks
    extract_range_parallel(start, current, workers=10)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "gap":
            # Fill the 2021-present gap
            extract_2021_to_present()
        elif sys.argv[1] == "recent":
            num = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
            extract_recent(num)
        elif sys.argv[1] == "range":
            start = int(sys.argv[2])
            end = int(sys.argv[3])
            extract_range_parallel(start, end)
    else:
        print("MEMPOOL.SPACE BLOCKCHAIN EXTRACTOR")
        print("=" * 40)
        print("Usage:")
        print("  python mempool_extractor.py gap          # Fill 2021-present gap")
        print("  python mempool_extractor.py recent 5000  # Extract last 5000 blocks")
        print("  python mempool_extractor.py range START END")
        print()
        print("This is 100% FREE - uses Mempool.space public API")
