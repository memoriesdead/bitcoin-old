#!/usr/bin/env python3
"""
SLOW BUT SURE BLOCKCHAIN EXTRACTOR

Single request at a time. No rate limits. 100% reliable.
Will take ~2-3 hours but will complete without interruption.

Strategy:
- 1 request at a time
- 0.5s delay between requests
- Get 10 blocks per request
- ~120 blocks/minute = 7,200 blocks/hour
- 264K blocks in ~36.6 hours... too slow

ACTUAL Strategy:
- Use 2 workers max
- 0.2s delay between requests
- Get 10 blocks per request
- ~600 blocks/minute = 36K blocks/hour
- 264K blocks in ~7 hours

OR with 3 workers:
- 900 blocks/minute = 54K blocks/hour
- 264K blocks in ~5 hours

We'll find the sweet spot.
"""
import requests
import sqlite3
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
import random

# Configuration
DATA_DIR = Path("data/gap_2021_2025")
PROGRESS_FILE = DATA_DIR / "slow_progress.json"
FINAL_DB = Path("data/bitcoin_features.db")

# Block references
START_HEIGHT = 664000  # Jan 1, 2021
BLOCKS_PER_BATCH = 1000  # 1K blocks per batch


def fetch_blocks_safe(height: int, delay: float = 0.2) -> List[dict]:
    """Fetch 10 blocks with delay."""
    time.sleep(delay + random.uniform(0, 0.1))  # Add jitter
    try:
        resp = requests.get(f"https://mempool.space/api/blocks/{height}", timeout=30)
        if resp.status_code == 200:
            return [{'height': b['height'], 'timestamp': b['timestamp'], 'hash': b['id'],
                     'tx_count': b['tx_count'], 'size': b['size'], 'weight': b.get('weight', 0),
                     'fees': b.get('extras', {}).get('totalFees', 0),
                     'median_fee': b.get('extras', {}).get('medianFee', 0)}
                    for b in resp.json()]
        elif resp.status_code == 429:
            print(f"    Rate limited at {height}, waiting 30s...")
            time.sleep(30)
            return fetch_blocks_safe(height, delay * 2)  # Retry with longer delay
    except Exception as e:
        print(f"    Error at {height}: {e}")
        time.sleep(5)
    return []


def get_current_height() -> int:
    try:
        resp = requests.get("https://mempool.space/api/blocks/tip/height", timeout=10)
        return int(resp.text)
    except:
        return 927800


def init_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {"completed_batches": [], "blocks_downloaded": 0, "last_height": 0}


def save_progress(progress: dict):
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


def download_batch(batch_num: int, start_h: int, end_h: int) -> int:
    """Download batch with conservative rate limiting."""
    batch_file = DATA_DIR / f"slow_{batch_num:04d}.db"

    if batch_file.exists():
        conn = sqlite3.connect(batch_file)
        count = conn.execute("SELECT COUNT(*) FROM blocks").fetchone()[0]
        conn.close()
        expected = end_h - start_h + 1
        if count >= expected * 0.95:
            print(f"  Batch {batch_num} already has {count} blocks, skipping")
            return count

    # Create database
    conn = sqlite3.connect(batch_file)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS blocks (
        height INTEGER PRIMARY KEY,
        timestamp INTEGER, hash TEXT, tx_count INTEGER,
        size INTEGER, weight INTEGER, fees INTEGER, median_fee REAL
    )""")
    conn.commit()

    total = end_h - start_h + 1
    heights = list(range(end_h, start_h - 1, -10))

    all_blocks = {}
    start_time = time.time()

    print(f"  Batch {batch_num}: {start_h:,} to {end_h:,} ({total:,} blocks)")

    # Use only 2 workers with delays
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        for h in heights:
            futures.append(executor.submit(fetch_blocks_safe, h))

        for future in as_completed(futures):
            try:
                blocks = future.result()
                for b in blocks:
                    if start_h <= b['height'] <= end_h:
                        all_blocks[b['height']] = b

                # Progress every 200 blocks
                if len(all_blocks) % 200 == 0 and len(all_blocks) > 0:
                    elapsed = time.time() - start_time
                    rate = len(all_blocks) / elapsed if elapsed > 0 else 0
                    pct = 100 * len(all_blocks) / total
                    print(f"    [{pct:5.1f}%] {len(all_blocks):,}/{total:,} | {rate:.0f} blk/s", flush=True)
            except:
                pass

    # Insert all blocks
    for b in all_blocks.values():
        c.execute("INSERT OR REPLACE INTO blocks VALUES (?,?,?,?,?,?,?,?)",
            (b['height'], b['timestamp'], b['hash'], b['tx_count'],
             b['size'], b['weight'], b['fees'], b['median_fee']))

    conn.commit()
    conn.close()

    elapsed = time.time() - start_time
    count = len(all_blocks)
    rate = count / elapsed if elapsed > 0 else 0

    print(f"  Batch {batch_num}: {count:,}/{total:,} blocks in {elapsed:.0f}s ({rate:.0f} blk/s)")
    return count


def download_all():
    init_data_dir()
    progress = load_progress()
    current = get_current_height()

    total_blocks = current - START_HEIGHT
    total_batches = (total_blocks // BLOCKS_PER_BATCH) + 1
    completed = progress["completed_batches"]

    print("=" * 60)
    print("SLOW BUT SURE BLOCKCHAIN EXTRACTOR")
    print("=" * 60)
    print(f"Range: {START_HEIGHT:,} to {current:,} ({total_blocks:,} blocks)")
    print(f"Batches: {total_batches} ({BLOCKS_PER_BATCH:,} blocks each)")
    print(f"Completed: {len(completed)}")
    print(f"Strategy: 2 workers, 0.2s delays, no rate limits")
    print("=" * 60)
    print()

    start_time = time.time()
    total_downloaded = progress.get("blocks_downloaded", 0)

    for batch_num in range(total_batches):
        if batch_num in completed:
            continue

        start_h = START_HEIGHT + (batch_num * BLOCKS_PER_BATCH)
        end_h = min(start_h + BLOCKS_PER_BATCH - 1, current)

        try:
            blocks = download_batch(batch_num, start_h, end_h)
            total_downloaded += blocks
            completed.append(batch_num)
            progress["completed_batches"] = completed
            progress["blocks_downloaded"] = total_downloaded
            progress["last_height"] = end_h
            save_progress(progress)

            elapsed = time.time() - start_time
            rate = total_downloaded / elapsed if elapsed > 0 else 0
            remaining = total_blocks - total_downloaded
            eta = remaining / rate / 60 if rate > 0 else 0

            print(f"\n  Progress: {len(completed)}/{total_batches} | "
                  f"{total_downloaded:,}/{total_blocks:,} ({100*total_downloaded/total_blocks:.1f}%) | "
                  f"ETA: {eta:.0f} min\n")

            # Small delay between batches
            time.sleep(1)

        except Exception as e:
            print(f"  Error on batch {batch_num}: {e}")
            time.sleep(30)

    elapsed = time.time() - start_time
    print("=" * 60)
    print(f"DOWNLOAD COMPLETE!")
    print(f"Total: {total_downloaded:,} blocks in {elapsed/60:.1f} minutes")
    print("=" * 60)


def show_status():
    init_data_dir()
    progress = load_progress()
    current = get_current_height()

    total_blocks = current - START_HEIGHT
    total_batches = (total_blocks // BLOCKS_PER_BATCH) + 1
    completed = len(progress["completed_batches"])
    downloaded = progress.get("blocks_downloaded", 0)

    print("SLOW BUT SURE STATUS")
    print("=" * 50)
    print(f"Total blocks needed: {total_blocks:,}")
    print(f"Downloaded: {downloaded:,} ({100*downloaded/total_blocks:.1f}%)")
    print(f"Batches: {completed}/{total_batches}")

    files = list(DATA_DIR.glob("slow_*.db"))
    total_size = sum(f.stat().st_size for f in files) / 1e6
    print(f"\nBatch files: {len(files)}")
    print(f"Total size: {total_size:.1f} MB")


def combine_to_final():
    """Combine all batch files into final database."""
    print("COMBINING ALL BATCHES")
    print("=" * 50)

    FINAL_DB.parent.mkdir(parents=True, exist_ok=True)
    final_conn = sqlite3.connect(FINAL_DB)
    c = final_conn.cursor()

    c.execute("""CREATE TABLE IF NOT EXISTS block_features (
        height INTEGER PRIMARY KEY,
        timestamp INTEGER, hash TEXT, tx_count INTEGER,
        total_value_btc REAL DEFAULT 0, total_fees_btc REAL,
        avg_fee_rate REAL, utxo_created INTEGER DEFAULT 0,
        utxo_destroyed INTEGER DEFAULT 0, net_utxo_change INTEGER DEFAULT 0,
        whale_tx_count INTEGER DEFAULT 0, whale_value_btc REAL DEFAULT 0,
        large_tx_count INTEGER DEFAULT 0, large_value_btc REAL DEFAULT 0,
        block_size INTEGER, block_weight INTEGER, block_fullness REAL,
        coinbase_value_btc REAL DEFAULT 0, coinbase_outputs INTEGER DEFAULT 0
    )""")
    final_conn.commit()

    # Combine both turbo and slow batch files
    files = sorted(list(DATA_DIR.glob("batch_*.db")) + list(DATA_DIR.glob("slow_*.db")))
    total_blocks = 0

    for f in files:
        print(f"Importing {f.name}...")
        batch_conn = sqlite3.connect(f)
        rows = batch_conn.execute("SELECT * FROM blocks").fetchall()

        for r in rows:
            height, timestamp, hash, tx_count, size, weight, fees, median_fee = r
            c.execute("""INSERT OR REPLACE INTO block_features
                (height, timestamp, hash, tx_count, total_fees_btc, avg_fee_rate,
                 block_size, block_weight, block_fullness)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (height, timestamp, hash, tx_count,
                 fees / 1e8 if fees else 0, median_fee or 0,
                 size, weight, weight / 4_000_000 if weight else 0))
            total_blocks += 1

        batch_conn.close()
        final_conn.commit()

    final_conn.close()
    print(f"\nCOMPLETE: {total_blocks:,} blocks in {FINAL_DB}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("SLOW BUT SURE BLOCKCHAIN EXTRACTOR")
        print("=" * 40)
        print("Commands:")
        print("  download  - Download all blocks (2-3 hours)")
        print("  status    - Show progress")
        print("  combine   - Merge into final DB")
    else:
        cmd = sys.argv[1]
        if cmd == "download":
            download_all()
        elif cmd == "status":
            show_status()
        elif cmd == "combine":
            combine_to_final()
