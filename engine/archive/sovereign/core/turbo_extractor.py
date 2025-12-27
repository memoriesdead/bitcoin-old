#!/usr/bin/env python3
"""
TURBO BLOCKCHAIN EXTRACTOR - RATE LIMIT AWARE

Uses conservative settings to avoid rate limits.
Alternates between APIs. Gets it done reliably.

Already downloaded: Check progress.json
Remaining: ~250K blocks in ~2-3 hours
"""
import requests
import sqlite3
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple
import random

# Configuration
DATA_DIR = Path("data/gap_2021_2025")
PROGRESS_FILE = DATA_DIR / "turbo_progress.json"
FINAL_DB = Path("data/bitcoin_features.db")

# Block references
START_HEIGHT = 664000  # Jan 1, 2021
BLOCKS_PER_BATCH = 2000  # Smaller batches = less rate limiting


def fetch_mempool(height: int) -> Tuple[str, List[dict]]:
    """Fetch 10 blocks from Mempool.space."""
    try:
        time.sleep(0.1)  # Gentle rate limiting
        resp = requests.get(f"https://mempool.space/api/blocks/{height}", timeout=20)
        if resp.status_code == 200:
            return ('mempool', [{'height': b['height'], 'timestamp': b['timestamp'], 'hash': b['id'],
                     'tx_count': b['tx_count'], 'size': b['size'], 'weight': b.get('weight', 0),
                     'fees': b.get('extras', {}).get('totalFees', 0),
                     'median_fee': b.get('extras', {}).get('medianFee', 0)}
                    for b in resp.json()])
        elif resp.status_code == 429:
            return ('rate_limited', [])
    except:
        pass
    return ('error', [])


def fetch_blockchain_com(height: int) -> Tuple[str, List[dict]]:
    """Fetch block from Blockchain.com (1 at a time)."""
    try:
        time.sleep(0.15)  # Gentle rate limiting
        resp = requests.get(f"https://blockchain.info/block-height/{height}?format=json", timeout=20)
        if resp.status_code == 200:
            data = resp.json()
            blocks = data.get('blocks', [])
            if blocks:
                b = blocks[0]
                return ('blockchain', [{'height': b['height'], 'timestamp': b['time'], 'hash': b['hash'],
                         'tx_count': b['n_tx'], 'size': b['size'], 'weight': b.get('weight', 0),
                         'fees': b.get('fee', 0), 'median_fee': 0}])
        elif resp.status_code == 429:
            return ('rate_limited', [])
    except:
        pass
    return ('error', [])


def get_current_height() -> int:
    """Get current blockchain height."""
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
    return {"completed_batches": [], "blocks_downloaded": 0}


def save_progress(progress: dict):
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


def download_batch(batch_num: int, start_h: int, end_h: int) -> int:
    """Download a batch using both APIs with retry logic."""
    batch_file = DATA_DIR / f"batch_{batch_num:04d}.db"

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
    heights = list(range(end_h, start_h - 1, -10))  # Every 10 blocks for Mempool

    all_blocks = {}  # Use dict to dedupe by height
    start_time = time.time()
    rate_limit_count = 0

    print(f"  Batch {batch_num}: {start_h:,} to {end_h:,} ({total:,} blocks)")

    # First pass: Use Mempool with 3 workers
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(fetch_mempool, h) for h in heights]

        for future in as_completed(futures):
            try:
                status, blocks = future.result()
                if status == 'rate_limited':
                    rate_limit_count += 1
                for b in blocks:
                    if start_h <= b['height'] <= end_h:
                        all_blocks[b['height']] = b
            except:
                pass

    processed = len(all_blocks)
    print(f"    Mempool pass: {processed} blocks (rate limits: {rate_limit_count})")

    # Fill gaps with Blockchain.com if needed
    missing = set(range(start_h, end_h + 1)) - set(all_blocks.keys())
    if missing and len(missing) < total * 0.5:
        print(f"    Filling {len(missing)} gaps with Blockchain.com...")
        missing_list = sorted(list(missing))

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(fetch_blockchain_com, h) for h in missing_list[:100]]

            for future in as_completed(futures):
                try:
                    status, blocks = future.result()
                    for b in blocks:
                        all_blocks[b['height']] = b
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
    final_count = len(all_blocks)
    rate = final_count / elapsed if elapsed > 0 else 0
    coverage = 100 * final_count / total

    print(f"  Batch {batch_num} complete: {final_count:,} blocks ({coverage:.0f}%) in {elapsed:.0f}s ({rate:.0f} blk/s)")
    return final_count


def download_all():
    """Download everything with rate limit awareness."""
    init_data_dir()
    progress = load_progress()
    current = get_current_height()

    total_blocks = current - START_HEIGHT
    total_batches = (total_blocks // BLOCKS_PER_BATCH) + 1
    completed = progress["completed_batches"]

    print("=" * 60)
    print("TURBO BLOCKCHAIN EXTRACTOR (Rate Limit Aware)")
    print("=" * 60)
    print(f"Range: {START_HEIGHT:,} to {current:,} ({total_blocks:,} blocks)")
    print(f"Batches: {total_batches} ({BLOCKS_PER_BATCH:,} blocks each)")
    print(f"Completed: {len(completed)}")
    print(f"APIs: Mempool.space + Blockchain.com (3 workers)")
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
            save_progress(progress)

            # Stats
            elapsed = time.time() - start_time
            rate = total_downloaded / elapsed if elapsed > 0 else 0
            remaining = total_blocks - total_downloaded
            eta = remaining / rate / 60 if rate > 0 else 0

            print(f"\n  Overall: {len(completed)}/{total_batches} batches | "
                  f"{total_downloaded:,}/{total_blocks:,} blocks ({100*total_downloaded/total_blocks:.1f}%) | ETA: {eta:.0f} min\n")

            # Delay between batches to avoid rate limits
            delay = random.uniform(2, 4)
            print(f"  Waiting {delay:.1f}s before next batch...")
            time.sleep(delay)

        except Exception as e:
            print(f"  Error on batch {batch_num}: {e}")
            print("  Waiting 30s before retry...")
            time.sleep(30)

    elapsed = time.time() - start_time
    print("=" * 60)
    print(f"DOWNLOAD COMPLETE!")
    print(f"Total: {total_downloaded:,} blocks in {elapsed/60:.1f} minutes")
    print(f"Rate: {total_downloaded/elapsed:.0f} blocks/second")
    print("=" * 60)
    print("\nRun: python turbo_extractor.py combine")


def show_status():
    """Show download status."""
    init_data_dir()
    progress = load_progress()
    current = get_current_height()

    total_blocks = current - START_HEIGHT
    total_batches = (total_blocks // BLOCKS_PER_BATCH) + 1
    completed = len(progress["completed_batches"])
    downloaded = progress.get("blocks_downloaded", 0)

    print("TURBO EXTRACTOR STATUS")
    print("=" * 50)
    print(f"Total blocks needed: {total_blocks:,}")
    print(f"Downloaded: {downloaded:,} ({100*downloaded/total_blocks:.1f}%)")
    print(f"Batches: {completed}/{total_batches}")

    files = list(DATA_DIR.glob("batch_*.db"))
    total_size = sum(f.stat().st_size for f in files) / 1e6
    print(f"\nBatch files: {len(files)}")
    print(f"Total size: {total_size:.1f} MB")

    # Estimate time
    if completed > 0 and downloaded > 0:
        remaining = total_blocks - downloaded
        avg_blocks_per_batch = downloaded / completed
        remaining_batches = remaining / avg_blocks_per_batch
        # Assume ~60s per batch with delays
        eta_minutes = remaining_batches * 1
        print(f"\nEstimated time remaining: {eta_minutes:.0f} minutes")


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

    files = sorted(DATA_DIR.glob("batch_*.db"))
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
        print("TURBO BLOCKCHAIN EXTRACTOR")
        print("=" * 40)
        print("Commands:")
        print("  download  - Download all blocks")
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
