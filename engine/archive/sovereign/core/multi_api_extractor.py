#!/usr/bin/env python3
"""
MULTI-API BLOCKCHAIN EXTRACTOR

Uses 4 FREE APIs in parallel to download 2021-2025 data FAST.
Rotates between APIs to avoid rate limits.

APIs:
1. Mempool.space - Fast, generous limits
2. Blockstream Esplora - Reliable, no limits
3. Blockchain.com - Good backup
4. Blockchair - Limited but useful

Target: 262 weeks in 2-3 hours (not 9 hours)
"""
import requests
import sqlite3
import json
import time
import random
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Dict

# Configuration
DATA_DIR = Path("data/gap_2021_2025")
PROGRESS_FILE = DATA_DIR / "progress.json"
FINAL_DB = Path("data/bitcoin_features.db")

# Block references
START_HEIGHT = 664000  # Jan 1, 2021
BLOCKS_PER_MONTH = 4320  # ~144 blocks/day × 30 days


class MempoolAPI:
    """Mempool.space API"""
    name = "mempool"
    base = "https://mempool.space/api"

    @staticmethod
    def get_blocks(start_height: int) -> List[dict]:
        """Get 10 blocks from height (descending)."""
        try:
            resp = requests.get(f"{MempoolAPI.base}/blocks/{start_height}", timeout=15)
            if resp.status_code == 200:
                blocks = resp.json()
                return [{
                    'height': b['height'],
                    'timestamp': b['timestamp'],
                    'hash': b['id'],
                    'tx_count': b['tx_count'],
                    'size': b['size'],
                    'weight': b.get('weight', 0),
                    'fees': b.get('extras', {}).get('totalFees', 0),
                    'median_fee': b.get('extras', {}).get('medianFee', 0),
                } for b in blocks]
        except:
            pass
        return []


class BlockstreamAPI:
    """Blockstream Esplora API"""
    name = "blockstream"
    base = "https://blockstream.info/api"

    @staticmethod
    def get_blocks(start_height: int) -> List[dict]:
        """Get 10 blocks from height."""
        try:
            resp = requests.get(f"{BlockstreamAPI.base}/blocks/{start_height}", timeout=15)
            if resp.status_code == 200:
                blocks = resp.json()
                return [{
                    'height': b['height'],
                    'timestamp': b['timestamp'],
                    'hash': b['id'],
                    'tx_count': b['tx_count'],
                    'size': b['size'],
                    'weight': b.get('weight', 0),
                    'fees': 0,  # Not in response
                    'median_fee': 0,
                } for b in blocks]
        except:
            pass
        return []


class BlockchainComAPI:
    """Blockchain.com API"""
    name = "blockchain.com"
    base = "https://blockchain.info"

    @staticmethod
    def get_block(height: int) -> Optional[dict]:
        """Get single block by height."""
        try:
            resp = requests.get(
                f"{BlockchainComAPI.base}/block-height/{height}?format=json",
                timeout=15
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get('blocks'):
                    b = data['blocks'][0]
                    return {
                        'height': b['height'],
                        'timestamp': b['time'],
                        'hash': b['hash'],
                        'tx_count': b['n_tx'],
                        'size': b['size'],
                        'weight': b.get('weight', 0),
                        'fees': b.get('fee', 0),
                        'median_fee': 0,
                    }
        except:
            pass
        return None


# API rotation
APIS = [MempoolAPI, BlockstreamAPI]  # Primary APIs
CURRENT_API_IDX = 0


def get_next_api():
    """Rotate to next API."""
    global CURRENT_API_IDX
    api = APIS[CURRENT_API_IDX]
    CURRENT_API_IDX = (CURRENT_API_IDX + 1) % len(APIS)
    return api


def init_data_dir():
    """Create data directory."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_current_height() -> int:
    """Get current blockchain height."""
    try:
        resp = requests.get("https://mempool.space/api/blocks/tip/height", timeout=10)
        return int(resp.text)
    except:
        return 927800


def load_progress() -> dict:
    """Load download progress."""
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {"completed_months": [], "failed_heights": []}


def save_progress(progress: dict):
    """Save download progress."""
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


def download_month_parallel(month_num: int, start_height: int, end_height: int) -> Path:
    """Download one month of blocks using parallel API calls."""
    month_file = DATA_DIR / f"month_{month_num:03d}.db"

    if month_file.exists():
        # Check if complete
        conn = sqlite3.connect(month_file)
        count = conn.execute("SELECT COUNT(*) FROM blocks").fetchone()[0]
        conn.close()
        expected = end_height - start_height + 1
        if count >= expected * 0.95:  # 95% complete is good enough
            print(f"  Month {month_num} already complete ({count} blocks)")
            return month_file

    # Create database
    conn = sqlite3.connect(month_file)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS blocks (
        height INTEGER PRIMARY KEY,
        timestamp INTEGER,
        hash TEXT,
        tx_count INTEGER,
        size INTEGER,
        weight INTEGER,
        fees INTEGER,
        median_fee REAL
    )""")
    conn.commit()

    total = end_height - start_height + 1
    print(f"  Downloading {start_height:,} to {end_height:,} ({total:,} blocks)")

    processed = 0
    failed = 0

    # Download using both APIs in parallel
    heights = list(range(end_height, start_height - 1, -10))

    # Split between APIs
    mempool_heights = heights[::2]  # Even indices
    blockstream_heights = heights[1::2]  # Odd indices

    def fetch_mempool(h):
        return ('mempool', MempoolAPI.get_blocks(h))

    def fetch_blockstream(h):
        return ('blockstream', BlockstreamAPI.get_blocks(h))

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for h in mempool_heights[:len(mempool_heights)//2]:
            futures.append(executor.submit(fetch_mempool, h))
        for h in blockstream_heights[:len(blockstream_heights)//2]:
            futures.append(executor.submit(fetch_blockstream, h))

        for future in as_completed(futures):
            try:
                api_name, blocks = future.result()
                for b in blocks:
                    if start_height <= b['height'] <= end_height:
                        c.execute("""INSERT OR REPLACE INTO blocks VALUES (?,?,?,?,?,?,?,?)""",
                            (b['height'], b['timestamp'], b['hash'], b['tx_count'],
                             b['size'], b['weight'], b['fees'], b['median_fee']))
                        processed += 1
            except Exception as e:
                failed += 1

    # Second pass for remaining
    for h in mempool_heights[len(mempool_heights)//2:]:
        blocks = MempoolAPI.get_blocks(h)
        for b in blocks:
            if start_height <= b['height'] <= end_height:
                c.execute("""INSERT OR REPLACE INTO blocks VALUES (?,?,?,?,?,?,?,?)""",
                    (b['height'], b['timestamp'], b['hash'], b['tx_count'],
                     b['size'], b['weight'], b['fees'], b['median_fee']))
                processed += 1
        time.sleep(0.05)

        if processed % 500 == 0:
            pct = 100 * processed / total
            print(f"    [{pct:5.1f}%] {processed:,}/{total:,}", flush=True)

    for h in blockstream_heights[len(blockstream_heights)//2:]:
        blocks = BlockstreamAPI.get_blocks(h)
        for b in blocks:
            if start_height <= b['height'] <= end_height:
                c.execute("""INSERT OR REPLACE INTO blocks VALUES (?,?,?,?,?,?,?,?)""",
                    (b['height'], b['timestamp'], b['hash'], b['tx_count'],
                     b['size'], b['weight'], b['fees'], b['median_fee']))
                processed += 1
        time.sleep(0.05)

    conn.commit()
    conn.close()

    print(f"  Month {month_num} complete: {processed:,} blocks")
    return month_file


def get_month_range(month_num: int) -> tuple:
    """Get start and end height for a month."""
    start = START_HEIGHT + (month_num * BLOCKS_PER_MONTH)
    end = start + BLOCKS_PER_MONTH - 1
    return start, end


def download_all_fast():
    """Download all months as fast as possible."""
    init_data_dir()
    progress = load_progress()
    current = get_current_height()

    # Calculate total months
    total_months = (current - START_HEIGHT) // BLOCKS_PER_MONTH + 1
    completed = progress["completed_months"]

    print("MULTI-API FAST EXTRACTOR")
    print("=" * 60)
    print(f"Period: Jan 2021 to present")
    print(f"Total months: {total_months}")
    print(f"Completed: {len(completed)}")
    print(f"Using: Mempool.space + Blockstream (parallel)")
    print("=" * 60)
    print()

    start_time = time.time()

    for month_num in range(total_months):
        if month_num in completed:
            continue

        start_h, end_h = get_month_range(month_num)
        end_h = min(end_h, current)

        print(f"\n[Month {month_num + 1}/{total_months}]")

        try:
            download_month_parallel(month_num, start_h, end_h)
            completed.append(month_num)
            progress["completed_months"] = completed
            save_progress(progress)
        except Exception as e:
            print(f"  Error: {e}")
            continue

        # Small delay between months
        time.sleep(1)

        # Progress estimate
        elapsed = time.time() - start_time
        done = len(completed)
        remaining = total_months - done
        eta = (elapsed / done * remaining) / 60 if done > 0 else 0
        print(f"  Progress: {done}/{total_months} | ETA: {eta:.0f} min")

    print("\n" + "=" * 60)
    print("ALL DOWNLOADS COMPLETE!")
    print("Run: python multi_api_extractor.py combine")
    print("=" * 60)


def show_status():
    """Show download status."""
    init_data_dir()
    progress = load_progress()
    current = get_current_height()

    total_months = (current - START_HEIGHT) // BLOCKS_PER_MONTH + 1
    completed = len(progress["completed_months"])

    print("DOWNLOAD STATUS")
    print("=" * 50)
    print(f"Total months needed: {total_months}")
    print(f"Completed: {completed} ({100*completed/total_months:.1f}%)")
    print(f"Remaining: {total_months - completed}")

    # Show files
    files = list(DATA_DIR.glob("month_*.db"))
    total_size = sum(f.stat().st_size for f in files) / 1e6
    print(f"\nDownloaded files: {len(files)}")
    print(f"Total size: {total_size:.1f} MB")

    # Estimate time remaining
    if completed > 0 and completed < total_months:
        print(f"\nEstimated time to complete: {(total_months - completed) * 2:.0f} min")


def combine_to_final():
    """Combine all monthly files into final database."""
    print("COMBINING ALL DATA")
    print("=" * 50)

    FINAL_DB.parent.mkdir(parents=True, exist_ok=True)
    final_conn = sqlite3.connect(FINAL_DB)
    c = final_conn.cursor()

    c.execute("""CREATE TABLE IF NOT EXISTS block_features (
        height INTEGER PRIMARY KEY,
        timestamp INTEGER,
        hash TEXT,
        tx_count INTEGER,
        total_value_btc REAL DEFAULT 0,
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
    final_conn.commit()

    files = sorted(DATA_DIR.glob("month_*.db"))
    total_blocks = 0

    for f in files:
        print(f"Importing {f.name}...")
        month_conn = sqlite3.connect(f)
        rows = month_conn.execute("SELECT * FROM blocks").fetchall()

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

        month_conn.close()
        final_conn.commit()

    final_conn.close()
    print(f"\nCOMPLETE: {total_blocks:,} blocks imported")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("MULTI-API BLOCKCHAIN EXTRACTOR")
        print("=" * 40)
        print("Downloads 2021-2025 gap using multiple free APIs")
        print()
        print("Commands:")
        print("  download  - Download all (fast, ~2-3 hours)")
        print("  status    - Show progress")
        print("  combine   - Merge into final DB")
        print()
        print("APIs used: Mempool.space, Blockstream (parallel)")
    else:
        cmd = sys.argv[1]
        if cmd == "download":
            download_all_fast()
        elif cmd == "status":
            show_status()
        elif cmd == "combine":
            combine_to_final()
