#!/usr/bin/env python3
"""
WEEKLY BLOCKCHAIN EXTRACTOR

Downloads Bitcoin blockchain data week-by-week to avoid IP/rate limit issues.
- Downloads one week at a time
- Saves progress (resume-safe)
- Combines with ORBITAAL automatically

Usage:
  python weekly_extractor.py download   # Download next pending week
  python weekly_extractor.py status     # Show progress
  python weekly_extractor.py combine    # Merge all into final DB
  python weekly_extractor.py auto       # Auto-download all (with delays)
"""
import requests
import sqlite3
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
DATA_DIR = Path("data/weekly_blocks")
PROGRESS_FILE = DATA_DIR / "progress.json"
FINAL_DB = Path("data/bitcoin_features.db")
API_BASE = "https://mempool.space/api"

# Block height references (approximate)
BLOCKS = {
    "2021-01-01": 664000,
    "2021-07-01": 689000,
    "2022-01-01": 716000,
    "2022-07-01": 743000,
    "2023-01-01": 769000,
    "2023-07-01": 795000,
    "2024-01-01": 822000,
    "2024-07-01": 850000,
    "2025-01-01": 876000,
    "current": None,  # Will be fetched
}

# Weeks to download (2021-01-01 to present)
START_HEIGHT = 664000  # Jan 1, 2021
BLOCKS_PER_WEEK = 1008  # ~144 blocks/day * 7 days


def init_data_dir():
    """Create data directory."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_current_height() -> int:
    """Get current blockchain height."""
    try:
        resp = requests.get(f"{API_BASE}/blocks/tip/height", timeout=10)
        return int(resp.text)
    except:
        return 927000  # Fallback


def load_progress() -> dict:
    """Load download progress."""
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {"completed_weeks": [], "last_height": START_HEIGHT}


def save_progress(progress: dict):
    """Save download progress."""
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


def get_blocks_bulk(start_height: int) -> list:
    """Get 10 blocks from API."""
    try:
        resp = requests.get(f"{API_BASE}/blocks/{start_height}", timeout=15)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        print(f"  API error: {e}")
    return []


def download_week(week_num: int, start_height: int, end_height: int) -> Path:
    """Download one week of blocks to SQLite file."""
    week_file = DATA_DIR / f"week_{week_num:04d}.db"

    if week_file.exists():
        print(f"  Week {week_num} already exists, skipping")
        return week_file

    # Create week database
    conn = sqlite3.connect(week_file)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS blocks (
        height INTEGER PRIMARY KEY,
        timestamp INTEGER,
        hash TEXT,
        tx_count INTEGER,
        size INTEGER,
        weight INTEGER,
        total_fees_sat INTEGER,
        median_fee REAL,
        difficulty REAL
    )""")
    conn.commit()

    print(f"  Downloading blocks {start_height:,} to {end_height:,}")

    processed = 0
    total = end_height - start_height + 1

    # Download in batches of 10 (API returns 10 blocks per call)
    heights = list(range(end_height, start_height - 1, -10))

    for i, h in enumerate(heights):
        blocks = get_blocks_bulk(h)

        for b in blocks:
            if start_height <= b['height'] <= end_height:
                extras = b.get('extras', {})
                c.execute("""INSERT OR REPLACE INTO blocks VALUES (?,?,?,?,?,?,?,?,?)""",
                    (b['height'], b['timestamp'], b['id'], b['tx_count'],
                     b['size'], b['weight'],
                     extras.get('totalFees', 0),
                     extras.get('medianFee', 0),
                     b.get('difficulty', 0)))
                processed += 1

        # Progress every 100 blocks
        if processed % 100 == 0:
            pct = 100 * processed / total
            print(f"    [{pct:5.1f}%] {processed}/{total}", flush=True)

        # Small delay between requests
        time.sleep(0.1)

        conn.commit()

    conn.close()
    print(f"  Week {week_num} complete: {processed} blocks")
    return week_file


def get_week_info(week_num: int) -> tuple:
    """Get start and end height for a week."""
    start = START_HEIGHT + (week_num * BLOCKS_PER_WEEK)
    end = start + BLOCKS_PER_WEEK - 1
    return start, end


def download_next_week():
    """Download the next pending week."""
    init_data_dir()
    progress = load_progress()
    current = get_current_height()

    # Calculate total weeks needed
    total_weeks = (current - START_HEIGHT) // BLOCKS_PER_WEEK + 1
    completed = progress["completed_weeks"]

    print(f"Progress: {len(completed)}/{total_weeks} weeks completed")

    # Find next week to download
    for week_num in range(total_weeks):
        if week_num not in completed:
            start, end = get_week_info(week_num)
            end = min(end, current)

            print(f"\nDownloading Week {week_num} ({start:,} - {end:,})")

            try:
                download_week(week_num, start, end)
                completed.append(week_num)
                progress["completed_weeks"] = completed
                progress["last_height"] = end
                save_progress(progress)
                print(f"Week {week_num} saved!")
                return True
            except Exception as e:
                print(f"Error downloading week {week_num}: {e}")
                return False

    print("All weeks downloaded!")
    return False


def show_status():
    """Show download progress."""
    init_data_dir()
    progress = load_progress()
    current = get_current_height()

    total_weeks = (current - START_HEIGHT) // BLOCKS_PER_WEEK + 1
    completed = len(progress["completed_weeks"])

    print("WEEKLY EXTRACTOR STATUS")
    print("=" * 50)
    print(f"Period: Jan 2021 to present")
    print(f"Block range: {START_HEIGHT:,} to {current:,}")
    print(f"Total blocks: {current - START_HEIGHT:,}")
    print(f"Total weeks: {total_weeks}")
    print(f"Completed: {completed} ({100*completed/total_weeks:.1f}%)")
    print(f"Remaining: {total_weeks - completed}")
    print()

    # Show downloaded files
    files = list(DATA_DIR.glob("week_*.db"))
    total_size = sum(f.stat().st_size for f in files) / 1e6
    print(f"Downloaded files: {len(files)}")
    print(f"Total size: {total_size:.1f} MB")


def combine_all():
    """Combine all weekly files into final database."""
    print("COMBINING ALL DATA")
    print("=" * 50)

    # Initialize final database
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

    c.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON block_features(timestamp)")
    final_conn.commit()

    # Import each weekly file
    files = sorted(DATA_DIR.glob("week_*.db"))
    total_blocks = 0

    for f in files:
        print(f"Importing {f.name}...")
        week_conn = sqlite3.connect(f)
        wc = week_conn.cursor()

        rows = wc.execute("SELECT * FROM blocks").fetchall()

        for r in rows:
            height, timestamp, hash, tx_count, size, weight, fees_sat, median_fee, difficulty = r
            c.execute("""INSERT OR REPLACE INTO block_features
                (height, timestamp, hash, tx_count, total_fees_btc, avg_fee_rate,
                 block_size, block_weight, block_fullness)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (height, timestamp, hash, tx_count,
                 fees_sat / 1e8, median_fee,
                 size, weight, weight / 4_000_000 if weight else 0))
            total_blocks += 1

        week_conn.close()
        final_conn.commit()

    final_conn.close()
    print(f"\nCOMPLETE: {total_blocks:,} blocks imported to {FINAL_DB}")


def auto_download(delay_minutes: int = 5):
    """Automatically download all weeks with delays between."""
    print("AUTO-DOWNLOAD MODE")
    print(f"Delay between weeks: {delay_minutes} minutes")
    print("Press Ctrl+C to stop")
    print()

    while download_next_week():
        print(f"\nWaiting {delay_minutes} minutes before next week...")
        time.sleep(delay_minutes * 60)

    print("\nAll downloads complete! Run 'combine' to merge.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("WEEKLY BLOCKCHAIN EXTRACTOR")
        print("=" * 40)
        print("Commands:")
        print("  download  - Download next pending week")
        print("  status    - Show progress")
        print("  combine   - Merge all weeks into final DB")
        print("  auto      - Auto-download all (with delays)")
        print()
        print("This downloads data week-by-week to avoid IP issues.")
        print("Run 'download' multiple times or use 'auto' mode.")
    else:
        cmd = sys.argv[1]
        if cmd == "download":
            download_next_week()
        elif cmd == "status":
            show_status()
        elif cmd == "combine":
            combine_all()
        elif cmd == "auto":
            delay = int(sys.argv[2]) if len(sys.argv) > 2 else 5
            auto_download(delay)
        else:
            print(f"Unknown command: {cmd}")
