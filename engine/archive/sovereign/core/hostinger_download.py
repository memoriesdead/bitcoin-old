#!/usr/bin/env python3
"""
HOSTINGER PARALLEL DOWNLOADER

Deploy this on your Hostinger Linux VPS.
Uses multiple IPs/VPNs to download 2021-2025 blockchain data FAST.

SETUP ON HOSTINGER:
1. SSH into your Hostinger VPS
2. Install requirements: pip3 install requests
3. Copy this script to the server
4. Run multiple instances with different IP ranges

STRATEGY:
- Split 264K blocks into chunks
- Run different chunks on different VPN connections
- Each VPN = different IP = no rate limits
- Combine all chunks at the end

USAGE:
  # Terminal 1 (VPN1): Download blocks 664000-730000
  python3 hostinger_download.py 664000 730000 chunk1

  # Terminal 2 (VPN2): Download blocks 730000-796000
  python3 hostinger_download.py 730000 796000 chunk2

  # Terminal 3 (VPN3): Download blocks 796000-862000
  python3 hostinger_download.py 796000 862000 chunk3

  # Terminal 4 (VPN4): Download blocks 862000-928000
  python3 hostinger_download.py 862000 928000 chunk4

  # After all complete:
  python3 hostinger_download.py combine
"""
import requests
import sqlite3
import json
import time
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
DATA_DIR = Path("blockchain_data")
API_BASE = "https://mempool.space/api"


def fetch_blocks(height: int) -> list:
    """Fetch 10 blocks from API."""
    try:
        time.sleep(0.15)  # Gentle delay
        resp = requests.get(f"{API_BASE}/blocks/{height}", timeout=20)
        if resp.status_code == 200:
            return [{'height': b['height'], 'timestamp': b['timestamp'], 'hash': b['id'],
                     'tx_count': b['tx_count'], 'size': b['size'], 'weight': b.get('weight', 0),
                     'fees': b.get('extras', {}).get('totalFees', 0),
                     'median_fee': b.get('extras', {}).get('medianFee', 0)}
                    for b in resp.json()]
        elif resp.status_code == 429:
            print(f"Rate limited at {height}, waiting 30s...")
            time.sleep(30)
            return fetch_blocks(height)  # Retry
    except Exception as e:
        print(f"Error at {height}: {e}")
        time.sleep(5)
    return []


def download_chunk(start_height: int, end_height: int, chunk_name: str):
    """Download a chunk of blocks to SQLite."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    db_file = DATA_DIR / f"{chunk_name}.db"
    progress_file = DATA_DIR / f"{chunk_name}_progress.json"

    # Load progress
    last_height = start_height
    if progress_file.exists():
        progress = json.loads(progress_file.read_text())
        last_height = progress.get("last_height", start_height)
        print(f"Resuming from height {last_height}")

    # Create database
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS blocks (
        height INTEGER PRIMARY KEY,
        timestamp INTEGER, hash TEXT, tx_count INTEGER,
        size INTEGER, weight INTEGER, fees INTEGER, median_fee REAL
    )""")
    conn.commit()

    total = end_height - start_height
    downloaded = last_height - start_height

    print("=" * 60)
    print(f"CHUNK: {chunk_name}")
    print(f"Range: {start_height:,} to {end_height:,} ({total:,} blocks)")
    print(f"Already downloaded: {downloaded:,}")
    print("=" * 60)

    start_time = time.time()

    # Download from last_height to end_height
    heights = list(range(end_height, last_height - 1, -10))

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for h in heights:
            futures.append(executor.submit(fetch_blocks, h))

        blocks_buffer = []
        for i, future in enumerate(as_completed(futures)):
            try:
                blocks = future.result()
                for b in blocks:
                    if start_height <= b['height'] <= end_height:
                        blocks_buffer.append(b)
                        downloaded += 1

                # Save every 500 blocks
                if len(blocks_buffer) >= 500:
                    for b in blocks_buffer:
                        c.execute("INSERT OR REPLACE INTO blocks VALUES (?,?,?,?,?,?,?,?)",
                            (b['height'], b['timestamp'], b['hash'], b['tx_count'],
                             b['size'], b['weight'], b['fees'], b['median_fee']))
                    conn.commit()

                    # Save progress
                    max_height = max(b['height'] for b in blocks_buffer)
                    progress_file.write_text(json.dumps({"last_height": max_height}))

                    elapsed = time.time() - start_time
                    rate = downloaded / elapsed if elapsed > 0 else 0
                    pct = 100 * downloaded / total
                    remaining = (total - downloaded) / rate / 60 if rate > 0 else 0

                    print(f"[{pct:5.1f}%] {downloaded:,}/{total:,} | "
                          f"{rate:.0f} blk/s | ETA: {remaining:.0f} min")

                    blocks_buffer = []

            except Exception as e:
                print(f"Error: {e}")

    # Save remaining
    for b in blocks_buffer:
        c.execute("INSERT OR REPLACE INTO blocks VALUES (?,?,?,?,?,?,?,?)",
            (b['height'], b['timestamp'], b['hash'], b['tx_count'],
             b['size'], b['weight'], b['fees'], b['median_fee']))
    conn.commit()
    conn.close()

    print(f"\nCHUNK {chunk_name} COMPLETE!")
    print(f"Downloaded: {downloaded:,} blocks")


def combine_chunks():
    """Combine all chunk databases into one."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    final_db = DATA_DIR / "bitcoin_2021_2025.db"

    conn = sqlite3.connect(final_db)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS blocks (
        height INTEGER PRIMARY KEY,
        timestamp INTEGER, hash TEXT, tx_count INTEGER,
        size INTEGER, weight INTEGER, fees INTEGER, median_fee REAL
    )""")
    conn.commit()

    chunk_files = list(DATA_DIR.glob("chunk*.db"))
    total = 0

    print("COMBINING ALL CHUNKS")
    print("=" * 50)

    for f in sorted(chunk_files):
        print(f"Importing {f.name}...")
        chunk_conn = sqlite3.connect(f)
        rows = chunk_conn.execute("SELECT * FROM blocks").fetchall()

        for r in rows:
            c.execute("INSERT OR REPLACE INTO blocks VALUES (?,?,?,?,?,?,?,?)", r)
            total += 1

        chunk_conn.close()
        conn.commit()

    conn.close()

    print(f"\nCOMPLETE: {total:,} blocks in {final_db}")
    print("\nTransfer this file back to your Windows machine:")
    print(f"  scp user@hostinger:{final_db} ./data/")


def show_status():
    """Show download status for all chunks."""
    print("CHUNK STATUS")
    print("=" * 50)

    for f in sorted(DATA_DIR.glob("*_progress.json")):
        chunk_name = f.stem.replace("_progress", "")
        progress = json.loads(f.read_text())
        db_file = DATA_DIR / f"{chunk_name}.db"

        if db_file.exists():
            conn = sqlite3.connect(db_file)
            count = conn.execute("SELECT COUNT(*) FROM blocks").fetchone()[0]
            min_h = conn.execute("SELECT MIN(height) FROM blocks").fetchone()[0]
            max_h = conn.execute("SELECT MAX(height) FROM blocks").fetchone()[0]
            conn.close()
            print(f"{chunk_name}: {count:,} blocks ({min_h:,} - {max_h:,})")
        else:
            print(f"{chunk_name}: No data yet")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("""
HOSTINGER PARALLEL DOWNLOADER
=============================

SETUP:
1. Connect 4 different VPNs in 4 terminals
2. Run each command in a different terminal:

   # VPN1 Terminal:
   python3 hostinger_download.py 664000 730000 chunk1

   # VPN2 Terminal:
   python3 hostinger_download.py 730000 796000 chunk2

   # VPN3 Terminal:
   python3 hostinger_download.py 796000 862000 chunk3

   # VPN4 Terminal:
   python3 hostinger_download.py 862000 928000 chunk4

3. After all complete:
   python3 hostinger_download.py combine

4. Transfer to Windows:
   scp user@hostinger:blockchain_data/bitcoin_2021_2025.db ./data/

COMMANDS:
  python3 hostinger_download.py START END CHUNK_NAME
  python3 hostinger_download.py status
  python3 hostinger_download.py combine
""")
    elif sys.argv[1] == "combine":
        combine_chunks()
    elif sys.argv[1] == "status":
        show_status()
    else:
        start = int(sys.argv[1])
        end = int(sys.argv[2])
        chunk = sys.argv[3]
        download_chunk(start, end, chunk)
