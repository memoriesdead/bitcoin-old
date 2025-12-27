#!/bin/bash
# HOSTINGER QUICK SETUP - Run this first on your VPS
# Usage: bash hostinger_setup.sh

echo "=========================================="
echo "HOSTINGER BLOCKCHAIN DOWNLOADER SETUP"
echo "=========================================="

# Install dependencies
apt update
apt install -y python3 python3-pip tmux screen curl wget

pip3 install requests

# Create working directory
mkdir -p /root/blockchain_data
cd /root

# Create the download script
cat > hostinger_download.py << 'SCRIPT_END'
#!/usr/bin/env python3
"""
HOSTINGER PARALLEL DOWNLOADER
Run different chunks on different VPNs/IPs.
"""
import requests
import sqlite3
import json
import time
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

DATA_DIR = Path("blockchain_data")
API_BASE = "https://mempool.space/api"

def fetch_blocks(height: int) -> list:
    try:
        time.sleep(0.15)
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
            return fetch_blocks(height)
    except Exception as e:
        print(f"Error at {height}: {e}")
        time.sleep(5)
    return []

def download_chunk(start_height: int, end_height: int, chunk_name: str):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    db_file = DATA_DIR / f"{chunk_name}.db"
    progress_file = DATA_DIR / f"{chunk_name}_progress.json"

    last_height = start_height
    if progress_file.exists():
        progress = json.loads(progress_file.read_text())
        last_height = progress.get("last_height", start_height)
        print(f"Resuming from height {last_height}")

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
    heights = list(range(end_height, last_height - 1, -10))

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(fetch_blocks, h) for h in heights]
        blocks_buffer = []

        for future in as_completed(futures):
            try:
                blocks = future.result()
                for b in blocks:
                    if start_height <= b['height'] <= end_height:
                        blocks_buffer.append(b)
                        downloaded += 1

                if len(blocks_buffer) >= 500:
                    for b in blocks_buffer:
                        c.execute("INSERT OR REPLACE INTO blocks VALUES (?,?,?,?,?,?,?,?)",
                            (b['height'], b['timestamp'], b['hash'], b['tx_count'],
                             b['size'], b['weight'], b['fees'], b['median_fee']))
                    conn.commit()

                    max_height = max(b['height'] for b in blocks_buffer)
                    progress_file.write_text(json.dumps({"last_height": max_height}))

                    elapsed = time.time() - start_time
                    rate = downloaded / elapsed if elapsed > 0 else 0
                    pct = 100 * downloaded / total
                    remaining = (total - downloaded) / rate / 60 if rate > 0 else 0

                    print(f"[{pct:5.1f}%] {downloaded:,}/{total:,} | {rate:.0f} blk/s | ETA: {remaining:.0f} min")
                    blocks_buffer = []
            except Exception as e:
                print(f"Error: {e}")

    for b in blocks_buffer:
        c.execute("INSERT OR REPLACE INTO blocks VALUES (?,?,?,?,?,?,?,?)",
            (b['height'], b['timestamp'], b['hash'], b['tx_count'],
             b['size'], b['weight'], b['fees'], b['median_fee']))
    conn.commit()
    conn.close()
    print(f"\nCHUNK {chunk_name} COMPLETE! Downloaded: {downloaded:,} blocks")

def combine_chunks():
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

def show_status():
    print("CHUNK STATUS")
    print("=" * 50)
    for f in sorted(DATA_DIR.glob("*_progress.json")):
        chunk_name = f.stem.replace("_progress", "")
        db_file = DATA_DIR / f"{chunk_name}.db"
        if db_file.exists():
            conn = sqlite3.connect(db_file)
            count = conn.execute("SELECT COUNT(*) FROM blocks").fetchone()[0]
            min_h = conn.execute("SELECT MIN(height) FROM blocks").fetchone()[0]
            max_h = conn.execute("SELECT MAX(height) FROM blocks").fetchone()[0]
            conn.close()
            print(f"{chunk_name}: {count:,} blocks ({min_h:,} - {max_h:,})")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("""
HOSTINGER PARALLEL DOWNLOADER
=============================
COMMANDS:
  python3 hostinger_download.py 664000 730000 chunk1  (Terminal 1)
  python3 hostinger_download.py 730000 796000 chunk2  (Terminal 2)
  python3 hostinger_download.py 796000 862000 chunk3  (Terminal 3)
  python3 hostinger_download.py 862000 928000 chunk4  (Terminal 4)

  python3 hostinger_download.py combine  (After all complete)
  python3 hostinger_download.py status   (Check progress)
""")
    elif sys.argv[1] == "combine":
        combine_chunks()
    elif sys.argv[1] == "status":
        show_status()
    else:
        download_chunk(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
SCRIPT_END

chmod +x hostinger_download.py

echo ""
echo "=========================================="
echo "SETUP COMPLETE!"
echo "=========================================="
echo ""
echo "NOW RUN THESE IN 4 SEPARATE TERMINALS:"
echo ""
echo "  Terminal 1: python3 hostinger_download.py 664000 730000 chunk1"
echo "  Terminal 2: python3 hostinger_download.py 730000 796000 chunk2"
echo "  Terminal 3: python3 hostinger_download.py 796000 862000 chunk3"
echo "  Terminal 4: python3 hostinger_download.py 862000 928000 chunk4"
echo ""
echo "Or use tmux to run all in background:"
echo ""
echo "  tmux new-session -d -s c1 'python3 hostinger_download.py 664000 730000 chunk1'"
echo "  tmux new-session -d -s c2 'python3 hostinger_download.py 730000 796000 chunk2'"
echo "  tmux new-session -d -s c3 'python3 hostinger_download.py 796000 862000 chunk3'"
echo "  tmux new-session -d -s c4 'python3 hostinger_download.py 862000 928000 chunk4'"
echo ""
echo "Check status: python3 hostinger_download.py status"
echo "Combine after: python3 hostinger_download.py combine"
echo ""
