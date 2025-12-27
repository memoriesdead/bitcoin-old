#!/usr/bin/env python3
"""
FAST Download exchange addresses from WalletExplorer.com API
Uses parallel threads for maximum speed
"""

import json
import sqlite3
import urllib.request
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Major exchanges to download
EXCHANGES = [
    "Binance.com",
    "Coinbase.com",
    "Kraken.com",
    "OKX.com",
    "OKCoin.com",
    "Bybit.com",
    "Bitfinex.com",
    "Huobi.com",
    "KuCoin.com",
    "Gemini.com",
    "Bitstamp.net",
    "Gate.io",
    "Crypto.com",
    "BitMEX.com",
    "Deribit.com",
    "FTX.com",
    "Bittrex.com",
    "Poloniex.com",
]

API_URL = "https://www.walletexplorer.com/api/1/wallet-addresses"
BATCH_SIZE = 1000
DB_PATH = "/root/sovereign/walletexplorer_addresses.db"
NUM_THREADS = 20  # Aggressive parallel

db_lock = threading.Lock()


def init_db(db_path: str):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS addresses (
            address TEXT PRIMARY KEY,
            exchange TEXT,
            balance_sat INTEGER
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_exchange ON addresses(exchange)")
    conn.commit()
    return conn


def fetch_batch(wallet: str, offset: int) -> list:
    """Fetch one batch of addresses."""
    url = f"{API_URL}?wallet={wallet}&from={offset}&count={BATCH_SIZE}"
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "Mozilla/5.0")
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            return data.get("addresses", [])
    except:
        return []


def download_exchange_fast(conn: sqlite3.Connection, wallet: str):
    """Download all addresses for one exchange using parallel requests."""
    print(f"\n[{wallet}] Starting...")

    exchange_name = wallet.replace(".com", "").replace(".net", "").replace(".io", "").lower()

    # First, get total count by fetching first batch
    first_batch = fetch_batch(wallet, 0)
    if not first_batch:
        print(f"[{wallet}] No addresses found")
        return 0

    # Estimate total (assume ~300K max per exchange)
    # We'll fetch until we get empty results
    offsets = list(range(0, 500000, BATCH_SIZE))

    all_addresses = []

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = {executor.submit(fetch_batch, wallet, off): off for off in offsets}

        for future in as_completed(futures):
            batch = future.result()
            if batch:
                all_addresses.extend(batch)
                if len(all_addresses) % 10000 == 0:
                    print(f"[{wallet}] {len(all_addresses):,} addresses...")

    # Deduplicate
    seen = set()
    unique = []
    for addr_data in all_addresses:
        addr = addr_data.get("address", "")
        if addr and addr not in seen:
            seen.add(addr)
            unique.append(addr_data)

    # Bulk insert
    with db_lock:
        c = conn.cursor()
        for addr_data in unique:
            addr = addr_data.get("address", "")
            balance = int(addr_data.get("balance", 0) * 1e8)
            try:
                c.execute(
                    "INSERT OR REPLACE INTO addresses (address, exchange, balance_sat) VALUES (?, ?, ?)",
                    (addr, exchange_name, balance)
                )
            except:
                pass
        conn.commit()

    print(f"[{wallet}] DONE: {len(unique):,} addresses")
    return len(unique)


def main():
    print("=" * 70)
    print("FAST WALLETEXPLORER DOWNLOADER")
    print(f"Threads: {NUM_THREADS} | Started: {datetime.now()}")
    print("=" * 70)

    conn = init_db(DB_PATH)
    total = 0

    # Download exchanges in parallel too
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(download_exchange_fast, conn, ex): ex for ex in EXCHANGES}
        for future in as_completed(futures):
            try:
                count = future.result()
                total += count
            except Exception as e:
                print(f"Error: {e}")

    conn.close()

    print()
    print("=" * 70)
    print(f"COMPLETE: {total:,} total addresses")
    print(f"Database: {DB_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
