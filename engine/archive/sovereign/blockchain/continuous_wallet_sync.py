#!/usr/bin/env python3
"""
CONTINUOUS WALLET ADDRESS SYNC - Runs 24/7
Downloads exchange addresses from WalletExplorer.com on repeat
"""

import json
import sqlite3
import urllib.request
import time
import sys
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    "Bittrex.com",
    "Poloniex.com",
    "Luno.com",
    "Paxful.com",
    "LocalBitcoins.com",
]

API_URL = "https://www.walletexplorer.com/api/1/wallet-addresses"
BATCH_SIZE = 1000
DB_PATH = "/root/sovereign/walletexplorer_addresses.db"
NUM_THREADS = 15
SYNC_INTERVAL = 3600  # Sync every hour


def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS addresses (
            address TEXT PRIMARY KEY,
            exchange TEXT,
            balance_sat INTEGER,
            last_seen TEXT
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_exchange ON addresses(exchange)")
    c.execute("""
        CREATE TABLE IF NOT EXISTS sync_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            exchange TEXT,
            count INTEGER,
            synced_at TEXT
        )
    """)
    conn.commit()
    return conn


def fetch_batch(wallet: str, offset: int) -> list:
    url = f"{API_URL}?wallet={wallet}&from={offset}&count={BATCH_SIZE}"
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "Mozilla/5.0 (compatible; WalletSync/1.0)")
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            return data.get("addresses", [])
    except Exception as e:
        return []


def sync_exchange(conn: sqlite3.Connection, wallet: str) -> int:
    """Sync all addresses for one exchange."""
    exchange_name = wallet.replace(".com", "").replace(".net", "").replace(".io", "").lower()

    # Parallel fetch all pages
    offsets = list(range(0, 500000, BATCH_SIZE))
    all_addresses = []

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = {executor.submit(fetch_batch, wallet, off): off for off in offsets}
        for future in as_completed(futures):
            batch = future.result()
            if batch:
                all_addresses.extend(batch)

    if not all_addresses:
        return 0

    # Deduplicate and insert
    now = datetime.now().isoformat()
    seen = set()
    c = conn.cursor()
    count = 0

    for addr_data in all_addresses:
        addr = addr_data.get("address", "")
        if addr and addr not in seen:
            seen.add(addr)
            balance = int(addr_data.get("balance", 0) * 1e8)
            try:
                c.execute(
                    "INSERT OR REPLACE INTO addresses (address, exchange, balance_sat, last_seen) VALUES (?, ?, ?, ?)",
                    (addr, exchange_name, balance, now)
                )
                count += 1
            except:
                pass

    # Log sync
    c.execute("INSERT INTO sync_log (exchange, count, synced_at) VALUES (?, ?, ?)",
              (exchange_name, count, now))
    conn.commit()

    return count


def get_stats(conn: sqlite3.Connection) -> dict:
    """Get current database stats."""
    c = conn.cursor()
    c.execute("SELECT exchange, COUNT(*) FROM addresses GROUP BY exchange ORDER BY COUNT(*) DESC")
    return {row[0]: row[1] for row in c.fetchall()}


def run_sync_cycle(conn: sqlite3.Connection):
    """Run one complete sync cycle for all exchanges."""
    log("=" * 60)
    log("STARTING SYNC CYCLE")
    log("=" * 60)

    total = 0
    for wallet in EXCHANGES:
        try:
            count = sync_exchange(conn, wallet)
            if count > 0:
                log(f"  {wallet}: {count:,} addresses")
            total += count
        except Exception as e:
            log(f"  {wallet}: ERROR - {e}")

    # Print stats
    stats = get_stats(conn)
    total_addrs = sum(stats.values())

    log("-" * 60)
    log(f"CYCLE COMPLETE: {total:,} addresses synced")
    log(f"DATABASE TOTAL: {total_addrs:,} addresses")
    log("-" * 60)

    return total


def main():
    log("=" * 70)
    log("CONTINUOUS WALLET SYNC - 24/7 SERVICE")
    log(f"Database: {DB_PATH}")
    log(f"Exchanges: {len(EXCHANGES)}")
    log(f"Sync interval: {SYNC_INTERVAL}s ({SYNC_INTERVAL/3600:.1f}h)")
    log("=" * 70)

    conn = init_db()

    # Initial stats
    stats = get_stats(conn)
    if stats:
        log("Current database:")
        for ex, cnt in sorted(stats.items(), key=lambda x: -x[1])[:10]:
            log(f"  {ex}: {cnt:,}")

    cycle = 0
    while True:
        cycle += 1
        log(f"\n>>> CYCLE {cycle} <<<")

        try:
            run_sync_cycle(conn)
        except Exception as e:
            log(f"CYCLE ERROR: {e}")

        log(f"Sleeping {SYNC_INTERVAL}s until next cycle...")
        time.sleep(SYNC_INTERVAL)


if __name__ == "__main__":
    main()
