#!/usr/bin/env python3
"""
LOAD ENTITY ADDRESSES - 7.6M EXCHANGE ADDRESSES
================================================
Load pre-labeled exchange addresses from CSV into database.
"""

import sys
import csv
import sqlite3
from datetime import datetime

def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "/root/sovereign/entity_addresses/Exchanges_full_detailed.csv"
    db_path = "/root/sovereign/walletexplorer_addresses.db"  # Main database

    print("=" * 70)
    print("LOADING ENTITY ADDRESSES")
    print("=" * 70)
    print(f"Source: {csv_path}")
    print(f"Target: {db_path}")
    print()

    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Use existing schema
    c.execute("""
        CREATE TABLE IF NOT EXISTS addresses (
            address TEXT PRIMARY KEY,
            exchange TEXT,
            balance_sat INTEGER,
            downloaded_at TEXT
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_exchange ON addresses(exchange)")

    # Read CSV and insert
    batch = []
    batch_size = 50000
    total = 0
    exchanges = {}

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            addr = row.get('hashAdd', '').strip()
            exchange = row.get('exchange', 'unknown').strip()

            if not addr:
                continue

            # Normalize exchange name
            exchange = exchange.lower().replace('.com', '').replace('.net', '').replace('.io', '')

            batch.append((addr, exchange, datetime.now().isoformat()))
            exchanges[exchange] = exchanges.get(exchange, 0) + 1
            total += 1

            if len(batch) >= batch_size:
                c.executemany("INSERT OR IGNORE INTO addresses (address, exchange, downloaded_at) VALUES (?, ?, ?)", batch)
                conn.commit()
                print(f"\r  Loaded {total:,} addresses...", end='', flush=True)
                batch = []

    # Final batch
    if batch:
        c.executemany("INSERT OR IGNORE INTO addresses (address, exchange, downloaded_at) VALUES (?, ?, ?)", batch)
        conn.commit()

    print()
    print()
    print("=" * 70)
    print(f"LOADED {total:,} ADDRESSES")
    print("=" * 70)

    # Show per-exchange counts
    for ex, count in sorted(exchanges.items(), key=lambda x: -x[1])[:30]:
        print(f"  {ex:<30} {count:>10,}")

    # Verify
    c.execute("SELECT COUNT(*) FROM addresses")
    db_count = c.fetchone()[0]
    print()
    print(f"Total in database: {db_count:,}")

    conn.close()


if __name__ == '__main__':
    main()
