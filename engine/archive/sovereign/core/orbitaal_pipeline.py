#!/usr/bin/env python3
"""
ORBITAAL PIPELINE - USE WHAT WE HAVE

Direct pipeline from ORBITAAL Parquet files (2009-2021).
12 years of pre-processed blockchain data = READY NOW.

No API calls. No waiting. Just read the parquet files.
"""
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict

# Paths
ORBITAAL_DIR = Path("data/orbitaal")
STREAM_GRAPH = ORBITAAL_DIR / "STREAM_GRAPH" / "EDGES"
SNAPSHOT_DIR = ORBITAAL_DIR / "SNAPSHOT" / "EDGES"
NODE_TABLE = ORBITAAL_DIR / "NODE_TABLE"
FEATURES_DB = Path("data/bitcoin_features.db")


def load_stream_graph_year(year: int) -> pd.DataFrame:
    """Load transaction flow data for a specific year."""
    file_id = year - 2008  # 2009=01, 2010=02, etc.
    filename = f"orbitaal-stream_graph-date-{year}-file-id-{file_id:02d}.snappy.parquet"
    filepath = STREAM_GRAPH / filename

    if not filepath.exists():
        print(f"File not found: {filepath}")
        return pd.DataFrame()

    print(f"Loading {year} data...")
    df = pq.read_table(filepath).to_pandas()
    print(f"  {len(df):,} transactions")
    return df


def load_all_stream_graph() -> pd.DataFrame:
    """Load all years of transaction data."""
    dfs = []
    for year in range(2009, 2022):
        df = load_stream_graph_year(year)
        if len(df) > 0:
            dfs.append(df)

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()


def aggregate_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate transaction data into daily features for backtesting.

    Features:
    - tx_count: Number of transactions
    - total_value_btc: Total BTC transferred
    - total_value_usd: Total USD value
    - unique_senders: Unique sending entities
    - unique_receivers: Unique receiving entities
    - whale_tx_count: Transactions > 100 BTC
    - whale_value_btc: Value in whale transactions
    """
    # Convert timestamp to date
    df['date'] = pd.to_datetime(df['TIMESTAMP'], unit='s').dt.date

    # Whale threshold: 100 BTC = 10B satoshi
    WHALE_THRESHOLD = 100 * 1e8

    # Aggregate by day
    daily = df.groupby('date').agg({
        'TIMESTAMP': 'first',  # Keep one timestamp per day
        'VALUE_SATOSHI': ['count', 'sum'],
        'VALUE_USD': 'sum',
        'SRC_ID': 'nunique',
        'DST_ID': 'nunique',
    })

    # Flatten column names
    daily.columns = ['timestamp', 'tx_count', 'total_value_sat', 'total_value_usd',
                     'unique_senders', 'unique_receivers']

    # Convert satoshi to BTC
    daily['total_value_btc'] = daily['total_value_sat'] / 1e8

    # Whale transactions
    whale_df = df[df['VALUE_SATOSHI'] >= WHALE_THRESHOLD]
    whale_daily = whale_df.groupby(pd.to_datetime(whale_df['TIMESTAMP'], unit='s').dt.date).agg({
        'VALUE_SATOSHI': ['count', 'sum']
    })
    whale_daily.columns = ['whale_tx_count', 'whale_value_sat']
    whale_daily['whale_value_btc'] = whale_daily['whale_value_sat'] / 1e8

    # Merge whale data
    daily = daily.join(whale_daily[['whale_tx_count', 'whale_value_btc']], how='left')
    daily = daily.fillna(0)

    daily = daily.reset_index()
    daily = daily.rename(columns={'index': 'date'})

    return daily


def build_features_from_orbitaal():
    """
    Build complete feature database from ORBITAAL data.

    This gives us 12 years of daily features for backtesting.
    """
    print("=" * 60)
    print("ORBITAAL FEATURE BUILDER")
    print("Building 12 years of features from pre-processed data")
    print("=" * 60)
    print()

    # Initialize database
    FEATURES_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(FEATURES_DB)
    c = conn.cursor()

    c.execute("""CREATE TABLE IF NOT EXISTS daily_features (
        date TEXT PRIMARY KEY,
        timestamp INTEGER,
        tx_count INTEGER,
        total_value_btc REAL,
        total_value_usd REAL,
        unique_senders INTEGER,
        unique_receivers INTEGER,
        whale_tx_count INTEGER,
        whale_value_btc REAL,
        net_flow REAL
    )""")
    conn.commit()

    # Process each year
    for year in range(2009, 2022):
        print(f"\n[{year}] Processing...")

        df = load_stream_graph_year(year)
        if len(df) == 0:
            continue

        # Aggregate to daily
        daily = aggregate_daily_features(df)

        # Calculate net flow (entities)
        daily['net_flow'] = daily['unique_receivers'] - daily['unique_senders']

        # Store in database
        for _, row in daily.iterrows():
            c.execute("""INSERT OR REPLACE INTO daily_features VALUES
                (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (str(row['date']), int(row['timestamp']), int(row['tx_count']),
                 float(row['total_value_btc']), float(row['total_value_usd']),
                 int(row['unique_senders']), int(row['unique_receivers']),
                 int(row['whale_tx_count']), float(row['whale_value_btc']),
                 float(row['net_flow'])))

        conn.commit()
        print(f"  Stored {len(daily)} days")

        # Free memory
        del df
        del daily

    # Summary
    total = c.execute("SELECT COUNT(*) FROM daily_features").fetchone()[0]
    date_range = c.execute("SELECT MIN(date), MAX(date) FROM daily_features").fetchone()

    conn.close()

    print()
    print("=" * 60)
    print("COMPLETE")
    print(f"Total days: {total:,}")
    print(f"Date range: {date_range[0]} to {date_range[1]}")
    print(f"Database: {FEATURES_DB}")
    print("=" * 60)


def quick_stats():
    """Show quick stats about available ORBITAAL data."""
    print("ORBITAAL DATA INVENTORY")
    print("=" * 50)

    # Stream graph files
    print("\nSTREAM_GRAPH (Transaction Flows):")
    total_size = 0
    total_rows = 0
    for year in range(2009, 2022):
        file_id = year - 2008
        filename = f"orbitaal-stream_graph-date-{year}-file-id-{file_id:02d}.snappy.parquet"
        filepath = STREAM_GRAPH / filename
        if filepath.exists():
            size = filepath.stat().st_size / 1e9
            total_size += size
            # Quick row count from metadata
            meta = pq.read_metadata(filepath)
            rows = meta.num_rows
            total_rows += rows
            print(f"  {year}: {rows:>15,} txs  ({size:.2f} GB)")

    print(f"\n  TOTAL: {total_rows:,} transactions ({total_size:.1f} GB)")

    # Check if features DB exists
    print(f"\nFeatures DB: {FEATURES_DB}")
    if FEATURES_DB.exists():
        conn = sqlite3.connect(FEATURES_DB)
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        print(f"  Tables: {[t[0] for t in tables]}")

        if ('daily_features',) in tables:
            count = conn.execute("SELECT COUNT(*) FROM daily_features").fetchone()[0]
            print(f"  Daily features: {count:,} days")
        conn.close()
    else:
        print("  Not created yet. Run: python orbitaal_pipeline.py build")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("ORBITAAL PIPELINE")
        print("=" * 40)
        print("Commands:")
        print("  stats  - Show data inventory")
        print("  build  - Build features from ORBITAAL (takes ~10 min)")
        print()
        print("ORBITAAL = 12 years of pre-processed Bitcoin data")
        print("No API calls needed. Just read parquet files.")
    else:
        cmd = sys.argv[1]
        if cmd == "stats":
            quick_stats()
        elif cmd == "build":
            build_features_from_orbitaal()
        else:
            print(f"Unknown command: {cmd}")
