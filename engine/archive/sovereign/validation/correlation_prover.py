"""
Correlation Prover - Historical Validation

PROVES (or disproves) the 1:1 correlation between:
- Blockchain INFLOW to exchanges → Price DROP
- Blockchain OUTFLOW from exchanges → Price RISE

Uses historical orbitaal daily transaction data + daily price data.
Must run BEFORE trading to validate the mathematical edge.
"""
import os
import sys
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = os.path.join(BASE_DIR, "data")
ORBITAAL_DIR = os.path.join(DATA_DIR, "orbitaal", "SNAPSHOT", "EDGES", "day")
ENTITY_DIR = os.path.join(DATA_DIR, "entity_addresses", "data")
RESULTS_DB = os.path.join(DATA_DIR, "correlation_results.db")


def init_database():
    """Initialize the correlation results database."""
    conn = sqlite3.connect(RESULTS_DB)
    cursor = conn.cursor()

    cursor.execute('''CREATE TABLE IF NOT EXISTS historical_correlation (
        id INTEGER PRIMARY KEY,
        flow_type TEXT,
        min_btc REAL,
        max_btc REAL,
        sample_count INTEGER,
        correlation REAL,
        win_rate REAL,
        avg_move REAL,
        timeframe TEXT,
        calculated_at TEXT
    )''')

    cursor.execute('''CREATE TABLE IF NOT EXISTS daily_flows (
        id INTEGER PRIMARY KEY,
        date TEXT UNIQUE,
        inflow_btc REAL,
        outflow_btc REAL,
        net_flow REAL,
        inflow_count INTEGER,
        outflow_count INTEGER,
        inflow_100btc INTEGER,
        outflow_100btc INTEGER,
        price_open REAL,
        price_close REAL,
        price_change REAL,
        next_day_change REAL
    )''')

    conn.commit()
    return conn


def load_exchange_addresses():
    """Load exchange address IDs for fast lookup."""
    print("[1/5] Loading exchange addresses...")

    exchanges_file = os.path.join(ENTITY_DIR, "Exchanges_full_detailed.csv")
    df = pd.read_csv(exchanges_file, usecols=['add_num'])

    # Create set for O(1) lookup
    exchange_ids = set(df['add_num'].values)

    print(f"      Loaded {len(exchange_ids):,} exchange addresses")
    return exchange_ids


def load_daily_prices():
    """Load daily price data."""
    print("[2/5] Loading daily price data...")

    # Try unified_bitcoin.db first (has 'date' TEXT column)
    db_path = os.path.join(DATA_DIR, "unified_bitcoin.db")
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        df = pd.read_sql("SELECT date, open, close FROM prices ORDER BY date", conn)
        conn.close()

        df['change'] = (df['close'] - df['open']) / df['open']
        df['next_day_change'] = df['change'].shift(-1)

        prices = df.set_index('date').to_dict('index')
        print(f"      Loaded {len(prices):,} daily prices from unified_bitcoin.db")
        return prices

    # Fallback to bitcoin_features.db (has 'timestamp' INTEGER column)
    db_path = os.path.join(DATA_DIR, "bitcoin_features.db")
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        df = pd.read_sql("SELECT timestamp, open, close FROM prices ORDER BY timestamp", conn)
        conn.close()

        df['date'] = pd.to_datetime(df['timestamp'], unit='s').dt.strftime('%Y-%m-%d')
        df['change'] = (df['close'] - df['open']) / df['open']
        df['next_day_change'] = df['change'].shift(-1)

        prices = df.set_index('date').to_dict('index')
        print(f"      Loaded {len(prices):,} daily prices from bitcoin_features.db")
        return prices

    print("      ERROR: No price data found!")
    return {}


def process_daily_file(filepath, exchange_ids, date_str):
    """Process one day's transaction data."""
    try:
        df = pd.read_parquet(filepath)
    except Exception as e:
        return None

    # Find flows: sender OR receiver is exchange
    inflow_mask = df['DST_ID'].isin(exchange_ids)
    outflow_mask = df['SRC_ID'].isin(exchange_ids)

    # Exclude internal transfers (both sender and receiver are exchanges)
    internal_mask = inflow_mask & outflow_mask

    # Calculate flows
    inflows = df[inflow_mask & ~internal_mask]
    outflows = df[outflow_mask & ~internal_mask]

    inflow_btc = inflows['VALUE_SATOSHI'].sum() / 1e8
    outflow_btc = outflows['VALUE_SATOSHI'].sum() / 1e8

    # Count significant flows (100+ BTC)
    inflow_100btc = (inflows['VALUE_SATOSHI'] >= 100e8).sum()
    outflow_100btc = (outflows['VALUE_SATOSHI'] >= 100e8).sum()

    return {
        'date': date_str,
        'inflow_btc': inflow_btc,
        'outflow_btc': outflow_btc,
        'net_flow': inflow_btc - outflow_btc,  # Positive = net inflow
        'inflow_count': len(inflows),
        'outflow_count': len(outflows),
        'inflow_100btc': inflow_100btc,
        'outflow_100btc': outflow_100btc
    }


def process_all_files(exchange_ids, prices, sample_rate=1):
    """Process orbitaal daily files."""
    print("[3/5] Processing daily flow files...")

    files = sorted([f for f in os.listdir(ORBITAAL_DIR) if f.endswith('.parquet')])
    print(f"      Found {len(files)} daily files")

    # Filter to dates we have prices for
    price_dates = set(prices.keys())
    print(f"      Price data for {len(price_dates)} dates")

    results = []
    processed = 0
    matched = 0

    for i, filename in enumerate(files[::sample_rate]):
        # Extract date from filename: orbitaal-snapshot-date-YYYY-MM-DD-file-id-XXXX
        parts = filename.split('-')
        if len(parts) >= 6:
            date_str = f"{parts[3]}-{parts[4]}-{parts[5]}"
        else:
            continue

        # Only process if we have price data
        if date_str not in price_dates:
            continue

        if matched % 50 == 0:
            print(f"      Processing {matched+1}... ({date_str})")

        filepath = os.path.join(ORBITAAL_DIR, filename)
        flow_data = process_daily_file(filepath, exchange_ids, date_str)

        if flow_data:
            # Add price data
            price_info = prices[date_str]
            flow_data['price_open'] = price_info.get('open', 0)
            flow_data['price_close'] = price_info.get('close', 0)
            flow_data['price_change'] = price_info.get('change', 0)
            flow_data['next_day_change'] = price_info.get('next_day_change', None)

            results.append(flow_data)
            matched += 1

        processed += 1

    print(f"      Processed {processed} files, matched {matched} with price data")
    return results


def analyze_correlation(df, conn):
    """Analyze correlation between flows and price changes."""
    print("\n" + "="*70)
    print("BLOCKCHAIN-TO-EXCHANGE CORRELATION ANALYSIS")
    print("="*70)

    print(f"\nData points: {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Total INFLOW: {df['inflow_btc'].sum():,.0f} BTC")
    print(f"Total OUTFLOW: {df['outflow_btc'].sum():,.0f} BTC")

    # Same-day correlation
    print(f"\n{'='*70}")
    print("SAME-DAY CORRELATION")
    print(f"{'='*70}")

    # Net flow vs price change
    corr_net = df['net_flow'].corr(df['price_change'])
    print(f"\nNet Flow vs Price Change: {corr_net:+.4f}")
    print(f"  (Negative = CONFIRMS: net inflow → price drop)")

    # When net inflow is high, does price drop?
    high_inflow = df[df['net_flow'] > df['net_flow'].quantile(0.75)]
    high_outflow = df[df['net_flow'] < df['net_flow'].quantile(0.25)]

    high_inflow_change = high_inflow['price_change'].mean()
    high_outflow_change = high_outflow['price_change'].mean()

    print(f"\nHigh NET INFLOW days (top 25%, {len(high_inflow)} days):")
    print(f"  Average price change: {high_inflow_change*100:+.3f}%")
    print(f"  Expected: NEGATIVE (they're depositing to sell)")
    win_rate_inflow = (high_inflow['price_change'] < 0).mean()
    print(f"  Win rate (SHORT): {win_rate_inflow*100:.1f}%")

    if win_rate_inflow > 0.52:
        print(f"  ✓ CONFIRMED")
    else:
        print(f"  ✗ NOT CONFIRMED")

    print(f"\nHigh NET OUTFLOW days (top 25%, {len(high_outflow)} days):")
    print(f"  Average price change: {high_outflow_change*100:+.3f}%")
    print(f"  Expected: POSITIVE (they're withdrawing to hold)")
    win_rate_outflow = (high_outflow['price_change'] > 0).mean()
    print(f"  Win rate (LONG): {win_rate_outflow*100:.1f}%")

    if win_rate_outflow > 0.52:
        print(f"  ✓ CONFIRMED")
    else:
        print(f"  ✗ NOT CONFIRMED")

    # Save to database
    conn.execute('''INSERT OR REPLACE INTO historical_correlation
        (flow_type, min_btc, max_btc, sample_count, correlation, win_rate, avg_move, timeframe, calculated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
        ('NET_INFLOW', 0, float('inf'), len(high_inflow), corr_net, win_rate_inflow,
         high_inflow_change, 'same_day', datetime.now().isoformat()))

    conn.execute('''INSERT OR REPLACE INTO historical_correlation
        (flow_type, min_btc, max_btc, sample_count, correlation, win_rate, avg_move, timeframe, calculated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
        ('NET_OUTFLOW', 0, float('inf'), len(high_outflow), -corr_net, win_rate_outflow,
         high_outflow_change, 'same_day', datetime.now().isoformat()))

    # Next-day prediction
    print(f"\n{'='*70}")
    print("NEXT-DAY PREDICTION")
    print(f"{'='*70}")

    df_next = df.dropna(subset=['next_day_change'])
    if len(df_next) > 0:
        corr_next = df_next['net_flow'].corr(df_next['next_day_change'])
        print(f"\nNet Flow vs NEXT-DAY Price Change: {corr_next:+.4f}")

        high_inflow_next = df_next[df_next['net_flow'] > df_next['net_flow'].quantile(0.75)]
        high_outflow_next = df_next[df_next['net_flow'] < df_next['net_flow'].quantile(0.25)]

        print(f"\nHigh NET INFLOW → Next day: {high_inflow_next['next_day_change'].mean()*100:+.3f}%")
        print(f"High NET OUTFLOW → Next day: {high_outflow_next['next_day_change'].mean()*100:+.3f}%")

    # 100+ BTC flows analysis
    print(f"\n{'='*70}")
    print("MEGA-FLOW ANALYSIS (100+ BTC transactions)")
    print(f"{'='*70}")

    mega_inflow_days = df[df['inflow_100btc'] > 0]
    mega_outflow_days = df[df['outflow_100btc'] > 0]

    if len(mega_inflow_days) > 10:
        avg_change = mega_inflow_days['price_change'].mean()
        wr = (mega_inflow_days['price_change'] < 0).mean()
        print(f"\nDays with 100+ BTC INFLOWS ({len(mega_inflow_days)} days):")
        print(f"  Avg price change: {avg_change*100:+.3f}%")
        print(f"  Win rate (SHORT): {wr*100:.1f}%")

        conn.execute('''INSERT OR REPLACE INTO historical_correlation
            (flow_type, min_btc, max_btc, sample_count, correlation, win_rate, avg_move, timeframe, calculated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            ('MEGA_INFLOW', 100, float('inf'), len(mega_inflow_days), 0, wr,
             avg_change, 'same_day', datetime.now().isoformat()))

    if len(mega_outflow_days) > 10:
        avg_change = mega_outflow_days['price_change'].mean()
        wr = (mega_outflow_days['price_change'] > 0).mean()
        print(f"\nDays with 100+ BTC OUTFLOWS ({len(mega_outflow_days)} days):")
        print(f"  Avg price change: {avg_change*100:+.3f}%")
        print(f"  Win rate (LONG): {wr*100:.1f}%")

        conn.execute('''INSERT OR REPLACE INTO historical_correlation
            (flow_type, min_btc, max_btc, sample_count, correlation, win_rate, avg_move, timeframe, calculated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            ('MEGA_OUTFLOW', 100, float('inf'), len(mega_outflow_days), 0, wr,
             avg_change, 'same_day', datetime.now().isoformat()))

    conn.commit()

    # Final verdict
    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")

    if win_rate_inflow > 0.52 and win_rate_outflow > 0.52:
        print("\n✓ EDGE CONFIRMED")
        print(f"  Inflow SHORT win rate: {win_rate_inflow*100:.1f}%")
        print(f"  Outflow LONG win rate: {win_rate_outflow*100:.1f}%")
        print("\n  → Safe to trade with blockchain flow signals")
    elif win_rate_inflow > 0.52:
        print("\n⚠ PARTIAL EDGE: Only INFLOW→SHORT confirmed")
        print(f"  Win rate: {win_rate_inflow*100:.1f}%")
        print("\n  → Only trade SHORT signals")
    elif win_rate_outflow > 0.52:
        print("\n⚠ PARTIAL EDGE: Only OUTFLOW→LONG confirmed")
        print(f"  Win rate: {win_rate_outflow*100:.1f}%")
        print("\n  → Only trade LONG signals")
    else:
        print("\n✗ NO EDGE DETECTED")
        print(f"  Inflow SHORT win rate: {win_rate_inflow*100:.1f}%")
        print(f"  Outflow LONG win rate: {win_rate_outflow*100:.1f}%")
        print("\n  → DO NOT TRADE with this strategy")

    return {
        'inflow_win_rate': win_rate_inflow,
        'outflow_win_rate': win_rate_outflow,
        'correlation': corr_net
    }


def main():
    print("="*70)
    print("CORRELATION PROVER - Historical Validation")
    print("="*70)
    print("\nProving blockchain-to-exchange correlation with REAL DATA.\n")

    # Initialize database
    conn = init_database()

    # Load data
    exchange_ids = load_exchange_addresses()
    prices = load_daily_prices()

    if not prices:
        print("\nERROR: Cannot proceed without price data!")
        return

    # Process flows
    results = process_all_files(exchange_ids, prices, sample_rate=1)

    if not results:
        print("\nERROR: No flow data processed!")
        return

    # Convert to DataFrame and save
    df = pd.DataFrame(results)

    print(f"\n[4/5] Saving {len(df)} daily flow records...")
    for _, row in df.iterrows():
        try:
            conn.execute('''INSERT OR REPLACE INTO daily_flows
                (date, inflow_btc, outflow_btc, net_flow, inflow_count, outflow_count,
                 inflow_100btc, outflow_100btc, price_open, price_close, price_change, next_day_change)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (row['date'], row['inflow_btc'], row['outflow_btc'], row['net_flow'],
                 row['inflow_count'], row['outflow_count'], row['inflow_100btc'], row['outflow_100btc'],
                 row['price_open'], row['price_close'], row['price_change'], row.get('next_day_change')))
        except:
            pass
    conn.commit()

    # Analyze
    print("\n[5/5] Analyzing correlation...")
    analyze_correlation(df, conn)

    conn.close()
    print(f"\nResults saved to: {RESULTS_DB}")


if __name__ == "__main__":
    main()
