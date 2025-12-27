"""
Flow-to-Price Correlation Validator

Validates the fundamental hypothesis:
- INFLOW to exchange → Price drops (they're selling)
- OUTFLOW from exchange → Price rises (they're holding)

Uses historical orbitaal data (2009-2021) + price data to prove/disprove correlation.
"""
import os
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict

# Paths
DATA_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ORBITAAL_DIR = os.path.join(DATA_DIR, "data", "orbitaal", "SNAPSHOT", "EDGES", "day")
ENTITY_DIR = os.path.join(DATA_DIR, "data", "entity_addresses", "data")
FEATURES_DB = os.path.join(DATA_DIR, "data", "bitcoin_features.db")


def load_exchange_addresses():
    """Load exchange address IDs from entity data."""
    print("[1/4] Loading exchange addresses...")

    exchanges_file = os.path.join(ENTITY_DIR, "Exchanges_full_detailed.csv")
    df = pd.read_csv(exchanges_file, usecols=['add_num', 'exchange'])

    # Create set of exchange address IDs for fast lookup
    exchange_ids = set(df['add_num'].values)

    # Also create mapping to exchange name
    id_to_exchange = dict(zip(df['add_num'], df['exchange']))

    print(f"    Loaded {len(exchange_ids):,} exchange addresses")
    return exchange_ids, id_to_exchange


def load_prices():
    """Load daily price data."""
    print("[2/4] Loading price data...")

    conn = sqlite3.connect(FEATURES_DB)
    df = pd.read_sql("SELECT timestamp, open, close, high, low FROM prices", conn)
    conn.close()

    # Convert timestamp to date
    df['date'] = pd.to_datetime(df['timestamp'], unit='s').dt.date

    # Calculate daily return
    df['return'] = (df['close'] - df['open']) / df['open']
    df['next_day_return'] = df['return'].shift(-1)

    # Create date-indexed lookup
    price_data = df.set_index('date').to_dict('index')

    print(f"    Loaded {len(df)} days of price data")
    print(f"    Date range: {df['date'].min()} to {df['date'].max()}")

    return price_data


def process_daily_flows(exchange_ids, start_date=None, end_date=None, sample_days=100):
    """Process orbitaal daily snapshots to compute exchange flows."""
    print(f"[3/4] Processing daily flows (sampling {sample_days} days)...")

    files = sorted([f for f in os.listdir(ORBITAAL_DIR) if f.endswith('.parquet')])

    # Parse dates from filenames and filter
    dated_files = []
    for f in files:
        # Extract date from filename: orbitaal-snapshot-date-YYYY-MM-DD-file-id-XXXX.snappy.parquet
        parts = f.split('-')
        if len(parts) >= 6:
            try:
                date = datetime(int(parts[3]), int(parts[4]), int(parts[5])).date()
                if start_date and date < start_date:
                    continue
                if end_date and date > end_date:
                    continue
                dated_files.append((date, f))
            except:
                continue

    # Sample evenly across the date range
    if len(dated_files) > sample_days:
        step = len(dated_files) // sample_days
        dated_files = dated_files[::step][:sample_days]

    print(f"    Processing {len(dated_files)} daily snapshots...")

    daily_flows = {}

    for i, (date, filename) in enumerate(dated_files):
        if i % 20 == 0:
            print(f"    Processing {i+1}/{len(dated_files)}: {date}")

        try:
            filepath = os.path.join(ORBITAAL_DIR, filename)
            df = pd.read_parquet(filepath)

            # Identify exchange transactions
            # INFLOW: DST_ID is in exchange_ids (money going TO exchange)
            # OUTFLOW: SRC_ID is in exchange_ids (money leaving FROM exchange)

            inflow_mask = df['DST_ID'].isin(exchange_ids)
            outflow_mask = df['SRC_ID'].isin(exchange_ids)

            # Calculate totals (convert satoshi to BTC)
            inflow_btc = df.loc[inflow_mask, 'VALUE_SATOSHI'].sum() / 1e8
            outflow_btc = df.loc[outflow_mask, 'VALUE_SATOSHI'].sum() / 1e8

            # Count significant transactions (100+ BTC)
            inflow_100btc = (df.loc[inflow_mask, 'VALUE_SATOSHI'] >= 100e8).sum()
            outflow_100btc = (df.loc[outflow_mask, 'VALUE_SATOSHI'] >= 100e8).sum()

            daily_flows[date] = {
                'inflow_btc': inflow_btc,
                'outflow_btc': outflow_btc,
                'net_flow': inflow_btc - outflow_btc,  # Positive = net inflow
                'inflow_100btc_count': inflow_100btc,
                'outflow_100btc_count': outflow_100btc,
                'total_tx': len(df)
            }

        except Exception as e:
            print(f"    Error processing {filename}: {e}")
            continue

    print(f"    Processed {len(daily_flows)} days with flow data")
    return daily_flows


def analyze_correlation(daily_flows, price_data):
    """Analyze correlation between flows and price movements."""
    print("[4/4] Analyzing flow-price correlation...")

    results = []

    for date, flows in daily_flows.items():
        if date not in price_data:
            continue

        price = price_data[date]

        results.append({
            'date': date,
            'inflow_btc': flows['inflow_btc'],
            'outflow_btc': flows['outflow_btc'],
            'net_flow': flows['net_flow'],
            'inflow_100btc': flows['inflow_100btc_count'],
            'outflow_100btc': flows['outflow_100btc_count'],
            'same_day_return': price['return'],
            'next_day_return': price.get('next_day_return', 0)
        })

    df = pd.DataFrame(results)

    if len(df) == 0:
        print("    ERROR: No overlapping data between flows and prices!")
        return None

    print(f"\n{'='*70}")
    print("FLOW-TO-PRICE CORRELATION ANALYSIS")
    print(f"{'='*70}")
    print(f"Data points: {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # Basic correlations
    print(f"\n--- CORRELATIONS ---")
    corr_inflow_same = df['inflow_btc'].corr(df['same_day_return'])
    corr_outflow_same = df['outflow_btc'].corr(df['same_day_return'])
    corr_net_same = df['net_flow'].corr(df['same_day_return'])
    corr_net_next = df['net_flow'].corr(df['next_day_return'])

    print(f"Inflow vs Same-Day Return:  {corr_inflow_same:+.4f}")
    print(f"Outflow vs Same-Day Return: {corr_outflow_same:+.4f}")
    print(f"Net Flow vs Same-Day Return: {corr_net_same:+.4f}")
    print(f"Net Flow vs Next-Day Return: {corr_net_next:+.4f}")

    # Hypothesis test: High inflow days should have negative returns
    print(f"\n--- HYPOTHESIS VALIDATION ---")
    print("Theory: INFLOW → Price DROP, OUTFLOW → Price RISE")

    # Categorize days
    median_net = df['net_flow'].median()
    high_inflow = df[df['net_flow'] > df['net_flow'].quantile(0.75)]
    high_outflow = df[df['net_flow'] < df['net_flow'].quantile(0.25)]

    high_inflow_return = high_inflow['same_day_return'].mean()
    high_outflow_return = high_outflow['same_day_return'].mean()

    print(f"\nHigh INFLOW days (top 25%, {len(high_inflow)} days):")
    print(f"  Average return: {high_inflow_return*100:+.3f}%")
    print(f"  Expected: NEGATIVE (price drops)")
    print(f"  Result: {'CONFIRMED' if high_inflow_return < 0 else 'REJECTED'}")

    print(f"\nHigh OUTFLOW days (top 25%, {len(high_outflow)} days):")
    print(f"  Average return: {high_outflow_return*100:+.3f}%")
    print(f"  Expected: POSITIVE (price rises)")
    print(f"  Result: {'CONFIRMED' if high_outflow_return > 0 else 'REJECTED'}")

    # Signal analysis: 100+ BTC transactions
    print(f"\n--- MEGA-FLOW SIGNAL ANALYSIS (100+ BTC) ---")

    mega_inflow_days = df[df['inflow_100btc'] > 0]
    mega_outflow_days = df[df['outflow_100btc'] > 0]

    if len(mega_inflow_days) > 0:
        mega_in_return = mega_inflow_days['same_day_return'].mean()
        print(f"Days with 100+ BTC INFLOWS ({len(mega_inflow_days)} days):")
        print(f"  Average return: {mega_in_return*100:+.3f}%")
        print(f"  Win rate for SHORT: {(mega_inflow_days['same_day_return'] < 0).mean()*100:.1f}%")

    if len(mega_outflow_days) > 0:
        mega_out_return = mega_outflow_days['same_day_return'].mean()
        print(f"\nDays with 100+ BTC OUTFLOWS ({len(mega_outflow_days)} days):")
        print(f"  Average return: {mega_out_return*100:+.3f}%")
        print(f"  Win rate for LONG: {(mega_outflow_days['same_day_return'] > 0).mean()*100:.1f}%")

    # Next-day prediction
    print(f"\n--- NEXT-DAY PREDICTION ---")
    if not df['next_day_return'].isna().all():
        next_day_df = df.dropna(subset=['next_day_return'])
        high_inflow_next = next_day_df[next_day_df['net_flow'] > next_day_df['net_flow'].quantile(0.75)]
        high_outflow_next = next_day_df[next_day_df['net_flow'] < next_day_df['net_flow'].quantile(0.25)]

        print(f"High INFLOW days → Next-day return: {high_inflow_next['next_day_return'].mean()*100:+.3f}%")
        print(f"High OUTFLOW days → Next-day return: {high_outflow_next['next_day_return'].mean()*100:+.3f}%")

    print(f"\n{'='*70}")

    return df


def main():
    print("="*70)
    print("HISTORICAL FLOW-PRICE CORRELATION VALIDATOR")
    print("="*70)
    print()

    # Load data
    exchange_ids, id_to_exchange = load_exchange_addresses()
    price_data = load_prices()

    # Get date range of price data
    price_dates = list(price_data.keys())
    min_price_date = min(price_dates)
    max_price_date = max(price_dates)

    print(f"\nPrice data range: {min_price_date} to {max_price_date}")
    print(f"Orbitaal data range: 2009-01-03 to 2021-01-25")
    print(f"Overlap period: {max(min_price_date, datetime(2009,1,3).date())} to {min(max_price_date, datetime(2021,1,25).date())}")

    # Process flows for overlapping period
    daily_flows = process_daily_flows(
        exchange_ids,
        start_date=min_price_date,
        end_date=datetime(2021, 1, 25).date(),
        sample_days=200  # Sample 200 days for faster processing
    )

    # Analyze correlation
    results_df = analyze_correlation(daily_flows, price_data)

    if results_df is not None:
        # Save results
        output_path = os.path.join(DATA_DIR, "data", "flow_correlation_results.csv")
        results_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")

    return results_df


if __name__ == "__main__":
    main()
