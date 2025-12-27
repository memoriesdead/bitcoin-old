"""
Fast Correlation Analysis - Sample 100 days from 2020-2021
"""
import os
import sqlite3
import pandas as pd
import numpy as np

DATA_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = os.path.join(DATA_DIR, "data")
ORBITAAL_DIR = os.path.join(DATA_DIR, "orbitaal", "SNAPSHOT", "EDGES", "day")
ENTITY_DIR = os.path.join(DATA_DIR, "entity_addresses", "data")

print("="*70)
print("FAST CORRELATION ANALYSIS (100-day sample)")
print("="*70)

# Load exchange addresses
print("[1/4] Loading exchange addresses...")
exchanges_file = os.path.join(ENTITY_DIR, "Exchanges_full_detailed.csv")
df = pd.read_csv(exchanges_file, usecols=["add_num"])
exchange_ids = set(df["add_num"].values)
print(f"      Loaded {len(exchange_ids):,} exchange addresses")

# Load prices
print("[2/4] Loading prices...")
db_path = os.path.join(DATA_DIR, "unified_bitcoin.db")
conn = sqlite3.connect(db_path)
prices_df = pd.read_sql("SELECT date, open, close FROM prices ORDER BY date", conn)
conn.close()
prices_df["change"] = (prices_df["close"] - prices_df["open"]) / prices_df["open"]
prices_df["next_day_change"] = prices_df["change"].shift(-1)
prices = prices_df.set_index("date").to_dict("index")
print(f"      Loaded {len(prices):,} daily prices")

# Get orbitaal files
files = sorted([f for f in os.listdir(ORBITAAL_DIR) if f.endswith(".parquet")])
print(f"      Found {len(files)} orbitaal daily files")

# Focus on 2020-2021 period, sample every 4th day
target_files = []
for f in files:
    parts = f.split("-")
    if len(parts) >= 6:
        date_str = f"{parts[3]}-{parts[4]}-{parts[5]}"
        if date_str >= "2020-01-01" and date_str <= "2021-01-25":
            if date_str in prices:
                target_files.append((date_str, f))

# Sample every 4th day to get ~100 days
target_files = target_files[::4][:100]
print(f"      Sampling {len(target_files)} days from 2020-2021")

# Process
print("[3/4] Processing sampled days...")
results = []

for i, (date_str, filename) in enumerate(target_files):
    filepath = os.path.join(ORBITAAL_DIR, filename)
    try:
        df = pd.read_parquet(filepath)

        # Identify exchange transactions
        inflow_mask = df["DST_ID"].isin(exchange_ids)
        outflow_mask = df["SRC_ID"].isin(exchange_ids)
        internal_mask = inflow_mask & outflow_mask

        inflows = df[inflow_mask & ~internal_mask]
        outflows = df[outflow_mask & ~internal_mask]

        inflow_btc = inflows["VALUE_SATOSHI"].sum() / 1e8
        outflow_btc = outflows["VALUE_SATOSHI"].sum() / 1e8
        net_flow = inflow_btc - outflow_btc

        # Count 100+ BTC flows
        inflow_100btc = (inflows["VALUE_SATOSHI"] >= 100e8).sum()
        outflow_100btc = (outflows["VALUE_SATOSHI"] >= 100e8).sum()

        price_info = prices[date_str]

        results.append({
            "date": date_str,
            "inflow_btc": inflow_btc,
            "outflow_btc": outflow_btc,
            "net_flow": net_flow,
            "inflow_100btc": inflow_100btc,
            "outflow_100btc": outflow_100btc,
            "price_change": price_info.get("change", 0),
            "next_day_change": price_info.get("next_day_change", None)
        })

        if i % 20 == 0:
            print(f"      Processed {i+1}/{len(target_files)}: {date_str}")

    except Exception as e:
        print(f"      Error on {date_str}: {e}")
        continue

print(f"      Got {len(results)} valid days")

# Analyze
print()
print("="*70)
print("[4/4] CORRELATION RESULTS")
print("="*70)

df = pd.DataFrame(results)
print(f"Data points: {len(df)}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Total INFLOW: {df['inflow_btc'].sum():,.0f} BTC")
print(f"Total OUTFLOW: {df['outflow_btc'].sum():,.0f} BTC")

# Same-day correlation
print()
print("--- SAME-DAY CORRELATION ---")
corr_net = df["net_flow"].corr(df["price_change"])
print(f"Net Flow vs Price Change: {corr_net:+.4f}")
print("  (Negative = CONFIRMS: net inflow -> price drop)")

# High inflow/outflow analysis
q75 = df["net_flow"].quantile(0.75)
q25 = df["net_flow"].quantile(0.25)

high_inflow = df[df["net_flow"] > q75]
high_outflow = df[df["net_flow"] < q25]

print()
print(f"HIGH NET INFLOW days (top 25%, {len(high_inflow)} days):")
print(f"  Average price change: {high_inflow['price_change'].mean()*100:+.3f}%")
win_rate_inflow = (high_inflow["price_change"] < 0).mean()
print(f"  Win rate (SHORT): {win_rate_inflow*100:.1f}%")
if win_rate_inflow > 0.52:
    print("  CONFIRMED")
else:
    print("  NOT CONFIRMED")

print()
print(f"HIGH NET OUTFLOW days (top 25%, {len(high_outflow)} days):")
print(f"  Average price change: {high_outflow['price_change'].mean()*100:+.3f}%")
win_rate_outflow = (high_outflow["price_change"] > 0).mean()
print(f"  Win rate (LONG): {win_rate_outflow*100:.1f}%")
if win_rate_outflow > 0.52:
    print("  CONFIRMED")
else:
    print("  NOT CONFIRMED")

# Next-day prediction
print()
print("--- NEXT-DAY PREDICTION ---")
df_next = df.dropna(subset=["next_day_change"])
if len(df_next) > 0:
    corr_next = df_next["net_flow"].corr(df_next["next_day_change"])
    print(f"Net Flow vs NEXT-DAY Price Change: {corr_next:+.4f}")

    high_inflow_next = df_next[df_next["net_flow"] > df_next["net_flow"].quantile(0.75)]
    high_outflow_next = df_next[df_next["net_flow"] < df_next["net_flow"].quantile(0.25)]

    print(f"High NET INFLOW -> Next day: {high_inflow_next['next_day_change'].mean()*100:+.3f}%")
    print(f"High NET OUTFLOW -> Next day: {high_outflow_next['next_day_change'].mean()*100:+.3f}%")

# 100+ BTC flows analysis
print()
print("--- MEGA-FLOW ANALYSIS (100+ BTC transactions) ---")

mega_inflow_days = df[df["inflow_100btc"] > 0]
mega_outflow_days = df[df["outflow_100btc"] > 0]

if len(mega_inflow_days) >= 5:
    avg_change = mega_inflow_days["price_change"].mean()
    wr = (mega_inflow_days["price_change"] < 0).mean()
    print(f"Days with 100+ BTC INFLOWS ({len(mega_inflow_days)} days):")
    print(f"  Avg price change: {avg_change*100:+.3f}%")
    print(f"  Win rate (SHORT): {wr*100:.1f}%")

if len(mega_outflow_days) >= 5:
    avg_change = mega_outflow_days["price_change"].mean()
    wr = (mega_outflow_days["price_change"] > 0).mean()
    print(f"Days with 100+ BTC OUTFLOWS ({len(mega_outflow_days)} days):")
    print(f"  Avg price change: {avg_change*100:+.3f}%")
    print(f"  Win rate (LONG): {wr*100:.1f}%")

# Final verdict
print()
print("="*70)
print("VERDICT")
print("="*70)
if win_rate_inflow > 0.52 and win_rate_outflow > 0.52:
    print("EDGE CONFIRMED")
    print(f"  Inflow SHORT win rate: {win_rate_inflow*100:.1f}%")
    print(f"  Outflow LONG win rate: {win_rate_outflow*100:.1f}%")
    print()
    print("  -> Safe to trade with blockchain flow signals")
elif win_rate_inflow > 0.52:
    print("PARTIAL EDGE: Only INFLOW->SHORT confirmed")
    print(f"  Win rate: {win_rate_inflow*100:.1f}%")
    print()
    print("  -> Only trade SHORT signals")
elif win_rate_outflow > 0.52:
    print("PARTIAL EDGE: Only OUTFLOW->LONG confirmed")
    print(f"  Win rate: {win_rate_outflow*100:.1f}%")
    print()
    print("  -> Only trade LONG signals")
else:
    print("NO EDGE DETECTED")
    print(f"  Inflow SHORT win rate: {win_rate_inflow*100:.1f}%")
    print(f"  Outflow LONG win rate: {win_rate_outflow*100:.1f}%")
    print()
    print("  -> DO NOT TRADE with this strategy")

print("="*70)
