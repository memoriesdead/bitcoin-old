#!/usr/bin/env python3
"""
LONG FORMULA - Pure Deterministic
==================================
SHORT works: High inflows = selling = price DOWN
LONG must be: Low inflows = NO selling = price UP

The signal is NOT outflows. It's ABSENCE OF INFLOWS.

FORMULA:
- Measure inflow rate over window
- Compare to rolling average
- If current << average = sellers exhausted = LONG
"""

import json
import sqlite3
import urllib.request
import base64
from datetime import datetime
from collections import defaultdict, deque

# RPC config
RPC_USER = "bitcoin"
RPC_PASS = "bitcoin123secure"
RPC_HOST = "127.0.0.1"
RPC_PORT = 8332
DB_PATH = "/root/sovereign/walletexplorer_addresses.db"

auth = base64.b64encode(f"{RPC_USER}:{RPC_PASS}".encode()).decode()

def rpc(method, params=None):
    payload = json.dumps({
        "jsonrpc": "1.0", "id": "long", "method": method, "params": params or []
    }).encode()
    try:
        req = urllib.request.Request(f"http://{RPC_HOST}:{RPC_PORT}")
        req.add_header("Authorization", f"Basic {auth}")
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, payload, timeout=60) as resp:
            return json.loads(resp.read()).get('result')
    except Exception as e:
        return None

def get_price(timestamp_ms):
    """Get BTC price at timestamp."""
    try:
        url = f"https://api.binance.us/api/v3/klines?symbol=BTCUSD&interval=1m&startTime={timestamp_ms}&limit=1"
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "Mozilla/5.0")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            if data:
                return float(data[0][4])
    except:
        pass
    return None

def load_exchange_addresses():
    """Load exchange addresses."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT address, exchange FROM addresses")
    addrs = {row[0]: row[1] for row in c.fetchall()}
    conn.close()
    print(f"Loaded {len(addrs):,} exchange addresses")
    return addrs

def analyze_window(start_height, window_size, exchange_addrs):
    """
    Analyze a window and return INFLOW metrics.

    KEY INSIGHT:
    - Whale deposits (>1 BTC) = selling pressure = bearish
    - Only small deposits = no whales selling = bullish
    - NO deposits at all = exhausted sellers = bullish

    We measure:
    1. Whale inflows (>1 BTC each)
    2. Retail inflows (<1 BTC each)
    3. Max single inflow (whale detection)
    """
    whale_inflow_btc = 0  # >1 BTC transactions
    retail_inflow_btc = 0  # <1 BTC transactions
    whale_count = 0
    retail_count = 0
    max_inflow = 0
    window_timestamp = 0
    all_inflows = []

    for h in range(start_height, start_height + window_size):
        block_hash = rpc('getblockhash', [h])
        if not block_hash:
            continue

        # Use verbosity 2 (faster, no prevout)
        block = rpc('getblock', [block_hash, 2])
        if not block:
            continue

        if window_timestamp == 0:
            window_timestamp = block.get('time', 0)

        for tx in block.get('tx', []):
            # Check outputs for exchange addresses (INFLOWS)
            for vout in tx.get('vout', []):
                addr = vout.get('scriptPubKey', {}).get('address')
                value = vout.get('value', 0)
                if addr and addr in exchange_addrs and value >= 0.01:
                    all_inflows.append(value)
                    if value >= 1.0:  # WHALE
                        whale_inflow_btc += value
                        whale_count += 1
                    else:  # RETAIL
                        retail_inflow_btc += value
                        retail_count += 1
                    if value > max_inflow:
                        max_inflow = value

    return {
        'whale_btc': whale_inflow_btc,
        'retail_btc': retail_inflow_btc,
        'whale_count': whale_count,
        'retail_count': retail_count,
        'max_inflow': max_inflow,
        'total_btc': whale_inflow_btc + retail_inflow_btc,
        'timestamp': window_timestamp,
        'all_inflows': all_inflows
    }

def main():
    print("=" * 70)
    print("LONG FORMULA TEST - Inflow Exhaustion")
    print("=" * 70)
    print()
    print("HYPOTHESIS:")
    print("  WHALE deposits (>10 BTC) = big seller = price DOWN = SHORT")
    print("  ZERO whales for 3+ windows = sellers exhausted = LONG")
    print()
    print("FORMULA:")
    print("  SHORT = Single large whale deposit (>10 BTC max)")
    print("  LONG = No whale deposits for 3 consecutive windows")
    print()

    exchange_addrs = load_exchange_addresses()

    height = rpc('getblockcount')
    if not height:
        print("Failed to get block height")
        return

    print(f"Current block: {height:,}")
    print()

    # Parameters - LARGER for stronger signals
    window_size = 50   # ~8 hours per window (same as original SHORT)
    verify_hours = 4   # 4 hour verification (same as original SHORT)
    blocks_per_hour = 6
    buffer = verify_hours * blocks_per_hour + 10

    # Analyze multiple windows to build rolling average
    num_windows = 50   # Fewer but stronger windows
    start = height - buffer - (num_windows * window_size)

    print(f"Analyzing {num_windows} windows of {window_size} blocks")
    print(f"From block {start:,} to {height - buffer:,}")
    print()

    # Collect all window data
    windows = []

    for i in range(num_windows):
        window_start = start + (i * window_size)

        metrics = analyze_window(window_start, window_size, exchange_addrs)

        if metrics['timestamp'] == 0:
            continue

        # Get price change
        signal_ms = metrics['timestamp'] * 1000
        later_ms = signal_ms + (verify_hours * 3600 * 1000)

        price_at = get_price(signal_ms)
        price_later = get_price(later_ms)

        if not price_at or not price_later:
            continue

        price_change = (price_later - price_at) / price_at
        metrics['price_change'] = price_change
        metrics['price_at'] = price_at
        metrics['price_later'] = price_later

        windows.append(metrics)

        direction = "UP" if price_change > 0 else "DOWN"
        whale_marker = "WHALE" if metrics['whale_count'] > 0 else "retail"
        print(f"Window {i+1}: whale={metrics['whale_btc']:.1f}({metrics['whale_count']}) retail={metrics['retail_btc']:.1f}({metrics['retail_count']}) max={metrics['max_inflow']:.1f} | price {price_change*100:+.2f}% [{direction}] {whale_marker}")

    if len(windows) < 10:
        print("Not enough data")
        return

    # Count whales vs retail only windows
    whale_windows = [w for w in windows if w['whale_count'] > 0]
    retail_only = [w for w in windows if w['whale_count'] == 0 and w['retail_count'] > 0]

    print()
    print(f"Windows with whale deposits (>1 BTC): {len(whale_windows)}")
    print(f"Windows with retail only (<1 BTC): {len(retail_only)}")
    print()

    # Test the formula
    print("=" * 70)
    print("TESTING FORMULA")
    print("=" * 70)
    print()
    print("SHORT = Single whale deposit >10 BTC")
    print("LONG = No whales for 3+ consecutive windows")
    print()

    correct_long = 0
    total_long = 0
    correct_short = 0
    total_short = 0

    # Track consecutive no-whale windows
    no_whale_streak = 0

    for i, w in enumerate(windows):
        price_change = w['price_change']
        whale_count = w['whale_count']
        whale_btc = w['whale_btc']
        max_inflow = w['max_inflow']

        # Check for whale activity
        has_big_whale = max_inflow >= 10.0  # Single deposit >10 BTC

        if has_big_whale:
            no_whale_streak = 0  # Reset streak
            signal = "SHORT"
            total_short += 1
            if price_change < 0:
                correct_short += 1
                result = "CORRECT"
            else:
                result = "WRONG"
            detail = f"BIG WHALE max={max_inflow:.1f}BTC"
            color = '\033[92m' if 'CORRECT' in result else '\033[91m'
            reset = '\033[0m'
            print(f"{color}[{signal}] {detail} | price {price_change*100:+.2f}% | {result}{reset}")
        else:
            no_whale_streak += 1
            # LONG signal if no whales for 3+ consecutive windows
            if no_whale_streak >= 3:
                signal = "LONG"
                total_long += 1
                if price_change > 0:
                    correct_long += 1
                    result = "CORRECT"
                else:
                    result = "WRONG"
                detail = f"NO WHALES x{no_whale_streak} windows"
                color = '\033[92m' if 'CORRECT' in result else '\033[91m'
                reset = '\033[0m'
                print(f"{color}[{signal}] {detail} | price {price_change*100:+.2f}% | {result}{reset}")

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    if total_long > 0:
        long_acc = correct_long / total_long * 100
        print(f"LONG:  {correct_long}/{total_long} = {long_acc:.1f}%")
    else:
        print("LONG:  No signals")

    if total_short > 0:
        short_acc = correct_short / total_short * 100
        print(f"SHORT: {correct_short}/{total_short} = {short_acc:.1f}%")
    else:
        print("SHORT: No signals")

    total = total_long + total_short
    correct = correct_long + correct_short
    if total > 0:
        overall = correct / total * 100
        print()
        print(f"OVERALL: {correct}/{total} = {overall:.1f}%")

if __name__ == '__main__':
    main()
