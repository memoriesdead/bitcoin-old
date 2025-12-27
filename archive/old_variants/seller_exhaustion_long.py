#!/usr/bin/env python3
"""
SELLER EXHAUSTION LONG SIGNAL
=============================
Based on CryptoQuant Exchange Netflow methodology.
Source: https://userguide.cryptoquant.com/cryptoquant-metrics/exchange/exchange-in-outflow-and-netflow

LOGIC:
    SHORT works because: Large inflow → Person WILL sell → Price DOWN
    LONG must be: NO large inflows + sustained outflows → No sellers → Price UP

FORMULA:
    1. Track rolling average inflow per window
    2. Current inflow << rolling average = seller exhaustion
    3. Combined with: max_single_inflow < threshold
    4. Combined with: cumulative netflow is negative (outflows > inflows)

THRESHOLDS (conservative for 100%):
    - No single inflow > 5 BTC in current window
    - Current window inflow < 50% of rolling average
    - Netflow is negative (more outflows)
"""

import json
import sqlite3
import urllib.request
import base64
from datetime import datetime
from collections import deque

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
    except:
        return None

def get_price(timestamp_ms):
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
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT address, exchange FROM addresses")
    addrs = {row[0]: row[1] for row in c.fetchall()}
    conn.close()
    print(f"Loaded {len(addrs):,} exchange addresses")
    return addrs

def analyze_window(start_height, window_size, exchange_addrs):
    """
    Light analysis - verbosity 2 (faster).
    Track inflows and outflows.
    """
    total_inflow = 0
    total_outflow = 0
    max_single_inflow = 0
    inflow_count = 0
    outflow_count = 0
    window_timestamp = 0

    for h in range(start_height, start_height + window_size):
        block_hash = rpc('getblockhash', [h])
        if not block_hash:
            continue

        block = rpc('getblock', [block_hash, 2])
        if not block:
            continue

        if window_timestamp == 0:
            window_timestamp = block.get('time', 0)

        for tx in block.get('tx', []):
            # Track inflows (deposits)
            for vout in tx.get('vout', []):
                addr = vout.get('scriptPubKey', {}).get('address')
                value = vout.get('value', 0)
                if addr and addr in exchange_addrs and value >= 0.01:
                    total_inflow += value
                    inflow_count += 1
                    if value > max_single_inflow:
                        max_single_inflow = value

            # Track outflows (withdrawals) - need verbosity 3 for this
            # Skip for now - we'll focus on inflow exhaustion

    return {
        'total_inflow': total_inflow,
        'max_single_inflow': max_single_inflow,
        'inflow_count': inflow_count,
        'timestamp': window_timestamp
    }

def main():
    print("=" * 70)
    print("SELLER EXHAUSTION LONG SIGNAL")
    print("=" * 70)
    print()
    print("LOGIC:")
    print("  SHORT = Large inflow detected → selling coming → Price DOWN")
    print("  LONG = Inflow BELOW average for 3+ windows → sellers exhausted → Price UP")
    print()
    print("CONDITIONS FOR LONG:")
    print("  1. No single inflow > 5 BTC")
    print("  2. Total inflow < 50% of rolling average")
    print("  3. Sustained for 3+ consecutive windows")
    print()

    exchange_addrs = load_exchange_addresses()

    height = rpc('getblockcount')
    if not height:
        print("Failed to get block height")
        return

    print(f"Current block: {height:,}")

    # Parameters
    window_size = 10  # ~100 minutes
    verify_hours = 2
    blocks_per_hour = 6
    buffer = verify_hours * blocks_per_hour + 10
    rolling_window = 10  # Use last 10 windows for average

    num_windows = 40
    start = height - buffer - (num_windows * window_size)

    print(f"Window size: {window_size} blocks (~{window_size*10} minutes)")
    print(f"Testing {num_windows} windows from block {start:,}")
    print()

    # Track history for rolling average
    inflow_history = deque(maxlen=rolling_window)
    low_inflow_streak = 0

    signals = []

    for i in range(num_windows):
        window_start = start + (i * window_size)

        metrics = analyze_window(window_start, window_size, exchange_addrs)

        if metrics['timestamp'] == 0:
            continue

        total_inflow = metrics['total_inflow']
        max_inflow = metrics['max_single_inflow']

        # Calculate rolling average
        if len(inflow_history) >= 5:
            avg_inflow = sum(inflow_history) / len(inflow_history)
        else:
            avg_inflow = total_inflow + 1  # First windows, no signal

        inflow_history.append(total_inflow)

        # Get price change
        signal_ms = metrics['timestamp'] * 1000
        later_ms = signal_ms + (verify_hours * 3600 * 1000)

        price_at = get_price(signal_ms)
        price_later = get_price(later_ms)

        if not price_at or not price_later:
            print(f"Window {i+1}: in={total_inflow:.1f} max={max_inflow:.1f} avg={avg_inflow:.1f} (no price)")
            continue

        price_change = (price_later - price_at) / price_at

        # Check conditions
        no_whale = max_inflow < 5.0
        below_avg = total_inflow < (avg_inflow * 0.5) if avg_inflow > 0 else False

        if no_whale and below_avg:
            low_inflow_streak += 1
        else:
            low_inflow_streak = 0

        # Generate signal
        if low_inflow_streak >= 3:
            signal = "LONG"
            correct = price_change > 0
            signals.append({
                'streak': low_inflow_streak,
                'inflow': total_inflow,
                'avg': avg_inflow,
                'price_change': price_change,
                'correct': correct
            })
            color = '\033[92m' if correct else '\033[91m'
            status = "CORRECT" if correct else "WRONG"
        elif max_inflow >= 10.0:
            signal = "SHORT?"
            color = '\033[93m'
            status = ""
            correct = None
        else:
            signal = f"streak={low_inflow_streak}"
            color = '\033[0m'
            status = ""
            correct = None

        reset = '\033[0m'
        ratio = (total_inflow / avg_inflow * 100) if avg_inflow > 0 else 100

        print(f"{color}Window {i+1}: in={total_inflow:.1f} max={max_inflow:.1f} "
              f"avg={avg_inflow:.1f} ({ratio:.0f}%) | "
              f"${price_at:,.0f} → ${price_later:,.0f} ({price_change*100:+.2f}%) "
              f"{signal} {status}{reset}")

    # Summary
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    if signals:
        correct = sum(1 for s in signals if s['correct'])
        total = len(signals)
        acc = correct / total * 100
        print(f"LONG signals: {total}")
        print(f"Accuracy: {correct}/{total} = {acc:.1f}%")
        print()
        for s in signals:
            status = "OK" if s['correct'] else "WRONG"
            print(f"  streak={s['streak']} in={s['inflow']:.1f} avg={s['avg']:.1f} "
                  f"→ price {s['price_change']*100:+.2f}% [{status}]")
    else:
        print("No LONG signals generated")
        print("Conditions not met: Either inflows were high or no sustained exhaustion pattern")

if __name__ == '__main__':
    main()
