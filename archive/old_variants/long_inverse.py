#!/usr/bin/env python3
"""
LONG = INVERSE OF SHORT - Pure Deterministic
=============================================
SHORT works at 100%: Large inflows → Price DOWN
LONG must be: ABSENCE of large inflows → Price UP

If no one is depositing to sell, price goes UP.
This is the only mathematically valid LONG signal from blockchain data.

FORMULA:
- SHORT: max_inflow >= 10 BTC in window → Price DOWN
- LONG: max_inflow < 1 BTC for N consecutive windows → Price UP
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
        "jsonrpc": "1.0", "id": "sig", "method": method, "params": params or []
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

def analyze_window_light(start_height, window_size, exchange_addrs):
    """
    Light analysis - only track INFLOWS (verbosity 2, no prevout).
    We only need to know MAX inflow to detect selling pressure.
    """
    max_inflow = 0
    total_inflow = 0
    inflow_count = 0
    window_timestamp = 0

    for h in range(start_height, start_height + window_size):
        block_hash = rpc('getblockhash', [h])
        if not block_hash:
            continue

        block = rpc('getblock', [block_hash, 2])  # Verbosity 2 - lighter
        if not block:
            continue

        if window_timestamp == 0:
            window_timestamp = block.get('time', 0)

        for tx in block.get('tx', []):
            for vout in tx.get('vout', []):
                addr = vout.get('scriptPubKey', {}).get('address')
                value = vout.get('value', 0)
                if addr and addr in exchange_addrs and value >= 0.01:
                    total_inflow += value
                    inflow_count += 1
                    if value > max_inflow:
                        max_inflow = value

    return {
        'max_inflow': max_inflow,
        'total_inflow': total_inflow,
        'inflow_count': inflow_count,
        'timestamp': window_timestamp
    }

def main():
    print("=" * 70)
    print("LONG = INVERSE OF SHORT - Deterministic Test")
    print("=" * 70)
    print()
    print("HYPOTHESIS:")
    print("  SHORT: max_inflow >= 10 BTC → Someone selling big → Price DOWN")
    print("  LONG: max_inflow < 1 BTC for 3+ windows → No sellers → Price UP")
    print()

    exchange_addrs = load_exchange_addresses()

    height = rpc('getblockcount')
    if not height:
        print("Failed to get block height")
        return

    print(f"Current block: {height:,}")

    # Parameters
    window_size = 10  # ~100 minutes per window
    verify_hours = 2
    blocks_per_hour = 6
    buffer = verify_hours * blocks_per_hour + 10

    num_windows = 50
    start = height - buffer - (num_windows * window_size)

    print(f"Window size: {window_size} blocks (~{window_size*10} minutes)")
    print(f"Testing {num_windows} windows from block {start:,}")
    print()

    # Track windows
    windows = []
    no_seller_streak = 0

    # Results
    short_signals = []
    long_signals = []

    for i in range(num_windows):
        window_start = start + (i * window_size)

        metrics = analyze_window_light(window_start, window_size, exchange_addrs)

        if metrics['timestamp'] == 0:
            continue

        # Get price change
        signal_ms = metrics['timestamp'] * 1000
        later_ms = signal_ms + (verify_hours * 3600 * 1000)

        price_at = get_price(signal_ms)
        price_later = get_price(later_ms)

        if not price_at or not price_later:
            print(f"Window {i+1}: max_in={metrics['max_inflow']:.2f} total={metrics['total_inflow']:.1f} (no price)")
            continue

        price_change = (price_later - price_at) / price_at

        # Classify signal
        max_in = metrics['max_inflow']

        if max_in >= 10.0:
            # SHORT signal - big seller detected
            no_seller_streak = 0
            signal = "SHORT"
            correct = price_change < 0
            short_signals.append({
                'max_inflow': max_in,
                'price_change': price_change,
                'correct': correct
            })
        elif max_in < 1.0:
            # Potential LONG - no significant sellers
            no_seller_streak += 1
            if no_seller_streak >= 3:
                signal = "LONG"
                correct = price_change > 0
                long_signals.append({
                    'max_inflow': max_in,
                    'streak': no_seller_streak,
                    'price_change': price_change,
                    'correct': correct
                })
            else:
                signal = f"quiet({no_seller_streak})"
                correct = None
        else:
            # Between 1-10 BTC - no signal
            no_seller_streak = 0
            signal = "---"
            correct = None

        # Color output
        if correct is True:
            color = '\033[92m'  # Green
            result = "CORRECT"
        elif correct is False:
            color = '\033[91m'  # Red
            result = "WRONG"
        else:
            color = '\033[0m'
            result = ""
        reset = '\033[0m'

        print(f"{color}Window {i+1}: max_in={max_in:.2f} BTC | "
              f"${price_at:,.0f} → ${price_later:,.0f} ({price_change*100:+.2f}%) | "
              f"{signal} {result}{reset}")

    # Summary
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    if short_signals:
        correct = sum(1 for s in short_signals if s['correct'])
        total = len(short_signals)
        acc = correct / total * 100
        print(f"SHORT: {correct}/{total} = {acc:.1f}%")
        for s in short_signals:
            status = "OK" if s['correct'] else "WRONG"
            print(f"  max={s['max_inflow']:.1f} BTC, price={s['price_change']*100:+.2f}% [{status}]")
    else:
        print("SHORT: No signals")

    print()

    if long_signals:
        correct = sum(1 for s in long_signals if s['correct'])
        total = len(long_signals)
        acc = correct / total * 100
        print(f"LONG: {correct}/{total} = {acc:.1f}%")
        for s in long_signals:
            status = "OK" if s['correct'] else "WRONG"
            print(f"  streak={s['streak']}, max={s['max_inflow']:.2f} BTC, price={s['price_change']*100:+.2f}% [{status}]")
    else:
        print("LONG: No signals")

    print()
    total_signals = len(short_signals) + len(long_signals)
    total_correct = sum(1 for s in short_signals if s['correct']) + sum(1 for s in long_signals if s['correct'])
    if total_signals > 0:
        print(f"OVERALL: {total_correct}/{total_signals} = {total_correct/total_signals*100:.1f}%")

if __name__ == '__main__':
    main()
