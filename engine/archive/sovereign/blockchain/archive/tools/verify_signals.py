#!/usr/bin/env python3
"""
SIGNAL VERIFICATION - Price check for LONG and SHORT
=====================================================
Uses address_clusters.db (same as working pipeline).
Verifies signals with 2-hour price check.
"""

import json
import sqlite3
import urllib.request
import base64
from datetime import datetime
from collections import defaultdict

RPC_USER = "bitcoin"
RPC_PASS = "bitcoin123secure"
RPC_HOST = "127.0.0.1"
RPC_PORT = 8332
DB_PATH = "/root/sovereign/address_clusters.db"

auth = base64.b64encode(f"{RPC_USER}:{RPC_PASS}".encode()).decode()

def rpc(method, params=None):
    payload = json.dumps({
        "jsonrpc": "1.0", "id": "verify", "method": method, "params": params or []
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

def load_addresses():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT address, exchange FROM addresses")
    addrs = {row[0]: row[1] for row in c.fetchall()}
    conn.close()
    print(f"Loaded {len(addrs):,} addresses")
    return addrs

def main():
    print("=" * 70)
    print("SIGNAL VERIFICATION WITH PRICE")
    print("=" * 70)

    addrs = load_addresses()

    height = rpc('getblockcount')
    if not height:
        print("Failed to get block height")
        return

    print(f"Current block: {height:,}")

    # Parameters
    verify_hours = 2
    buffer = verify_hours * 6 + 10  # blocks for verification
    test_blocks = 100

    start = height - buffer - test_blocks
    print(f"Testing blocks {start:,} to {start + test_blocks:,}")
    print()

    # Thresholds for signals
    MIN_SHORT_BTC = 10.0   # Minimum for SHORT signal
    MIN_LONG_BTC = 10.0    # Minimum for LONG signal

    short_signals = []
    long_signals = []

    for h in range(start, start + test_blocks):
        block_hash = rpc('getblockhash', [h])
        if not block_hash:
            continue

        block = rpc('getblock', [block_hash, 3])
        if not block:
            continue

        block_time = block.get('time', 0)

        # Track flows per exchange in this block
        inflows = defaultdict(float)
        outflows = defaultdict(float)

        for tx in block.get('tx', []):
            tx_in = defaultdict(float)
            tx_out = defaultdict(float)

            # Outputs = inflows
            for vout in tx.get('vout', []):
                addr = vout.get('scriptPubKey', {}).get('address')
                value = vout.get('value', 0)
                if addr and addr in addrs and value >= 0.01:
                    tx_in[addrs[addr]] += value

            # Inputs = outflows
            for vin in tx.get('vin', []):
                if 'coinbase' in vin:
                    continue
                prevout = vin.get('prevout', {})
                if prevout:
                    addr = prevout.get('scriptPubKey', {}).get('address')
                    value = prevout.get('value', 0)
                    if addr and addr in addrs and value >= 0.01:
                        tx_out[addrs[addr]] += value

            # Only external flows
            for ex in set(tx_in.keys()) | set(tx_out.keys()):
                if ex in tx_in and ex in tx_out:
                    continue  # Internal
                if ex in tx_in:
                    inflows[ex] += tx_in[ex]
                if ex in tx_out:
                    outflows[ex] += tx_out[ex]

        # Generate signals
        for ex in set(inflows.keys()) | set(outflows.keys()):
            inflow = inflows.get(ex, 0)
            outflow = outflows.get(ex, 0)

            # SHORT: large inflow (deposit = will sell)
            if inflow >= MIN_SHORT_BTC and inflow > outflow * 2:
                signal_ms = block_time * 1000
                later_ms = signal_ms + (verify_hours * 3600 * 1000)

                price_at = get_price(signal_ms)
                price_later = get_price(later_ms)

                if price_at and price_later:
                    price_change = (price_later - price_at) / price_at
                    correct = price_change < 0  # SHORT expects DOWN

                    short_signals.append({
                        'exchange': ex,
                        'btc': inflow,
                        'price_at': price_at,
                        'price_later': price_later,
                        'price_change': price_change,
                        'correct': correct
                    })

                    status = "CORRECT" if correct else "WRONG"
                    color = '\033[92m' if correct else '\033[91m'
                    reset = '\033[0m'
                    print(f"{color}[SHORT] {ex} {inflow:.1f} BTC | ${price_at:,.0f} -> ${price_later:,.0f} ({price_change*100:+.2f}%) {status}{reset}")

            # LONG: large outflow (withdrawal = supply leaving)
            if outflow >= MIN_LONG_BTC and outflow > inflow * 2:
                signal_ms = block_time * 1000
                later_ms = signal_ms + (verify_hours * 3600 * 1000)

                price_at = get_price(signal_ms)
                price_later = get_price(later_ms)

                if price_at and price_later:
                    price_change = (price_later - price_at) / price_at
                    correct = price_change > 0  # LONG expects UP

                    long_signals.append({
                        'exchange': ex,
                        'btc': outflow,
                        'price_at': price_at,
                        'price_later': price_later,
                        'price_change': price_change,
                        'correct': correct
                    })

                    status = "CORRECT" if correct else "WRONG"
                    color = '\033[92m' if correct else '\033[91m'
                    reset = '\033[0m'
                    print(f"{color}[LONG] {ex} {outflow:.1f} BTC | ${price_at:,.0f} -> ${price_later:,.0f} ({price_change*100:+.2f}%) {status}{reset}")

        if (h - start) % 20 == 0:
            print(f"  Block {h} done...")

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
    else:
        print("SHORT: No signals")

    if long_signals:
        correct = sum(1 for s in long_signals if s['correct'])
        total = len(long_signals)
        acc = correct / total * 100
        print(f"LONG: {correct}/{total} = {acc:.1f}%")
    else:
        print("LONG: No signals")

    # Combined
    all_signals = short_signals + long_signals
    if all_signals:
        correct = sum(1 for s in all_signals if s['correct'])
        total = len(all_signals)
        print(f"\nOVERALL: {correct}/{total} = {correct/total*100:.1f}%")

if __name__ == '__main__':
    main()
