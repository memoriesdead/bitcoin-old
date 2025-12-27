#!/usr/bin/env python3
"""
SOPR-BASED LONG SIGNAL - Academic Formula
==========================================
Based on: CryptoQuant SOPR methodology
Source: https://userguide.cryptoquant.com/utxo-data-indicators/spent-output-profit-ratio-sopr

FORMULA:
    SOPR = Σ(value × price_spent) / Σ(value × price_created)

SIGNAL LOGIC:
    SOPR < 1.0 = Sellers at LOSS = Capitulation
    SOPR < 0.95 = Significant capitulation = LONG signal

WHY THIS WORKS:
    When SOPR < 1, people are selling at a LOSS.
    This indicates panic/capitulation = market bottom.
    Smart money buys when weak hands are selling at a loss.

IMPLEMENTATION:
    For each spent UTXO:
    1. Get creation block → timestamp → historical price
    2. Get spend block → timestamp → current price
    3. Calculate: spent_value / created_value
    4. Aggregate across all spends in window
"""

import json
import sqlite3
import urllib.request
import base64
from datetime import datetime
from collections import defaultdict

# RPC config
RPC_USER = "bitcoin"
RPC_PASS = "bitcoin123secure"
RPC_HOST = "127.0.0.1"
RPC_PORT = 8332
DB_PATH = "/root/sovereign/walletexplorer_addresses.db"

auth = base64.b64encode(f"{RPC_USER}:{RPC_PASS}".encode()).decode()

# Price cache to avoid repeated API calls
PRICE_CACHE = {}

def rpc(method, params=None):
    payload = json.dumps({
        "jsonrpc": "1.0", "id": "sopr", "method": method, "params": params or []
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
    """Get BTC price at timestamp with caching."""
    # Round to minute for caching
    cache_key = timestamp_ms // 60000 * 60000

    if cache_key in PRICE_CACHE:
        return PRICE_CACHE[cache_key]

    try:
        url = f"https://api.binance.us/api/v3/klines?symbol=BTCUSD&interval=1m&startTime={cache_key}&limit=1"
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "Mozilla/5.0")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            if data:
                price = float(data[0][4])
                PRICE_CACHE[cache_key] = price
                return price
    except:
        pass
    return None

def get_block_timestamp(block_hash):
    """Get timestamp for a block."""
    block = rpc('getblockheader', [block_hash])
    if block:
        return block.get('time', 0)
    return 0

def load_exchange_addresses():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT address, exchange FROM addresses")
    addrs = {row[0]: row[1] for row in c.fetchall()}
    conn.close()
    print(f"Loaded {len(addrs):,} exchange addresses")
    return addrs

def calculate_window_sopr(start_height, window_size, exchange_addrs):
    """
    Calculate SOPR for a window of blocks.

    SOPR = Σ(value × price_now) / Σ(value × price_created)

    We track SPENT outputs going TO exchanges (deposits = selling).
    If sellers are selling at a loss (SOPR < 1), it's capitulation.
    """
    total_realized_value = 0  # value × price_spent
    total_created_value = 0   # value × price_created
    spend_count = 0
    window_timestamp = 0

    for h in range(start_height, start_height + window_size):
        block_hash = rpc('getblockhash', [h])
        if not block_hash:
            continue

        block = rpc('getblock', [block_hash, 3])  # Need prevout for SOPR
        if not block:
            continue

        if window_timestamp == 0:
            window_timestamp = block.get('time', 0)

        current_price = get_price(block.get('time', 0) * 1000)
        if not current_price:
            continue

        for tx in block.get('tx', []):
            # Check inputs (spent outputs)
            for vin in tx.get('vin', []):
                if 'coinbase' in vin:
                    continue

                prevout = vin.get('prevout', {})
                if not prevout:
                    continue

                value = prevout.get('value', 0)
                if value < 0.01:  # Skip dust
                    continue

                # Check if this spend is going TO an exchange (deposit = selling)
                is_exchange_deposit = False
                for vout in tx.get('vout', []):
                    addr = vout.get('scriptPubKey', {}).get('address')
                    if addr and addr in exchange_addrs:
                        is_exchange_deposit = True
                        break

                if not is_exchange_deposit:
                    continue  # Only care about deposits to exchanges

                # Get creation block info
                txid = vin.get('txid')
                vout_n = vin.get('vout', 0)

                # Get the original transaction to find creation time
                orig_tx = rpc('getrawtransaction', [txid, True])
                if not orig_tx:
                    continue

                orig_block_hash = orig_tx.get('blockhash')
                if not orig_block_hash:
                    continue

                created_timestamp = get_block_timestamp(orig_block_hash)
                if not created_timestamp:
                    continue

                created_price = get_price(created_timestamp * 1000)
                if not created_price:
                    continue

                # Calculate SOPR contribution
                realized = value * current_price
                created = value * created_price

                total_realized_value += realized
                total_created_value += created
                spend_count += 1

    # Calculate SOPR
    if total_created_value > 0:
        sopr = total_realized_value / total_created_value
    else:
        sopr = 1.0  # Neutral if no data

    return {
        'sopr': sopr,
        'spend_count': spend_count,
        'realized_usd': total_realized_value,
        'created_usd': total_created_value,
        'timestamp': window_timestamp
    }

def main():
    print("=" * 70)
    print("SOPR-BASED LONG SIGNAL - Academic Formula")
    print("=" * 70)
    print()
    print("FORMULA: SOPR = Σ(value × price_spent) / Σ(value × price_created)")
    print()
    print("SIGNAL LOGIC:")
    print("  SOPR > 1.0  = Sellers at PROFIT (neutral/bearish)")
    print("  SOPR < 1.0  = Sellers at LOSS (capitulation)")
    print("  SOPR < 0.95 = Significant capitulation = LONG signal")
    print()

    exchange_addrs = load_exchange_addresses()

    height = rpc('getblockcount')
    if not height:
        print("Failed to get block height")
        return

    print(f"Current block: {height:,}")

    # Parameters
    window_size = 6  # ~1 hour per window
    verify_hours = 2
    blocks_per_hour = 6
    buffer = verify_hours * blocks_per_hour + 10

    num_windows = 20
    start = height - buffer - (num_windows * window_size)

    print(f"Window size: {window_size} blocks (~1 hour)")
    print(f"Testing {num_windows} windows from block {start:,}")
    print()

    signals = []

    for i in range(num_windows):
        window_start = start + (i * window_size)

        print(f"Analyzing window {i+1}/{num_windows}...", end=" ", flush=True)

        metrics = calculate_window_sopr(window_start, window_size, exchange_addrs)

        if metrics['timestamp'] == 0 or metrics['spend_count'] == 0:
            print("no data")
            continue

        sopr = metrics['sopr']

        # Get price change for verification
        signal_ms = metrics['timestamp'] * 1000
        later_ms = signal_ms + (verify_hours * 3600 * 1000)

        price_at = get_price(signal_ms)
        price_later = get_price(later_ms)

        if not price_at or not price_later:
            print(f"SOPR={sopr:.3f} spends={metrics['spend_count']} (no price data)")
            continue

        price_change = (price_later - price_at) / price_at

        # Generate signal
        if sopr < 0.95:
            signal = "LONG"
            correct = price_change > 0
            signals.append({
                'sopr': sopr,
                'price_change': price_change,
                'correct': correct
            })
            color = '\033[92m' if correct else '\033[91m'
            status = "CORRECT" if correct else "WRONG"
        elif sopr > 1.05:
            signal = "---"  # Could be SHORT but we're focused on LONG
            color = '\033[0m'
            status = ""
            correct = None
        else:
            signal = "neutral"
            color = '\033[0m'
            status = ""
            correct = None

        reset = '\033[0m'
        print(f"{color}SOPR={sopr:.3f} spends={metrics['spend_count']} | "
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
            print(f"  SOPR={s['sopr']:.3f} → price {s['price_change']*100:+.2f}% [{status}]")
    else:
        print("No LONG signals generated (SOPR >= 0.95 in all windows)")
        print("This means sellers were NOT capitulating - no strong buy signal.")

if __name__ == '__main__':
    main()
