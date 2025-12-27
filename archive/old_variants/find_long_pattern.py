#!/usr/bin/env python3
"""
FIND LONG PATTERN - Pure Mathematical Analysis
===============================================
Analyze blockchain data to find what ACTUALLY precedes price going UP.

Method:
1. Get blocks where price went UP significantly (>0.5% in 4 hours)
2. Get blocks where price went DOWN significantly
3. Analyze the on-chain patterns in each
4. Find discriminating features

We have the data. We need to find the math.
"""

import json
import sqlite3
import urllib.request
import base64
from datetime import datetime
from collections import defaultdict
import statistics

# RPC config
RPC_USER = "bitcoin"
RPC_PASS = "bitcoin123secure"
RPC_HOST = "127.0.0.1"
RPC_PORT = 8332
DB_PATH = "/root/sovereign/walletexplorer_addresses.db"

auth = base64.b64encode(f"{RPC_USER}:{RPC_PASS}".encode()).decode()

def rpc(method, params=None):
    payload = json.dumps({
        "jsonrpc": "1.0", "id": "research", "method": method, "params": params or []
    }).encode()
    try:
        req = urllib.request.Request(f"http://{RPC_HOST}:{RPC_PORT}")
        req.add_header("Authorization", f"Basic {auth}")
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, payload, timeout=60) as resp:
            return json.loads(resp.read()).get('result')
    except Exception as e:
        print(f"RPC error: {e}")
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

def analyze_block_window(start_height, window_size, exchange_addrs):
    """
    Analyze a window of blocks and extract ALL on-chain metrics.

    Returns dict with metrics that might predict price movement.
    """
    metrics = {
        # Flow metrics
        'total_inflow_btc': 0,
        'total_outflow_btc': 0,
        'net_flow_btc': 0,
        'inflow_count': 0,
        'outflow_count': 0,

        # Size metrics
        'avg_inflow_size': 0,
        'avg_outflow_size': 0,
        'max_inflow': 0,
        'max_outflow': 0,

        # Whale metrics (>10 BTC)
        'whale_inflows': 0,
        'whale_outflows': 0,
        'whale_inflow_btc': 0,
        'whale_outflow_btc': 0,

        # Exchange breakdown
        'exchanges_with_inflow': set(),
        'exchanges_with_outflow': set(),

        # Transaction patterns
        'consolidation_txs': 0,  # Many inputs → 1 output
        'fanout_txs': 0,         # 1 input → many outputs
        'exchange_to_exchange': 0,

        # Timing
        'block_timestamp': 0,

        # Non-exchange metrics
        'total_btc_moved': 0,
        'total_tx_count': 0,
        'avg_tx_size': 0,

        # UTXO age (if we can detect)
        'old_coins_moved': 0,  # Coins > 1 year old
    }

    inflow_sizes = []
    outflow_sizes = []
    tx_sizes = []

    for h in range(start_height, start_height + window_size):
        block_hash = rpc('getblockhash', [h])
        if not block_hash:
            continue

        block = rpc('getblock', [block_hash, 3])
        if not block:
            continue

        if metrics['block_timestamp'] == 0:
            metrics['block_timestamp'] = block.get('time', 0)

        for tx in block.get('tx', []):
            tx_value = 0
            tx_inputs = []
            tx_outputs = []

            exchange_inputs = defaultdict(float)
            exchange_outputs = defaultdict(float)

            # Analyze inputs
            for vin in tx.get('vin', []):
                if 'coinbase' in vin:
                    continue
                prevout = vin.get('prevout', {})
                if prevout:
                    addr = prevout.get('scriptPubKey', {}).get('address')
                    value = prevout.get('value', 0)
                    tx_value += value
                    tx_inputs.append((addr, value))

                    if addr and addr in exchange_addrs:
                        ex = exchange_addrs[addr]
                        exchange_inputs[ex] += value

            # Analyze outputs
            for vout in tx.get('vout', []):
                addr = vout.get('scriptPubKey', {}).get('address')
                value = vout.get('value', 0)
                tx_outputs.append((addr, value))

                if addr and addr in exchange_addrs:
                    ex = exchange_addrs[addr]
                    exchange_outputs[ex] += value

            # Classify flows
            for ex, val in exchange_outputs.items():
                if ex not in exchange_inputs:  # True inflow (not internal)
                    metrics['total_inflow_btc'] += val
                    metrics['inflow_count'] += 1
                    metrics['exchanges_with_inflow'].add(ex)
                    inflow_sizes.append(val)
                    if val >= metrics['max_inflow']:
                        metrics['max_inflow'] = val
                    if val >= 10:
                        metrics['whale_inflows'] += 1
                        metrics['whale_inflow_btc'] += val

            for ex, val in exchange_inputs.items():
                if ex not in exchange_outputs:  # True outflow (not internal)
                    metrics['total_outflow_btc'] += val
                    metrics['outflow_count'] += 1
                    metrics['exchanges_with_outflow'].add(ex)
                    outflow_sizes.append(val)
                    if val >= metrics['max_outflow']:
                        metrics['max_outflow'] = val
                    if val >= 10:
                        metrics['whale_outflows'] += 1
                        metrics['whale_outflow_btc'] += val

            # Exchange to exchange
            if exchange_inputs and exchange_outputs:
                metrics['exchange_to_exchange'] += 1

            # Transaction patterns
            if len(tx_inputs) >= 5 and len(tx_outputs) <= 2:
                metrics['consolidation_txs'] += 1
            if len(tx_inputs) <= 2 and len(tx_outputs) >= 5:
                metrics['fanout_txs'] += 1

            # General metrics
            if tx_value > 0:
                tx_sizes.append(tx_value)
                metrics['total_btc_moved'] += tx_value
                metrics['total_tx_count'] += 1

    # Calculate averages
    metrics['net_flow_btc'] = metrics['total_outflow_btc'] - metrics['total_inflow_btc']

    if inflow_sizes:
        metrics['avg_inflow_size'] = sum(inflow_sizes) / len(inflow_sizes)
    if outflow_sizes:
        metrics['avg_outflow_size'] = sum(outflow_sizes) / len(outflow_sizes)
    if tx_sizes:
        metrics['avg_tx_size'] = sum(tx_sizes) / len(tx_sizes)

    # Convert sets to counts
    metrics['exchanges_with_inflow'] = len(metrics['exchanges_with_inflow'])
    metrics['exchanges_with_outflow'] = len(metrics['exchanges_with_outflow'])

    return metrics

def main():
    print("=" * 70)
    print("FIND LONG PATTERN - Mathematical Research")
    print("=" * 70)
    print()

    exchange_addrs = load_exchange_addresses()

    height = rpc('getblockcount')
    if not height:
        print("Failed to get block height")
        return

    print(f"Current block: {height:,}")
    print()

    # Analyze windows and categorize by price movement
    window_size = 50
    verify_hours = 4
    blocks_per_hour = 6
    buffer = verify_hours * blocks_per_hour + 10

    # Go back further in time
    num_windows = 100
    start = height - buffer - (num_windows * window_size)

    print(f"Analyzing {num_windows} windows of {window_size} blocks each")
    print(f"From block {start:,} to {height - buffer:,}")
    print()

    up_windows = []    # Price went UP
    down_windows = []  # Price went DOWN

    for i in range(num_windows):
        window_start = start + (i * window_size)

        # Get metrics
        metrics = analyze_block_window(window_start, window_size, exchange_addrs)

        if metrics['block_timestamp'] == 0:
            continue

        # Get price change
        signal_ms = metrics['block_timestamp'] * 1000
        later_ms = signal_ms + (verify_hours * 3600 * 1000)

        price_at = get_price(signal_ms)
        price_later = get_price(later_ms)

        if not price_at or not price_later:
            continue

        price_change = (price_later - price_at) / price_at
        metrics['price_change_pct'] = price_change * 100

        if price_change > 0.003:  # >0.3% up
            up_windows.append(metrics)
            marker = "UP"
        elif price_change < -0.003:  # >0.3% down
            down_windows.append(metrics)
            marker = "DOWN"
        else:
            marker = "FLAT"

        net = metrics['net_flow_btc']
        print(f"Window {i+1}: net={net:+.1f} BTC, "
              f"in={metrics['total_inflow_btc']:.1f}, out={metrics['total_outflow_btc']:.1f}, "
              f"price={price_change*100:+.2f}% [{marker}]")

    print()
    print("=" * 70)
    print("ANALYSIS RESULTS")
    print("=" * 70)
    print()
    print(f"Windows where price went UP:   {len(up_windows)}")
    print(f"Windows where price went DOWN: {len(down_windows)}")
    print()

    if not up_windows or not down_windows:
        print("Not enough data")
        return

    # Compare metrics between UP and DOWN windows
    print("COMPARING UP vs DOWN WINDOWS:")
    print("-" * 70)

    metric_keys = [
        'net_flow_btc', 'total_inflow_btc', 'total_outflow_btc',
        'inflow_count', 'outflow_count',
        'avg_inflow_size', 'avg_outflow_size',
        'max_inflow', 'max_outflow',
        'whale_inflows', 'whale_outflows',
        'whale_inflow_btc', 'whale_outflow_btc',
        'consolidation_txs', 'fanout_txs',
        'exchange_to_exchange',
        'total_btc_moved', 'total_tx_count', 'avg_tx_size'
    ]

    discriminating = []

    for key in metric_keys:
        up_vals = [w[key] for w in up_windows]
        down_vals = [w[key] for w in down_windows]

        up_avg = sum(up_vals) / len(up_vals) if up_vals else 0
        down_avg = sum(down_vals) / len(down_vals) if down_vals else 0

        # Calculate difference
        if down_avg != 0:
            diff_pct = ((up_avg - down_avg) / abs(down_avg)) * 100
        else:
            diff_pct = 0

        # Is this metric discriminating?
        if abs(diff_pct) > 20:
            discriminating.append((key, up_avg, down_avg, diff_pct))
            marker = "***"
        else:
            marker = ""

        print(f"{key:25} | UP: {up_avg:10.2f} | DOWN: {down_avg:10.2f} | diff: {diff_pct:+6.1f}% {marker}")

    print()
    print("=" * 70)
    print("DISCRIMINATING FEATURES (>20% difference):")
    print("=" * 70)
    for key, up_avg, down_avg, diff_pct in sorted(discriminating, key=lambda x: -abs(x[3])):
        direction = "HIGHER when UP" if diff_pct > 0 else "LOWER when UP"
        print(f"  {key}: {direction} ({diff_pct:+.1f}%)")

    print()
    print("=" * 70)
    print("POTENTIAL LONG FORMULAS:")
    print("=" * 70)

    # Based on the discriminating features, suggest formulas
    for key, up_avg, down_avg, diff_pct in sorted(discriminating, key=lambda x: -abs(x[3])):
        if diff_pct > 0:
            print(f"  IF {key} > {(up_avg + down_avg)/2:.2f} THEN LONG")
        else:
            print(f"  IF {key} < {(up_avg + down_avg)/2:.2f} THEN LONG")

if __name__ == '__main__':
    main()
